import os
import re
import pdb
from time import time
from typing import Optional, Dict, List, Tuple, Union

import numpy as np
import SimpleITK as sitk
from fastapi import FastAPI, Query, HTTPException, Request
from pydantic import BaseModel,Field

from contour import findEachRegion, findContourEachRegion, segment_within_muscular_with_kmeans
from triton_server_client_rengji2d import MedNeXt
import torch



def parse_cfg_file(file_path):
    config = {}
    with open(file_path, 'r') as file:
        for line in file:
            match = re.match(r'(\w+)\s*=\s*"([^"]+)"', line)
            if match:
                key, value = match.groups()
                config[key] = value
    return config



app = FastAPI()



# 定义请求数据的 Pydantic 模型
class PredictionRequest(BaseModel):
    method: str 
    data: Dict[str, str] = Field(..., example={"key1": "value1", "key2": "value1"})



class PredictionResponse(BaseModel):
    # uid: str  # 假设这是用户的唯一标识符
    beginTime: int = 0
    code: int  = 0
    # seriesUid: str = 'NA'
    data: Dict[str, Union[Dict[int, List[Tuple[int, int]]], int, str]] = {}
    endTime: int = 0
    msg: str = ''
    requestId: int =  100



def save_as_mha(source_image:sitk.Image, save_path, mask:np.ndarray):
    mask = mask.astype(np.uint8)
    mask = sitk.GetImageFromArray(mask)
    mask.CopyInformation(source_image)
    sitk.WriteImage(mask, save_path, useCompression=True)
    print(f"saved mask mha file to {save_path}")



def kmeans_segmentation(img, mask_pred):
    print('kmeans segmenting...')
    
    psoas_muscle_nodule_point_set,skeleton_muscle_nodule_point_set,subcutaneous_nodule_point_set = {}, {}, {}
    viscera_nodule_point_set,psoas_fat_nodule_point_set,skeleton_fat_nodule_point_set = {}, {}, {}

    psoas = np.zeros(img.shape)
    skeleton = np.zeros(img.shape)
    psoas_muscle = np.zeros(img.shape)
    psoas_fat = np.zeros(img.shape)
    skeleton_muscle = np.zeros(img.shape)
    skeleton_fat = np.zeros(img.shape)
    subcutaneous = np.zeros(img.shape)
    viscera = np.zeros(img.shape)
    print('mask_pred',mask_pred.shape)

    mask_of_class = mask_pred == 1
    print('mask tag: ', np.unique(mask_pred), len(img[mask_of_class]))
    if len(img[mask_of_class])>1:
        subclass_pixels, subclass_mask = segment_within_muscular_with_kmeans(img, mask_of_class)
        psoas_fat[mask_of_class] = subclass_mask
        psoas_muscle[mask_of_class] = 1-subclass_mask   

        psoas_fat_temp = np.zeros(img.shape)
        psoas_fat_temp[(img>-190)&(img<=-30)]=1
        psoas_fat_temp = psoas_fat_temp*psoas_fat
        mask_pred[psoas_fat_temp>0] = 5

    mask_of_class = mask_pred == 2 
    print('mask tag: ',np.unique(mask_pred),len(img[mask_of_class]))
    if len(img[mask_of_class])>1:
        subclass_pixels, subclass_mask = segment_within_muscular_with_kmeans(img, mask_of_class)
        skeleton_fat[mask_of_class] = subclass_mask
        skeleton_muscle[mask_of_class] = 1-subclass_mask

        skeleton_fat_temp = np.zeros(img.shape)
        skeleton_fat_temp[(img>-190)&(img<=-30)]=1
        skeleton_fat_temp = skeleton_fat_temp*skeleton_fat
        mask_pred[skeleton_fat_temp>0] = 6

    skeleton_fat = np.zeros(img.shape)
    psoas_fat = np.zeros(img.shape)
    
    psoas[mask_pred==1]=1
    skeleton[mask_pred==2]=1
    subcutaneous[mask_pred==3]=1
    viscera[mask_pred==4]=1
    psoas_fat[mask_pred==5]=1
    skeleton_fat[mask_pred==6]=1
    psoas_muscle = (1-psoas_fat)*psoas
    skeleton_muscle = (1-skeleton_fat)*skeleton

    print('sum ',psoas_muscle.sum().sum(), psoas_fat.sum().sum(), skeleton_muscle.sum().sum(), skeleton_fat.sum().sum(), subcutaneous.sum().sum(),viscera.sum().sum() )
    seglist = [psoas_muscle,psoas_fat,skeleton_muscle,skeleton_fat,subcutaneous,viscera]
 
    N = viscera.shape[0]
    # contour_results = {}
    for z in range(N):
        contour_list = findEachRegion(z,seglist)
        # psoas_muscle_nodule_point_set[z],psoas_fat_nodule_point_set[z], skeleton_muscle_nodule_point_set[z],\
        # skeleton_fat_nodule_point_set[z],subcutaneous_nodule_point_set[z], viscera_nodule_point_set[z] = contour_list
        
        if len(contour_list[0])>0:
            psoas_muscle_nodule_point_set[z] = contour_list[0]
        if len(contour_list[1])>0:
            psoas_fat_nodule_point_set[z] = contour_list[1]
        if len(contour_list[2])>0:
            skeleton_muscle_nodule_point_set[z] = contour_list[2]
        if len(contour_list[3])>0:
            skeleton_fat_nodule_point_set[z] = contour_list[3]
        if len(contour_list[4])>0:
            subcutaneous_nodule_point_set[z] = contour_list[4]
        if len(contour_list[5])>0:
            viscera_nodule_point_set[z] = contour_list[5]

    print('Kmeans Segment Done.')
    return mask_pred, [psoas_muscle_nodule_point_set,      # 腰大肌肌肉
                       skeleton_muscle_nodule_point_set,   # 骨骼肌肌肉
                       subcutaneous_nodule_point_set,      # 皮下脂肪
                       viscera_nodule_point_set,           # 内脏脂肪
                       psoas_fat_nodule_point_set,         # 腰大肌脂肪
                       skeleton_fat_nodule_point_set]      # 腰大肌肌肉



def ct_sarcopenia_predict(seriesUid: str, seriesPath: str):
    # 调用函数并打印结果
    config_data = parse_cfg_file('/opt/alpha/config/main_config.cfg')
    IP_adress = config_data['HOST_GPU']+':'+config_data['PORT_GPU']

    # 初始化
    try:
        predictor = MedNeXt(
            model_name = 'renji_seg',
            url = IP_adress,
            input_shape = (1,1,512,512)
        )
    except Exception as e:
        return PredictionResponse(code=1006, msg = f'Network Init Failed: {e}')

    # 推理
    try:
        sitk_image, sitk_pred, pred_logits = predictor.predict_from_dcm(seriesPath)
        img_raw = sitk.GetArrayFromImage(sitk_image)
        mask_pred = sitk.GetArrayFromImage(sitk_pred)
    except Exception as e:
        return PredictionResponse(code=1004, msg = f'Inference Failed: {e}')

    final_pred, kmeans_results = kmeans_segmentation(img_raw, mask_pred)
    PsoasMajor_Muscle = kmeans_results[0]   # 腰大肌肌肉
    PsoasMajor_Fat = kmeans_results[4]      # 腰大肌脂肪
    Skeletal_Muscle = kmeans_results[1]     # 骨骼肌肌肉
    Skeletal_Fat = kmeans_results[5]        # 骨骼肌脂肪
    Subcutaneous_Fat = kmeans_results[2]    # 皮下脂肪
    Visceral_Fat = kmeans_results[3]        # 内脏脂肪
    
    # debug
    # save_as_mha(source_image = sitk_image, 
    #             save_path = f"/fileser/rongkuan/serialization/renji_sarcopenia_2d/{seriesUid}.mha", 
    #             mask = final_pred)

    # print('finish saving')
    
    return (PsoasMajor_Muscle,
            PsoasMajor_Fat,
            Skeletal_Muscle,
            Skeletal_Fat,
            Subcutaneous_Fat,
            Visceral_Fat)


@app.post("/predict")
async def predict(request: PredictionRequest):
    beginTime = round(time())
    if request.method != "ct_sarcopenia_predict":
        raise HTTPException(status_code=400, detail="Invalid method")

    infoDict = request.dict()
    try:
        seriesPath = infoDict['data']['seriesPath']
        seriesUid = infoDict['data']['seriesUid']
    except:
        return PredictionResponse(code=1001, msg = 'para missing')

    if not (isinstance(seriesPath,str) and isinstance(seriesUid,str)):
        return PredictionResponse(code=1002, msg = 'type of param is wrong')

    if not ( (os.path.join(seriesPath)) and (os.path.isdir(seriesPath)) ):
        return PredictionResponse(seriesUid = seriesUid, code=1003, msg = 'image not exist. / no files under dir.')

    print("SeriesPath: ", seriesPath)
    print("SeriesUid: ", seriesUid)
    print("Inferencing...")
    result = ct_sarcopenia_predict(seriesUid, seriesPath)
    if isinstance(result, PredictionResponse):
        print("Fatal Error: ", result)
        return result

    print("Inference Done, Doing Kmeans...")
    psoas_muscle_nodule_point_set,psoas_fat_nodule_point_set,skeleton_muscle_nodule_point_set,skeleton_fat_nodule_point_set,subcutaneous_nodule_point_set,viscera_nodule_point_set = result

    print("Kmeans Done. Return.")
    endTime = round(time())
    print('duration: ',endTime-beginTime)

    final_res = [psoas_muscle_nodule_point_set,skeleton_muscle_nodule_point_set,subcutaneous_nodule_point_set,\
                viscera_nodule_point_set,psoas_fat_nodule_point_set,skeleton_fat_nodule_point_set]
    
    # save_path = f"/fileser/rongkuan/serialization/renji_sarcopenia_2d/{seriesUid}.pth"
    # print('save to ',save_path)
    # torch.save(final_res,save_path)

    # return PredictionResponse()
    return PredictionResponse(
                              beginTime= beginTime,
                            #   code = request_id,
                              data = {"seriesUid": seriesUid,\
                              "psoas_muscle_nodule_point_set": psoas_muscle_nodule_point_set,\
                              "psoas_fat_nodule_point_set": psoas_fat_nodule_point_set,\
                              "skeleton_muscle_nodule_point_set": skeleton_muscle_nodule_point_set,\
                              "skeleton_fat_nodule_point_set": skeleton_fat_nodule_point_set,\
                              "subcutaneous_nodule_point_set": subcutaneous_nodule_point_set,\
                              "viscera_nodule_point_set": viscera_nodule_point_set},
                              endTime = endTime)
                            #   requestId = request_id)




# 运行 FastAPI 应用的代码（通常在主函数中）
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) #8086
