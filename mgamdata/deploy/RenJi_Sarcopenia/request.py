'''
    这个脚本会用来向服务提交推理申请。
'''
import os
import requests

if __name__ == '__main__':
    # series_UIDs = ['1.2.840.113704.7.32.0.416.18871506993714250327819571029699638425',
    #                '1.2.840.113704.7.32.0.416.20167845203280575830090952514270292795',
    #                '1.2.840.113704.7.32.0.416.20309749149274563074103769645782910340',
    #                '1.2.840.113704.7.32.0.416.21223921189765378490639366905956020102',
    #                '1.2.840.113704.7.32.0.416.29757815228586849385905327079529929397',
    #                '1.2.840.113704.7.32.0.416.30399323834273571348168252924658231983',
    #                '1.2.840.113704.7.32.0.416.31802539265517237431565148631045079749',
    #                '1.2.840.113704.7.32.0.416.34245305332787047756620999469674888299',
    #                '1.2.840.113704.7.32.0.416.45923354817579326307964212010910755914',
    #                '1.2.840.113704.7.32.0.416.58672817887134689701659478294588851318',
    #                '1.2.840.113704.7.32.0.416.69238679340241086150369352853796992466',]
    series_UIDs = ['1.2.840.113704.7.32.005.14038013507713.230520145843.3.5184.59974']
    series_root = '/Data/zhangyiqin.sx/Sarcopenia_Data/Test_7986/dcm'
    
    for series_UID in series_UIDs:
        series_path = os.path.join(series_root, series_UID)
        
        data = {
            "method": "ct_sarcopenia_predict",
            "data": {
                    "seriesUid": '1.2.840.113704.7.32.0.416.18871506993714250327819571029699638425',
                    "seriesPath": series_path 
            }
        }
        url = 'http://10.100.39.130:31835/predict/'
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url=url, headers=headers, json=data)
        print(response.text)