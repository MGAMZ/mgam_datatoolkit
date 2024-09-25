'''
    这个脚本会用来向服务提交推理申请。
'''

import requests

if __name__ == '__main__':
    series_path = '/Data/zhangyiqin.sx/Sarcopenia_Data/Test_7986/dcm/1.2.840.113704.7.32.0.416.18871506993714250327819571029699638425/'
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