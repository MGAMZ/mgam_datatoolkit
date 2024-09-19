'''
    这个脚本会用来向服务提交推理申请。
'''

import requests, json

if __name__ == '__main__':
    series_path = '/fileser51/zhangwh.lw/workspace/projects/renji_sarcopenia/data/first_try/pic_/1.2.840.113704.7.32.0619.2.334.3.2831164355.531.1618186171.215.3'
    data = {
        "method": "ct_sarcopenia_predict",
        "data": {
                "seriesUid": '1.2.840.113704.7.32.0619.2.334.3.2831164355.531.1618186171.215.3',
                "seriesPath":series_path 
        }
    }

    url = 'http://10.100.39.138:32335/predict/'
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url=url, headers=headers, json=data)
    # print(response.text)