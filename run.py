import ultralytics
from ultralytics import YOLO
from PIL import Image, ImageFile
import requests
from io import BytesIO
import pandas as pd
import time
import functions
import os
ImageFile.LOAD_TRUNCATED_IMAGES = True
pd.options.display.max_columns = 21

# v3: 0: 'awb', 1: 'awb-njv', 2: 'weighing-platform-bulky', 3: 'weighing-platform-small'

def predicted_lzd(df,model3):
    awb = []
    ci_awb = []
    platform = []
    ci_platform = []
    ids = []
    predicted_value = []
    picture = []
    reason = []
    count = 0

    try:
        for i in range(len(df)):
    
            time.sleep(1)
            count = count + 1
            print(count)
            functions.stop(count)
            url = df['Picture'][i]
            url = f'https://{url}'
            picture.append(url)
            response = requests.get(url, verify=False, timeout=30)
            source = Image.open(BytesIO(response.content))
            results = model3(source)
            # results5 = model5(source)
            n = []
            cls4 = []
            ci_cls4 = []
    
            # Predict Fail or Pass
            if str(results[0].boxes.cls) == 'tensor([])':
                predicted_value.append('fail')
                reason.append('invisible')
            else:
                for box in results[0].boxes:
                    cls = box.cls
    
                    # Do not accept awb and weighing platform of confident interval of awb and weighing
                    # platform less than 64% and 70% respectively.
                    n.append(int(cls))
    
                if functions.unique(n) == [0]:
                    reason.append('strange stuffs')
    
                elif functions.unique(n) == [1] or functions.unique(n) == [2]:
                    reason.append('no awb')
    
                else:
                    reason.append('')
    
                if functions.unique(n) == [0, 1] or functions.unique(n) == [0, 2]:
                    predicted_value.append('pass')
                
                elif functions.unique(n) == [0,1,2]:
                    predicted_value.append('pass')
    
                else:
                    predicted_value.append('fail')
    
            # Input class and confident interval
            for box in results[0].boxes:
                cls = box.cls
                conf = box.conf
    
                cls4.append(int(cls))
                ci_cls4.append(float(conf))
                    
            model4_dict = functions.m_dict_3(cls4, ci_cls4)
            model4_dict = functions.transform_dict_3(model4_dict)
    
    
                    
            if model4_dict[0] > float(0):
                awb.append(1)
                ci_awb.append(model4_dict[0])
            else:
                awb.append(0)
                ci_awb.append(model4_dict[0])
    
            if model4_dict[1] > float(0) or model4_dict[2] > float(0):
                platform.append(1)
                if model4_dict[1] > float(0):
                    ci_platform.append(model4_dict[1])
                elif model4_dict[2] > float(0):
                    ci_platform.append(model4_dict[2])
            else:
                platform.append(0)
                ci_platform.append(0.0)
    
            
            ids.append(df['TID'][i])
    
        cm = pd.DataFrame(list(zip(ids, picture, predicted_value, reason, awb, ci_awb, platform, ci_platform)),
                    columns=['TID', 'Picture', 'Predicted_Value', 'Fail_Reason', 'AWB', 'Conf_AWB', 
                             'Weighing_Platform', 'Conf_Weighing_Platform'])
        return cm
    except:
        print('These is something wrong')
        print(f'The program stopped at row {count-2} (ID: {df.iloc[i-1,0]})')
        print('Result have saved')

        cm = pd.DataFrame(list(zip(ids, picture, predicted_value, reason, awb, ci_awb, platform, ci_platform)),
                    columns=['TID', 'Picture', 'Predicted_Value', 'Fail_Reason', 'AWB', 'Conf_AWB', 
                             'Weighing_Platform', 'Conf_Weighing_Platform'])
        return cm
        

def recommend_lzd(pred, ci_awb, platform, ci_platform):
    if pred == 'pass':
        if ci_awb < float(0.60):
            return 1
        elif platform == 1 and ci_platform < float(0.956):
            return 1
        else:
            return 0
    
    elif pred == 'fail':
        return 1

#-----------------------------------------------------------------------------------------------------#
# Non - partnership
def predicted_npsp(folder_pth, model):
    platform = [] 
    orders = []
    ci_platform = []
    ids = []
    predicted_value = []
    reason = []
    for i in os.listdir(folder_pth):
        id_path = f'{folder_pth}/{i}'
        print(id_path)

        for j in os.listdir(id_path):
            name = j.split('.')[0]
            length = len(name.split('_'))
            id = name.split('_')[0]
            order = name.split('_')[length-1]
            source = f'{id_path}/{j}'
            results = model(source)
            cls4 = []
            ci_cls4 = []

            # Predict Fail or Pass
            if str(results[0].boxes.cls) == 'tensor([])':
                reason.append('invisible')
                predicted_value.append('fail')
                

            else:
                for box in results[0].boxes:
                    cls = box.cls
                    ci = box.conf
                # Do not accept awb and weighing platform of confident interval of awb and weighing
                # platform less than 64% and 70% respectively.
                    cls4.append(int(cls))
                    ci_cls4.append(float(ci))

                if functions.unique(cls4) == [0]:
                    reason.append('strange stuffs')
                elif functions.unique(cls4) == [1]:
                    reason.append('strange stuffs')
                else:
                    reason.append('')

                if functions.unique(cls4) == [0, 2] or functions.unique(cls4) == [0, 3]:
                    predicted_value.append('pass')

                elif functions.unique(cls4) == [2] or functions.unique(cls4) == [3]:
                    predicted_value.append('pass')

                else:
                    predicted_value.append('fail')

                # Input class and confident interval

            model4_dict = functions.m_dict_5(cls4, ci_cls4)
            model4_dict = functions.transform_dict_5(model4_dict)


            if model4_dict[2] > float(0) or model4_dict[3] > float(0):
                platform.append(1)
                if model4_dict[2] > float(0):
                    ci_platform.append(model4_dict[2])
                elif model4_dict[3] > float(0):
                    ci_platform.append(model4_dict[3])
            else:
                platform.append(0)
                ci_platform.append(0.0)
            ids.append(id)
            orders.append(order)
            print(reason)

    cm = pd.DataFrame(list(zip(ids, orders, predicted_value, reason, platform, ci_platform)),
            columns=['TID', 'Photo_ID', 'Predicted_Value', 'Fail_Reason', 
                    'Weighing_Platform', 'Conf_Weighing_Platform'])
    return cm

def recommend_npsp(pred, platform, ci_platform):
    if pred == 'pass':
        if platform == 1 and ci_platform < float(0.948):
            return 1
        else:
            return 0
    
    elif pred == 'fail':
        return 1
    

