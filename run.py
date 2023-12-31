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

# v3: 0: 'awb', 1: 'weighing-platform-bulky', 2: 'weighing-platform-small'

# LZD-GCS
def predicted_lzd_gcs(df,model3):
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
            count = count + 1
            print(count)
            functions.stop(count)
            url = df['Picture'][i]
            url = f'https://{url}'
            picture.append(url)
            response = requests.get(url, verify=False, timeout=30)
            source = Image.open(BytesIO(response.content))
            results = model3(source)
            cls4 = []
            ci_cls4 = []
    
            # Predict Fail or Pass
            if str(results[0].boxes.cls) == 'tensor([])':
                predicted_value.append('fail')
                reason.append('invisible')
            else:
                for box in results[0].boxes:
                    cls = box.cls
                    conf = box.conf

                    # Append class value and confident score of awb and 2 platform 
                    cls4.append(int(cls))
                    ci_cls4.append(float(conf))

                # add reason of fail
                if functions.unique(cls4) == [0]:
                    reason.append('strange stuffs')
    
                elif functions.unique(cls4) == [1] or functions.unique(cls4) == [2]:
                    reason.append('no awb')
    
                else:
                    reason.append('')

                # add predict value
                if functions.unique(cls4) == [0, 1] or functions.unique(cls4) == [0, 2]:
                    predicted_value.append('pass')
                
                elif functions.unique(cls4) == [0,1,2]:
                    predicted_value.append('pass')
    
                else:
                    predicted_value.append('fail')
    
            # Input class and confident score  
            model4_dict = functions.m_dict_3(cls4, ci_cls4)

            # Add appearance value (0: no or 1: yes) and confident score of weighing platform 
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

        # Recheck Recommendation
        cm['Recheck_needed'] = [functions.recommend_lzd(pred, ci_awb, platform, ci_platform)
                                      for pred, ci_awb, platform, ci_platform in zip(cm['Predicted_Value'], cm['Conf_AWB'],
                                                                                      cm['Weighing_Platform'], cm['Conf_Weighing_Platform'])]

        column_to_move = cm.pop("Recheck_needed")
        cm.insert(4, "Recheck_needed", column_to_move)
        
        return cm
    except:
        print('These is something wrong')
        print(f'The program stopped at row {count-2} (ID: {df.iloc[i-1,0]})')
        print('Result have saved')

        cm = pd.DataFrame(list(zip(ids, picture, predicted_value, reason, awb, ci_awb, platform, ci_platform)),
                    columns=['TID', 'Picture', 'Predicted_Value', 'Fail_Reason', 'AWB', 'Conf_AWB', 
                             'Weighing_Platform', 'Conf_Weighing_Platform'])

        # Recheck Recommendation
        cm['Recheck_needed'] = [functions.recommend_lzd(pred, ci_awb, platform, ci_platform)
                                      for pred, ci_awb, platform, ci_platform in zip(cm['Predicted_Value'], cm['Conf_AWB'],
                                                                                      cm['Weighing_Platform'], cm['Conf_Weighing_Platform'])]

        column_to_move = cm.pop("Recheck_needed")
        cm.insert(4, "Recheck_needed", column_to_move)
        
        return cm
        

#-----------------------------------------------------------------------------------------------------#
# LZD - Drive

def predicted_lzd_drive(folder_pth, model):
    awb = []
    ci_awb = []
    platform = []
    ci_platform = []
    ids = []
    predicted_value = []
    reason = []
    orders = []

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
                predicted_value.append('fail')
                reason.append('error/invisible')
            else:
                for box in results[0].boxes:
                    cls = box.cls
                    ci = box.conf
                    # Append class value and confident score of awb and 2 platform
                    cls4.append(int(cls))
                    ci_cls4.append(float(ci))

                # add reason of fail
                if functions.unique(cls4) == [0]:
                    reason.append('strange stuffs')

                elif functions.unique(cls4) == [1] or functions.unique(cls4) == [2]:
                    reason.append('no awb')

                else:
                    reason.append('')

                # add predict value
                if functions.unique(cls4) == [0, 1] or functions.unique(cls4) == [0, 2]:
                    predicted_value.append('pass')
                
                elif functions.unique(cls4) == [0,1,2]:
                    predicted_value.append('pass')

                else:
                    predicted_value.append('fail')

            # Input class and confident score                   
            model4_dict = functions.m_dict_3(cls4, ci_cls4)
 
            if model4_dict[0] > float(0):
                awb.append(1)
                ci_awb.append(model4_dict[0])
            else:
                awb.append(0)
                ci_awb.append(model4_dict[0])

            # Add appearance value (0: no or 1: yes) and confident score of weighing platform and awb
            if model4_dict[1] > float(0) or model4_dict[2] > float(0):
                platform.append(1)
                if model4_dict[1] > float(0):
                    ci_platform.append(model4_dict[1])
                elif model4_dict[2] > float(0):
                    ci_platform.append(model4_dict[2])
            else:
                platform.append(0)
                ci_platform.append(0.0)
  
            ids.append(id)
            orders.append(order)

    cm = pd.DataFrame(list(zip(ids, orders, predicted_value, reason, awb, ci_awb, platform, ci_platform)),
                columns=['TID', 'Photo_ID', 'Predicted_Value', 'Fail_Reason', 'AWB', 'Conf_AWB', 
                        'Weighing_Platform', 'Conf_Weighing_Platform'])
    
    # Recheck Recommendation
    cm['Recheck_needed'] = [functions.recommend_lzd(pred, ci_awb, platform, ci_platform)
                                  for pred, ci_awb, platform, ci_platform in zip(cm['Predicted_Value'], cm['Conf_AWB'],
                                                                                  cm['Weighing_Platform'], cm['Conf_Weighing_Platform'])]

    column_to_move = cm.pop("Recheck_needed")
    cm.insert(4, "Recheck_needed", column_to_move)
    
    return cm
    

#-----------------------------------------------------------------------------------------------------#
# Non - partnership drive    
def predicted_npsp_drive(folder_pth, model):
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

                    cls4.append(int(cls))
                    ci_cls4.append(float(ci))

                # add reason of fail
                if functions.unique(cls4) == [0]:
                    reason.append('strange stuffs')
                else:
                    reason.append('')

                # add predict value
                if functions.unique(cls4) == [0, 1] or functions.unique(cls4) == [0, 2]:
                    predicted_value.append('pass')

                elif functions.unique(cls4) == [1] or functions.unique(cls4) == [2]:
                    predicted_value.append('pass')

                else:
                    predicted_value.append('fail')

            # Input class and confident score
            model4_dict = functions.m_dict_3(cls4, ci_cls4)

            # Add appearance value (0: no or 1: yes) and confident score of weighing platform
            if model4_dict[1] > float(0) or model4_dict[2] > float(0):
                platform.append(1)
                if model4_dict[1] > float(0):
                    ci_platform.append(model4_dict[1])
                elif model4_dict[2] > float(0):
                    ci_platform.append(model4_dict[2])
            else:
                platform.append(0)
                ci_platform.append(0.0)
            ids.append(id)
            orders.append(order)

    cm = pd.DataFrame(list(zip(ids, orders, predicted_value, reason, platform, ci_platform)),
            columns=['TID', 'Photo_ID', 'Predicted_Value', 'Fail_Reason', 
                    'Weighing_Platform', 'Conf_Weighing_Platform'])
    
    cm['Recheck_needed'] = [functions.recommend_npsp(pred, platform, ci_platform)
                                          for pred, platform, ci_platform in zip(cm['Predicted_Value'], cm['Weighing_Platform'],
                                                                                         cm['Conf_Weighing_Platform'])]
    
    column_to_move = cm.pop("Recheck_needed")
    cm.insert(4, "Recheck_needed", column_to_move)
    
    return cm


#-----------------------------------------------------------------------------------------------------#
# Non partnership GSC URL
def predicted_npsp_gcs(df, model5):
    platform = [] 
    ci_platform = []
    ids = []
    predicted_value = []
    reason = []
    picture = []
    count = 0
    try:

        for i in range(len(df)):
            count = count + 1
            functions.stop(count)
            url = df.iloc[i,1]
            url = f'http://{url}'

            picture.append(url)
            response = requests.get(url, timeout=30)
            source = Image.open(BytesIO(response.content))
            results = model5(source)
            # results5 = model5(source)
            cls4 = []
            ci_cls4 = []

            # Predict Fail or Pass
            if str(results[0].boxes.cls) == 'tensor([])':
                predicted_value.append('fail')
                reason.append('error/invisible')

            else:
                for box in results[0].boxes:
                    cls = box.cls
                    ci = box.conf
                    cls4.append(int(cls))
                    ci_cls4.append(float(ci))

                # add reason of fail
                if functions.unique(cls4) == [0]:
                    reason.append('strange stuffs')
                else:
                    reason.append('')

                # add predict value
                if functions.unique(cls4) == [0, 1] or functions.unique(cls4) == [0, 2]:
                    predicted_value.append('pass')

                elif functions.unique(cls4) == [1] or functions.unique(cls4) == [2]:
                    predicted_value.append('pass')

                else:
                    predicted_value.append('fail')

            # Input class and confident score
            model4_dict = functions.m_dict_3(cls4, ci_cls4)

            # Add appearance value (0: no or 1: yes) and confident score of weighing platform 
            if model4_dict[1] > float(0) or model4_dict[2] > float(0):
                platform.append(1)
                if model4_dict[1] > float(0):
                    ci_platform.append(model4_dict[1])
                elif model4_dict[2] > float(0):
                    ci_platform.append(model4_dict[2])
            else:
                platform.append(0)
                ci_platform.append(0.0)

            ids.append(df.iloc[i,0])
                
        cm = pd.DataFrame(list(zip(ids, picture, predicted_value, reason, platform, ci_platform)),
                columns=['TID', 'Pictures', 'Predicted_Value', 'Fail_Reason', 
                        'Weighing_Platform', 'Conf_Weighing_Platform'])

        # Recheck Recommedation
        cm['Recheck_needed'] = [functions.recommend_npsp(pred, platform, ci_platform)
                                      for pred, platform, ci_platform in zip(cm['Predicted_Value'], cm['Weighing_Platform'],
                                                                                     cm['Conf_Weighing_Platform'])]
    
        column_to_move = cm.pop("Recheck_needed")
        cm.insert(4, "Recheck_needed", column_to_move)
        
        return cm
    
    except:

        print('These is something wrong')
        print(f'The program stopped at row {count-2} (ID: {df.iloc[i-1,0]})')
        print('Result have saved')
        cm = pd.DataFrame(list(zip(ids, picture, predicted_value, reason, platform, ci_platform)),
                columns=['TID', 'Pictures', 'Predicted_Value', 'Fail_Reason', 
                        'Weighing_Platform', 'Conf_Weighing_Platform'])
        
        # Recheck Recommedation
        cm['Recheck_needed'] = [functions.recommend_npsp(pred, platform, ci_platform)
                                      for pred, platform, ci_platform in zip(cm['Predicted_Value'], cm['Weighing_Platform'],
                                                                                     cm['Conf_Weighing_Platform'])]
    
        column_to_move = cm.pop("Recheck_needed")
        cm.insert(4, "Recheck_needed", column_to_move)
        
        return cm
    

