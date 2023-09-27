import ultralytics
from ultralytics import YOLO
from PIL import Image, ImageFile
import requests
from io import BytesIO
import pandas as pd
import time
import functions
ImageFile.LOAD_TRUNCATED_IMAGES = True
pd.options.display.max_columns = 21

# v3: 0: 'awb', 1: 'awb-njv', 2: 'weighing-platform-bulky', 3: 'weighing-platform-small'

def predicted(df,model3):
    awb = []
    ci_awb = []
    platform = []
    ci_platform = []
    small = [] 
    ci_small = []
    bulky = []
    ci_bulky = []
    ids = []
    predicted_value = []
    picture = []
    reason = []
    count = 0

    for i in range(len(df)):

        time.sleep(2)
        count = count + 1
        print(count)
        functions.stop(count)
        url = df.iloc[i,1]
        url = f'http://{url}'
        picture.append(url)
        response = requests.get(url, timeout=30)
        source = Image.open(BytesIO(response.content))
        results = model3(source)
        # results5 = model5(source)
        n = []
        cls4 = []
        ci_cls4 = []

        # Predict Fail or Pass
        if str(results[0].boxes.cls) == 'tensor([])':
            predicted_value.append('fail')
            reason.append('error/invisible')
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
        print(model4_dict)

                
        if model4_dict[0] > float(0):
            awb.append(1)
            ci_awb.append(model4_dict[0])
        else:
            awb.append(0)
            ci_awb.append(model4_dict[0])

        if model4_dict[1] > float(0):
            platform.append(1)
            ci_platform.append(model4_dict[1])
        else:
            platform.append(0)
            ci_platform.append(model4_dict[1])

        if model4_dict[2] > float(0):
            platform.append(1)
            ci_platform.append(model4_dict[2])
        else:
            platform.append(0)
            ci_platform.append(model4_dict[2])
        
        ids.append(df.iloc[i,0])

    cm = pd.DataFrame(list(zip(ids, picture, predicted_value, reason, awb, ci_awb, platform, ci_platform)),
                columns=['TID', 'Picture', 'Predicted Value', 'Fail Reason', 'AWB', 'CI of AWB', 
                         'Weighing Platform', 'CI of Weighing Platform'])
    return cm

def recommend(pred, ci_awb, platform, ci_platform):
    if pred == 'pass':
        if ci_awb < float(0.60):
            return 1
        elif platform == 1 and ci_platform < float(0.956):
            return 1
        else:
            return 0
    
    elif pred == 'fail':
        return 1
    

