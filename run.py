import ultralytics
from ultralytics import YOLO
from PIL import Image, ImageFile
import requests
from io import BytesIO
import pandas as pd
import time
ImageFile.LOAD_TRUNCATED_IMAGES = True

# v4: 0: 'awb', 1: 'awb-njv', 2: 'weighing-platform-bulky', 3: 'weighing-platform-small'
def unique(numbers):

    list_of_unique_numbers = []

    unique_numbers = set(numbers)

    for number in unique_numbers:
        list_of_unique_numbers.append(number)

    return list_of_unique_numbers

def stop(count):
    if count % 50 == 0:
        time.sleep(60)

def predicted(df,model4, model5):
    ids = []
    predicted_value = []
    picture = []
    cls4 = []
    ci_cls4 = []
    cls5 = []
    ci_cls5 = []
    awb4 = []
    awb_njv4 = []
    bulky4 = []
    small4 = []
    ci_awb4 = []
    ci_awb_njv4 = []
    ci_bulky4 = []
    ci_small4 = []

    awb5 = []
    awb_njv5 = []
    bulky5 = []
    small5 = []
    ci_awb5 = []
    ci_awb_njv5 = []
    ci_bulky5 = []
    ci_small5 = []

    reason = []
    count = 0
    for i in range(len(df)):
        time.sleep(2)
        count = count + 1
        print(count)
        stop(count)
        url = df.iloc[i,1]
        url = f'http://{url}'
        picture.append(url)
        response = requests.get(url, timeout=30)
        source = Image.open(BytesIO(response.content))
        results4 = model4(source)
        results5 = model5(source)
        n = []
        cl4 = []
        ci4 = []

        cl5 = []
        ci5 = []

        # Input class and confident interval
        # model 4
        for box in results4[0].boxes:
            cls = box.cls
            conf = box.conf

            cl4.append(int(cls))
            ci4.append(float(conf))

        # model 5
        for box in results5[0].boxes:
            cls = box.cls
            conf = box.conf

            cl5.append(int(cls))
            ci5.append(float(conf))

        cls4.append(cl4)
        ci_cls4.append(ci4)
        cls5.append(cl5)
        ci_cls5.append(ci5)

            
        if str(results4[0].boxes.cls) == 'tensor([])' or str(results5[0].boxes.cls) == 'tensor([])':
            predicted_value.append('fail')
            reason.append('error/invisible')
    
        else:
            for box in results4[0].boxes:
                cls = box.cls

                # Do not accept awb and weighing platform of confident interval of awb and weighing
                # platform less than 64% and 70% respectively.
                if box.conf > float(0.26) and int(cls) == 0:
                    n.append(int(cls))

                elif box.conf > float(0.35) and int(cls) == 2:
                    n.append(int(cls))

                elif box.conf > float(0.29) and int(cls) == 3:
                    n.append(int(cls))
                
                elif box.conf > float(0.50) and int(cls) == 1:
                    n.append(int(cls))
            
            for box in results5[0].boxes:
                cls = box.cls

                # Do not accept awb and weighing platform of confident interval of awb and weighing
                # platform less than 64% and 70% respectively.
                if box.conf > float(0.26) and int(cls) == 0:
                    n.append(int(cls))

                elif box.conf > float(0.29) and int(cls) == 2:
                    n.append(int(cls))

                elif box.conf > float(0.29) and int(cls) == 3:
                    n.append(int(cls))
                
                elif box.conf > float(0.50) and int(cls) == 1:
                    n.append(int(cls))


            # Check valid or invalid
            if unique(n) == [0]:
                reason.append('strange stuffs')

            elif unique(n) == [1]:
                reason.append('strange stuffs')

            elif unique(n) == [2] or unique(n) == [3]:
                reason.append('no awb')

            else:
                reason.append('')

            if unique(n) == [0, 2] or unique(n) == [0, 3]:
                predicted_value.append('pass')

            elif unique(n) == [1, 2] or unique(n) == [1, 3]: #small
                predicted_value.append('pass')
            
            elif unique(n) == [0,2,3] or unique(n) == [1,2,3]: 
                predicted_value.append('pass')

            else:
                predicted_value.append('fail')
        
        ids.append(df.iloc[i,0])
    
    # cm = pd.DataFrame(list(zip(ids, picture, predicted_value, reason, awb4, ci_awb4, awb5, ci_awb5, awb_njv4, ci_awb_njv4, 
    #                            awb_njv5, ci_awb_njv5, small4, ci_awb4, small5, ci_small5, bulky4, ci_bulky4, bulky5, ci_bulky5)),
    #                 columns=['TID', 'Picture', 'Predicted value', 'Fail reason', 'awb1', 'ci_awb1', 'awb2', 'ci_awb2', 'awb_njv1',
    #                           'ci_awb_njv1', 'awb_njv2', 'ci_awb_njv2', 'small1', 'ci_small1', 'small2', 'ci_small2', 'bulky1', 'ci_bulky1',
    #                           'bulky2', 'ci_bulky2']) 

    cm = pd.DataFrame(list(zip(ids, picture, predicted_value, reason, cls4, ci_cls4, cls5, ci_cls5)),
                columns=['TID', 'Picture', 'Predicted value', 'Fail reason', 'class_model1', 'ci_class_model1', 'class_model2',
                         'ci_class_model2'])

    return cm
