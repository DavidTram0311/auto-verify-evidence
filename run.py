import ultralytics
from ultralytics import YOLO
from PIL import Image, ImageFile
import requests
from io import BytesIO
import pandas as pd
ImageFile.LOAD_TRUNCATED_IMAGES = True

# v4: 0: 'awb', 1: 'awb-njv', 2: 'weighing-platform-bulky', 3: 'weighing-platform-small'
def unique(numbers):

    list_of_unique_numbers = []

    unique_numbers = set(numbers)

    for number in unique_numbers:
        list_of_unique_numbers.append(number)

    return list_of_unique_numbers

def predicted(df,model4, model5):
    ids = []
    predicted_value = []
    conf_interval_before4 = []
    conf_interval_before5 = []
    class_pred4 = []
    class_pred5 = []
    picture = []

    reason = []

    for i in range(len(df)):
        url = df['Picture'][i]
        url = f'https://{url}'
        picture.append(url)
        response = requests.get(url)
        source = Image.open(BytesIO(response.content))
        results4 = model4(source)
        results5 = model5(source)
        n = []

        class_pred4.append(results4[0].boxes.cls)
        class_pred5.append(results5[0].boxes.cls)
        conf_interval_before4.append(results4[0].boxes.conf)
        conf_interval_before5.append(results5[0].boxes.conf)

        if str(results4[0].boxes.cls) == 'tensor([])':
            predicted_value.append('fail')
            reason.append('no awb or strange stuff on weighing platform')
    
        else:
            for box in results4[0].boxes:
                cls = box.cls

                # Do not accept awb and weighing platform of confident interval of awb and weighing
                # platform less than 64% and 70% respectively.
                if box.conf > float(0.26) and int(cls) == 0:
                    n.append(int(cls))

                elif box.conf > float(0.35) and int(cls) == 2:
                    n.append(int(cls))

                elif box.conf > float(0.35) and int(cls) == 3:
                    n.append(int(cls))
                
                elif box.conf > float(0.50) and int(cls) == 1:
                    n.append(int(cls))
            
            for box in results5[0].boxes:
                cls = box.cls

                # Do not accept awb and weighing platform of confident interval of awb and weighing
                # platform less than 64% and 70% respectively.
                if box.conf > float(0.26) and int(cls) == 0:
                    n.append(int(cls))

                elif box.conf > float(0.35) and int(cls) == 2:
                    n.append(int(cls))

                elif box.conf > float(0.35) and int(cls) == 3:
                    n.append(int(cls))
                
                elif box.conf > float(0.50) and int(cls) == 1:
                    n.append(int(cls))


            # Check valid or invalid
            if unique(n) == [0]:
                reason.append('strange stuffs on weighing platform')

            elif unique(n) == [1]:
                reason.append('strange stuffs on weighing platform')

            elif unique(n) == [2] or unique(n) == [3]:
                reason.append('no awb')

            else:
                reason.append('')

            if unique(n) == [0, 2] or unique(n) == [0, 3]:
                predicted_value.append('pass')

            elif unique(n) == [1, 2] or unique(n) == [1, 3]: #small
                predicted_value.append('pass')

            else:
                predicted_value.append('fail')
        
        ids.append(df['TID'][i])
    
    cm = pd.DataFrame(list(zip(ids, picture, predicted_value, reason, class_pred4, conf_interval_before4, class_pred5, conf_interval_before5)),
                    columns=['TID', 'Picture', 'Predicted value', 'Fail reason', 'class_1st_model', 'confident-interval_1st_model', 'class_2nd_model', 'confident-interval_2nd_model']) 

    return cm
