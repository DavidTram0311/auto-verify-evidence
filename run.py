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

def predicted(df,model):
    ids = []
    predicted_value = []
    conf_interval_before = []
    picture = []
    class_pred = []
    reason = []

    for i in range(len(df)):
        url = df['Picture'][i]
        url = f'https://{url}'
        picture.append(url)
        response = requests.get(url)
        source = Image.open(BytesIO(response.content))
        results = model(source)
        n = []
        
        class_pred.append(results[0].boxes.cls)
        conf_interval_before.append(results[0].boxes.conf)

        if str(results[0].boxes.cls) == 'tensor([])':
            predicted_value.append('fail')
            reason.append('no awb or strange stuff on weighing platform')
    
        else:
            for box in results[0].boxes:
                cls = box.cls

                # Do not accept awb and weighing platform of confident interval of awb and weighing
                # platform less than 64% and 70% respectively.
                if box.conf > float(0.64) and int(cls) == 0:
                    n.append(int(cls))

                elif box.conf > float(0.70) and int(cls) == 2:
                    n.append(int(cls))

                elif box.conf > float(0.70) and int(cls) == 3:
                    n.append(int(cls))


            # Check valid or invalid
            if n == [0]:
                reason.append('strange stuffs on weighing platform')

            elif n == [1]:
                reason.append('strange stuffs on weighing platform')

            elif n == [2] or n == [3]:
                reason.append('no awb')
                
            else:
                reason.append('')

            if unique(n) == [0, 2] or unique(n) == [0, 3]:
                predicted_value.append('pass')

            elif unique(n) == [1, 2] and unique(n) == [1, 3]: #small
                predicted_value.append('pass')

            else:
                predicted_value.append('fail')
        
        ids.append(df['TID'][i])
    
    cm = pd.DataFrame(list(zip(ids, picture, predicted_value, reason, class_pred, conf_interval_before)),
                    columns=['TID', 'Picture', 'Predicted value', 'Fail reason', 'class', 'confident interval']) 
    
    return cm
