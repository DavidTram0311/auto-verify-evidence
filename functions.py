import time
def unique(numbers):

    list_of_unique_numbers = []

    unique_numbers = set(numbers)

    for number in unique_numbers:
        list_of_unique_numbers.append(number)

    return list_of_unique_numbers

def stop(count):
    if count % 50 == 0:
        time.sleep(30)

# Create dict for model 3 - run for gcs tool
def m_dict_3(cls, ci):
    my_dict = {}
    awb = []
    bulky = []
    small = []

    for i in range(len(cls)):
        if cls[i] == 0:
            awb.append(ci[i])
        elif cls[i] == 1:
            bulky.append(ci[i])
        elif cls[i] == 2:
            small.append(ci[i])

    my_dict[0] = awb
    my_dict[1] = bulky
    my_dict[2] = small
    return my_dict

def average(lst):
    return sum(lst) / len(lst)

def transform_dict_3(my_dict):
    for i in range(0,3):    
        if str(my_dict[i]) == '[]':
            my_dict[i] = [0]

    for i in range(0,3):
        my_dict[i] = average(my_dict[i])
    
    return my_dict

#---------------------------------------------------------------------------------------------------------#

# Create dict for model 5 - run for gcs tool
def m_dict_5(cls, ci):
    my_dict = {}
    awb = []
    bulky = []
    small = []
    awb_njv = []

    for i in range(len(cls)):
        if cls[i] == 0:
            awb.append(ci[i])
        elif cls[i] == 1:
            awb_njv.append(ci[i])
        elif cls[i] == 2:
            bulky.append(ci[i])
        elif cls[i] == 3:
            small.append(ci[i])
    
    my_dict[0] = awb
    my_dict[1] = awb_njv
    my_dict[2] = bulky
    my_dict[3] = small

    return my_dict

def transform_dict_5(my_dict):
    for i in range(0,4):    
        if str(my_dict[i]) == '[]':
            my_dict[i] = [0]

    for i in range(0,4):
        my_dict[i] = average(my_dict[i])
    
    return my_dict
