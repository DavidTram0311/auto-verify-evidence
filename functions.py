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

# Create dict for model
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

    # Transform dict
    for i in range(0,3):    
        if str(my_dict[i]) == '[]':
            my_dict[i] = [0]

    for i in range(0,3):
        my_dict[i] = sum(my_dict[i])/len(my_dict[i])
    
    return my_dict

# Recheck Recommendation of LZD tool
def recommend_lzd(pred, ci_awb, platform, ci_platform):
    if pred == 'pass':
        if ci_awb < float(0.87):
            return 1
        elif platform == 1 and ci_platform < float(0.96):
            return 1
        else:
            return 0
    
    elif pred == 'fail':
        return 1

# Recheck Recommendation of Non-psp tool
def recommend_npsp(pred, platform, ci_platform):
    if pred == 'pass':
        if platform == 1 and ci_platform < float(0.96):
            return 1
        else:
            return 0
    
    elif pred == 'fail':
        return 1



# def transform_dict_3(my_dict):
#     for i in range(0,3):    
#         if str(my_dict[i]) == '[]':
#             my_dict[i] = [0]

#     for i in range(0,3):
#         my_dict[i] = average(my_dict[i])
    
#     return my_dict
