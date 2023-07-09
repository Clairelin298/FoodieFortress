

'''
get the predict result from YOLO model.

args:
- img: image file

return:
- predict result (string)
    e.g. 'banana'
'''

from ultralytics import YOLO
from numpy import argmax
import torch
import os
import sys
import re




names1 =  {0: 'apple', 1: 'banana', 2: 'beans', 3: 'carrot', 4: 'chicken', 5: 'milk', 6: 'orangejuice', 7: 'pizza', 8: 'potato', 9: 'rice', 10: 'salad', 11: 'spaghetti'}

names = ["Apple",
    "Banana",
    "Corn",
    "Eggplant",
    "Guava",
    "Lemon",
    "Orange",
    "Peach",
    "Pear",
    "Strawberry"]

def parse_res(results):
    # Assuming 'result' is a list of Results objects
    if len(results) > 0:
        highest_prob_label = None
        highest_prob = 0.0

        for res in results:
            pred_labels = res.boxes.cls
            if len(pred_labels) > 0:
                pred_probs = res.boxes.conf
                max_prob_idx = torch.argmax(pred_probs)
                # max_prob_idx = pred_probs.index(max(pred_probs))
                if pred_probs[max_prob_idx] > highest_prob:
                    highest_prob = pred_probs[max_prob_idx]
                    highest_prob_label = pred_labels[max_prob_idx]

        if highest_prob_label is not None:
            print("Highest probability label:", highest_prob_label)
            return highest_prob_label
        else:
            print("No objects detected.")
            return -1
    else:
        print("Empty result.")
        return -1
    
def predict(img):
    ''' 
    initialize both model path and image path
    '''
    model_path = 'app/cls_weight/cls_best_v3.pt'
    # image = 'banana_multiple.jpg'
    image_path = img
    
    '''
    load the model and predict
    '''
    model = YOLO(model_path)
    # model = YOLO('yolov8n-cls.pt')
    results = model(image_path)

    '''
    get the highest prediction value index, and reference the names dictionary
    results[0] 會用到的兩個參數

    - results[0].names : 基本上就是全部的class，形式大概會像這樣
        names: {0: 'apple', 1: 'banana', 2: 'beans', 3: 'carrot', 4: 'chicken', 5: 'milk', 6: 'orangejuice', 7: 'pizza', 8: 'potato', 9: 'rice', 10: 'salad', 11: 'spaghetti'}

    - results[0].probs : 每個class的機率，大概像這樣
        probs: tensor([2.8797e-07, 9.9694e-01, 7.5722e-06, 2.1525e-06, 5.7474e-06, 4.6163e-05, 2.6578e-03, 2.7533e-07, 2.3013e-04, 2.2348e-07, 1.0754e-04, 8.0169e-07], device='cuda:0')
    '''
   
    print(f'{"="*60}')
    print(results[0].probs)
    print(f'{"="*60}')
    print(results[0].probs.data)
    
    print(f'{"="*60}')
    print(torch.argmax(results[0].probs.data))
    print(f'{"="*60}')

    # print(results[0].probs.get_max_prob())
    # print(f'{"="*60}')
    # index = torch.argmax(results[0].probs)
    # index = argmax(results[0].probs.data.cpu()).item()
    index = torch.argmax(results[0].probs.data).item()
    predict_result = results[0].names[index]

    return predict_result
def take_pic():

    os.system("fswebcam -d /dev/video0 --no-banner -r 1080x720 image.jpg")
    return predict("image.jpg")


# print(predict('test_images/banana17_jpg.rf.b8f382a0fd500bcd47a12b4cb5a73de2.jpg'))
# print(predict('image.jpg'))


print(take_pic())