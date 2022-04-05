'''
Author:     Jixin Jia (Gin)
Date:       2022.03.15
Version:    1.0
Purpose:    This script evaluates model performance by running it against a defined set of test images, 
            and track its performance in Run Experiments 
'''

import cv2
import os
import numpy as np
import base64
import json
from urllib.parse import urlparse
import argparse
import sys

# App utilities
from utils import config
from utils import vehicle_plate as vp
from utils import video_utilities as vu

# experiment tracking with azureml/mlflow
from azureml.core.run import Run
run = Run.get_context()

# get FUSE mount dataset
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data_folder", type=str, default="datasets", help="path to test dataset on FUSE mount")
args = ap.parse_args()

# debug
dataset_path = os.path.join(args.data_folder,'dataset')
print('[GIN] dataset path on remote compute:', dataset_path)

# load trained model (.pb)
try:
    print('Loading vehicle plate detection model...', end='')
    model_path = config.MODEL_PATH
    label_path = config.LABEL_PATH
    
    model = vp.Model(model_path)

    with open(label_path) as f:
        labels = [l.strip() for l in f.readlines()]
    
    print('Done')

except Exception as e:
    print(e.args)
    sys.exit(0)

# create an outputs folder for tracking artifacts
os.makedirs('outputs', exist_ok=True)

# inference on test dataset
image_count = 0
results = []

for f in os.listdir(dataset_path):

    if f.lower().endswith(('jpg','png')):
        
        # output name
        output_name = f'auto-test_{f}'
        output_path = os.path.join('outputs',output_name)

        # read image
        image = cv2.imread(os.path.join(dataset_path, f))

        # detect vehicle plates
        preds = model.predict(image)

        # inspect preds
        for pred in zip(*preds):
            lpBBOX, prob = pred[:2]
            
            if prob > config.LP_CONFIDENCE:
                # get safe bbox
                startX, startY, endX, endY = model.translate_bbox(image, lpBBOX)

                # pack into output list
                results.append({'bbox': (startX, startY, endX, endY), 'prob': prob})

                # draw
                image = vu.draw_bbox_with_label(image, f'{prob:.3f}', (startX, startY, endX, endY))

        # log image
        cv2.imwrite(output_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), config.JPEG_QUALITY])
        run.log_image(name=output_name, path=output_path, plot=None, description='auto-test')
        
        image_count += 1

# simple evaluation logic, EXAMPLE only
total_predicted = len(results)
avg_prob = np.mean([i['prob'] for i in results])
test_acc = round(total_predicted / image_count, 2)

# log metrics
run.log('total_predicted', total_predicted)
run.log('avg_prob', avg_prob)
run.log('test_acc', test_acc)