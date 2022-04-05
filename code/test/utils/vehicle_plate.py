import re
import requests
import json
import base64
import cv2
import imutils
import numpy as np
import tensorflow as tf
import os
from . import config
import logging


class Model:
    OUTPUT_TENSOR_NAMES = ['detected_boxes', 'detected_scores', 'detected_classes']

    def __init__(self, model_path):
        model = tf.saved_model.load(os.path.dirname(model_path))
        self.serve = model.signatures['serving_default']
        self.input_shape = self.serve.inputs[0].shape[1:3]


    def predict(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.input_shape[0], self.input_shape[1]), interpolation = cv2.INTER_AREA)
        
        inputs = np.array(image, dtype=np.float32)[np.newaxis, :, :, :]
        inputs = tf.convert_to_tensor(inputs)
        outputs = self.serve(inputs)
        return [outputs[n] for n in self.OUTPUT_TENSOR_NAMES]
    

    def translate_bbox(self, image, prediction_bbox):
        if len(prediction_bbox) == 4:
            h, w = image.shape[:2]
            startX = min(max(int(w * prediction_bbox[0]), 0), w)
            startY = min(max(int(h * prediction_bbox[1]), 0), h)
            endX = min(max(int(w * prediction_bbox[2]), 0), w)
            endY = min(max(int(h * prediction_bbox[3]), 0), h)
            return (startX, startY, endX, endY)
        else:
            return (None, None, None, None)