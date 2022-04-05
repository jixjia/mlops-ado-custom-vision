
'''
Author:     Jixin Jia (Gin)
Date:       2022.03.15
Version:    1.0
Purpose:    Create this scoring function to serve vehicle license plate detection as an API webservice
'''

import cv2
import os
import numpy as np
import base64
import json
from urllib.parse import urlparse

# load model class and utilities
from utils import config
from utils import vehicle_plate as vp
from utils import video_utilities as vu

def init():
    global model

    # AZUREML_MODEL_DIR is an environment variable created during deployment for ACS/AKS (./azureml-models/$MODEL_NAME/$VERSION)
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), config.MODEL_PATH)
    label_path = config.LABEL_PATH

    # load serialized trained model (.pb)
    try:
        print('Loading vehicle plate detection model...')
        model = vp.Model(model_path)

        print('[DEBUG] loaded model ->', model)
        print('Done!')

    except Exception as e:
        print('[DEBUG]', e.args)


def run(raw_data):

    # parse json payload
    data = json.loads(raw_data)

    try:
        # check if data contains a URL
        parsed = urlparse(data['data'])

        if all([parsed.scheme, parsed.netloc]):
            # retrieve image via the supplied URL
            image = vu.download_image(data['data'])
        
        # else decode base64 image string
        else:
            image = base64.b64decode(data['data'])
            img_array = np.asarray(bytearray(image), dtype=np.uint8)
            image = cv2.imdecode(img_array, -1)
    
    except Exception as e:
        return json.dumps({'message':'Unable to decode supplied image. Supply a valid URL or a base64 encoded image string.'})


    # detect vehicle plates
    preds = model.predict(image)

    # inspect predictions
    results = []
    for pred in zip(*preds):
        lpBBOX, prob = pred[:2]

        # convert 0d tensor to numpy
        prob = prob.numpy()
        
        if prob > config.LP_CONFIDENCE:
            # get safe bbox
            startX, startY, endX, endY = model.translate_bbox(image, lpBBOX)

            # pack into output list
            results.append({'startX':startX, 'startY':startY, 'endX':endX, 'endY':endY, 'prob': float(prob)})
    
    return results
