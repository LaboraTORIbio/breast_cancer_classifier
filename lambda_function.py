#!/usr/bin/env python
# coding: utf-8

import base64
from io import BytesIO
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite


def load_and_preprocess_input(base64_str):
    image_data = base64.b64decode(base64_str)
    img = Image.open(BytesIO(image_data))
    img = img.resize((224, 224), Image.NEAREST)

    x = np.array(img, dtype='float32')
    x = x[..., ::-1]
    mean = [103.939, 116.779, 123.68]
    x[..., 0] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]
    X = np.expand_dims(x, axis=0)
    
    return X


def predict(base64_str):
    interpreter = tflite.Interpreter(model_path='breast_cancer_classifier.tflite')
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']

    X = load_and_preprocess_input(base64_str)
    
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    classes = ['Benign', 'Malignant', 'Normal']
    formatted_preds = {class_name: f"{round(float(prob) * 100, 1)}%" for class_name, prob in zip(classes, preds[0])}
    
    return formatted_preds


def lambda_handler(event, context):
    base64_str = event['image']
    result = predict(base64_str)
    return result
