# lambda_handler.py
import json
import os
from io import BytesIO
from urllib import request

import numpy as np
from PIL import Image
import onnxruntime as ort

MODEL_PATH = os.environ.get("MODEL_PATH", "/var/task/hair_classifier_empty.onnx") 


TARGET_SIZE = (200, 200)
MEAN = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)


session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_names = [o.name for o in session.get_outputs()]


def download_image(url: str) -> Image.Image:
    with request.urlopen(url) as resp:
        buffer = resp.read()
    return Image.open(BytesIO(buffer))


def prepare_image(img: Image.Image, target_size):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size, Image.NEAREST)
    return img


def image_to_tensor(img: Image.Image):
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    arr = (arr - MEAN) / STD
    return arr[np.newaxis, :, :, :].astype(np.float32)


def predict_from_url(url: str):
    img = download_image(url)
    img = prepare_image(img, TARGET_SIZE)
    tensor = image_to_tensor(img)
    outputs = session.run(output_names, {input_name: tensor})
    out = outputs[0]
    flat = np.array(out).ravel()
    return float(flat[0]) if flat.size > 0 else None


def lambda_handler(event, context):
 
    url = None
    if event.get("queryStringParameters") and event["queryStringParameters"].get("url"):
        url = event["queryStringParameters"]["url"]
    else:
        try:
            body = json.loads(event.get("body") or "{}")
            url = body.get("url")
        except Exception:
            url = None

    if not url:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "Please provide 'url' to an image."}),
        }

    try:
        score = predict_from_url(url)
        return {"statusCode": 200, "body": json.dumps({"score": score})}
    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
