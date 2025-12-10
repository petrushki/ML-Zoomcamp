# predict.py
import os
import sys
from io import BytesIO
from urllib import request

import numpy as np
from PIL import Image


import onnxruntime as ort


MODEL_ONNX = "hair_classifier_v1.onnx"
MODEL_DATA = "hair_classifier_v1.onnx.data"  

IMAGE_URL = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"


def download_file(url, path):
    print(f"Downloading {url} -> {path} ...")
    with request.urlopen(url) as resp:
        data = resp.read()
    with open(path, "wb") as f:
        f.write(data)
    print("done.")


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img: Image.Image, target_size):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size, Image.NEAREST)
    return img


def image_to_tensor(img: Image.Image):
    arr = np.array(img).astype(np.float32) / 255.0  
    arr = arr.transpose(2, 0, 1)
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    arr = (arr - mean) / std
    return arr[np.newaxis, :, :, :] 


def inspect_onnx(model_path):
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    inputs = sess.get_inputs()
    outputs = sess.get_outputs()
    print("Inputs:")
    for i in inputs:
        print(f"  name: {i.name}, shape: {i.shape}, type: {i.type}")
    print("Outputs:")
    for o in outputs:
        print(f"  name: {o.name}, shape: {o.shape}, type: {o.type}")
    return sess


def run_inference(sess, input_tensor):
    # assume single input
    inp_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    print(f"Using input node: {inp_name}")
    print(f"Output node(s): {out_names}")
    out = sess.run(out_names, {inp_name: input_tensor.astype(np.float32)})
    return out


def main():
    # if model files not present, download them
    if not os.path.exists(MODEL_ONNX):
        print("Model file not found locally, downloading...")
        download_file("https://github.com/alexeygrigorev/large-datasets/releases/download/hairstyle/hair_classifier_v1.onnx", MODEL_ONNX)
    else:
        print(f"Found {MODEL_ONNX} locally.")

    if not os.path.exists(MODEL_DATA):
        print("Model .data file not found locally; downloading.")
        download_file("https://github.com/alexeygrigorev/large-datasets/releases/download/hairstyle/hair_classifier_v1.onnx.data", MODEL_DATA)
    else:
        print(f"Found {MODEL_DATA} locally.")

    sess = inspect_onnx(MODEL_ONNX)

    target_size = (200, 200)
    print(f"Using target size: {target_size} (width x height)")


    img = download_image(IMAGE_URL)
    print("Original image size:", img.size)
    prepared = prepare_image(img, target_size)
    print("Resized image size:", prepared.size)

 
    tensor = image_to_tensor(prepared) 
 
    first_pixel_r = tensor[0, 0, 0, 0]  
    print("First pixel (R channel) after preprocessing:", float(first_pixel_r))


    outputs = run_inference(sess, tensor)
    print("Raw model outputs:")

    for i, o in enumerate(outputs):
        print(f"  output[{i}] -> shape {np.array(o).shape}, values (flattened) {np.array(o).ravel()[:10]}")

    if len(outputs) == 1 and np.array(outputs[0]).size == 1:
        print("Model output (single scalar):", float(np.array(outputs[0]).ravel()[0]))
    else:
        print("Model primary output value (first element):", float(np.array(outputs[0]).ravel()[0]))


if __name__ == "__main__":
    main()
