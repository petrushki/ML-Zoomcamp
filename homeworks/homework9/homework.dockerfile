FROM agrigorev/model-2025-hairstyle:v1


RUN pip install --no-cache-dir \
    onnxruntime \
    pillow \
    numpy

COPY lambda_handler.py /var/task/
