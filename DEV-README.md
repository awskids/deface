# Development Readme

python -m deface.deface ../frigate-with-ai/debug/cctv.mp4

## model

https://github.com/Star-Clouds/centerface

## stubs

python -c 'import os, cv2; print(os.path.dirname(cv2.__file__))'

curl -sSL https://raw.githubusercontent.com/microsoft/python-type-stubs/main/cv2/__init__.pyi \
    -o .venv/lib/python3.10/site-packages/cv2/__init__.pyi

 Module: cv2.cv2, version: 4.4.0 <--- more than two years old

opencv_version = "4.7.0.68"

opencv-python

Bad stubs (none seem to be on Internet....)

## onnx

https://github.com/onnx/tutorials
https://github.com/onnx/models

Onnx & OpenVino are similar, (save money running optimized models on CPU)
https://blog.ml6.eu/openvino-vs-onnx-for-transformers-in-production-3e10c01520c8

## Conv (Layer) learning

https://towardsdatascience.com/understanding-input-and-output-shapes-in-convolution-network-keras-f143923d56ca

https://stackoverflow.com/questions/44747343/keras-input-explanation-input-shape-units-batch-size-dim-etc

keras defaults to channels last

