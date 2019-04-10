# Flask server demo for car make and model recognition

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

## Introduction

A Python example for using [Spectrico's car make and model classifier](http://spectrico.com/car-make-model-recognition.html). The Flask server exposes REST API for car make&model recognition. It consists of an object detector for finding the cars, and a classifier to recognize the makes and models of the detected cars. The object detector is an implementation of YOLOv3 (OpenCV DNN backend). It doesn't use GPU and one frame takes 1s to process on Intel Core i5-7600 CPU. YOLOv3 weights were downloaded from [YOLO website](https://pjreddie.com/darknet/yolo/). The classifier is based on Mobilenet v2 (TensorFlow backend). It takes 35 milliseconds on Intel Core i5-7600 CPU for single classification. The API is simple: make a HTTP POST request to local host on port 6000. The input image must be send using multipart/form-data encoding. It has to be jpg or png. Tested on Windows 10 and Ubuntu Linux 16.04 LTS.
The demo doesn't include the classifier for car make and model recognition. It is a commercial product and is available for purchase at [http://spectrico.com/car-make-model-recognition.html](http://spectrico.com/car-make-model-recognition.html).

![image](https://github.com/spectrico/car-make-model-classifier-yolo3-cpp/raw/master/car-make-model.png?raw=true)

---

#### Usage
The server is started using:
```
$ python car_make_model_recognition_server.py
```
The request format using curl is:
```
curl "http://127.0.0.1:6000" -H "Content-Type: multipart/form-data" --form "image=@cars.jpg"
```
The response is in JSON format:
```
{
  "cars" : [
    {
      "make" : Volkswagen,
      "model" : Arteon,
      "prob" : 0.93550926,
      "rect" : {
        "left" : 606,
        "top" : 143,
        "width" : 440,
        "height" : 193
      }
    },
    {
      "make" : Volkswagen,
      "model" : Polo,
      "prob" : 0.8868938,
      "rect" : {
        "left" : 958,
        "top" : 157,
        "width" : 318,
        "height" : 158
      }
    },
    {
      "make" : Volkswagen,
      "model" : Tiguan,
      "prob" : 0.9830027,
      "rect" : {
        "left" : 361,
        "top" : 137,
        "width" : 277,
        "height" : 176
      }
    }
  ]
}
```
---
## Requirements
  - python
  - numpy
  - tensorflow
  - opencv
  - PIL
  - yolov3.weights must be downloaded from [https://pjreddie.com/media/files/yolov3.weights](https://pjreddie.com/media/files/yolov3.weights) and saved in folder yolo-coco

---
## Configuration

The settings are stored in python file named config.py:
```
model_file = "model-weights-spectrico-mmr-mobilenet-224x224-908A6A8C.pb"
label_file = "labels.txt"
input_layer = "input_1"
output_layer = "softmax/Softmax"
classifier_input_size = (224, 224)
```
***model_file*** is the path to the car make and model classifier
***classifier_input_size*** is the input size of the classifier
***label_file*** is the path to the text file, containing a list with the supported makes and models

---
## Credits
The example is based on the tutorial by Adrian Rosebrock: [YOLO object detection with OpenCV](https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/)
The YOLOv3 object detector is from: [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolo/)
```
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
```

The car make and model classifier is based on MobileNetV2 mobile architecture: [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
