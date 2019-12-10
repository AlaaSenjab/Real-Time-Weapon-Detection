# Real Time Weapon Detection Using Tesorflow Object Detection API.

![Image description](https://github.com/AlaaSenjab/Real-Time-Weapon-Detection/blob/master/Demo/de1.gif)
## Probelm statement:

Over the past eight years, the number of school mass shootings has increased drastically. Gun violence on school grounds resulted in many wounded or dead school kids which mirrors the gun violence problem in the United States. The failure to address the root cause of school gun violence is having lasting consequences for millions of American children.
As we are waiting for our leaders to find a solution for gun violence as a whole and on school grounds specifically, we need to find an alternative way to minimize the number of incidents or prevent it from occurring in the first place.

On average, police response time is around 18 minutes. When a shooting occur, sometimes calling 911 can't be done right away, instead, a person should be in a safe place for him to be able to call for help. and that might take so much more time. So, the 18 minutes police response time does not include the time it takes to call 911.


![Image description](https://github.com/AlaaSenjab/Real-Time-Weapon-Detection/blob/master/Demo/respons_time.png)
## Project Summary:

In this project, i am proposing a way to reduce the police time greatly by using real-time weapon detection that could be implemented on any cctv camera.
The object detection model is made so it could find a weapon in a picture, video or most importantly, a live video feed.

Once a weapon is detected in a live video, police could be alerted. They can see the live feed and verify that someone has a gun before dispatching to the incident location. This method does not require anyone to actually call the police, moreover, this will cut the response time and may prevent a shooting from happening.

## Object Detection Method Used:
Training a model for object detection from scratch is hard, time and resource consuming, and might not perform well. Instead of training a model from scratch, transfer learning fast and easy.
Tensorflow is one of the most well known open source framework that allows to build object detection models using its object detection API. It uses models that were trained on overall similar objects such as animals and cars. Using The idea of transfer learning makes it easier and faster to train your own custom dataset to detect the objects of your choice.

Using these state-of-the-art pretrained model is pretty straightforward in Tensorflow object detection API, but picking one of them will depend on what you are trying to achieve.

Here is a list of some of the pretrained models.

| Model name  | Speed (ms) | COCO mAP ||
| ------------ | :--------------: | :--------------: | :-------------: |
| [ssd_mobilenet_v1_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz) | 30 | 21 |
| [ssd_mobilenet_v1_0.75_depth_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03.tar.gz) | 26 | 18 |
| [ssd_mobilenet_v1_quantized_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18.tar.gz) | 29 | 18 |
| [ssd_mobilenet_v1_0.75_depth_quantized_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18.tar.gz) | 29 | 16 |
| [ssd_mobilenet_v1_ppn_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03.tar.gz) | 26 | 20 |
| [ssd_mobilenet_v1_fpn_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz) | 56 | 32 |
| [ssd_resnet_50_fpn_coco](http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz) | 76 | 35 |
| [ssd_mobilenet_v2_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz) | 31 | 22 |
| [ssd_mobilenet_v2_quantized_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz) | 29 | 22 |
| [ssdlite_mobilenet_v2_coco](http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz) | 27 | 22 |
| [ssd_inception_v2_coco](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz) | 42 | 24 |
| [faster_rcnn_inception_v2_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz) | 58 | 28 |
| [faster_rcnn_resnet50_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz) | 89 | 30 |


You can check the other available pretrained models from the official [Git repo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).

I have used `ssd_mobilenet_v2_coco` for this project for 2 reasons:
1. It has a low inference time (31 ms) which will allow my model to run using a real time webcam feed.
2. It has an acceptable mAP score (Think of mean Average Precision (mAP) as the accuracy score in object detection. [Read more](https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173#:~:targetText=AP%20(Average%20precision)%20is%20a,illustrate%20it%20with%20an%20example.))


## The Data
The images and labels are taken from [University of Granada research group](https://sci2s.ugr.es/weapons-detection).
For this project, i am using the region proposals approach. The data contains 3000 images of guns (mostly handguns) in `.jpg` format and their appropriate labels/annotations.

For each image, there is an xml file that contains the class name of the object (pistol) and the coordinates of the box surrounding the object.

Here is some of the fields from one of the xml files:
```
<annotation>
  <filename>armas (3)</filename>    <-- file name
  <size>
    <width>1300</width>             <-- image width
    <height>866</height>            <-- image height
    <depth>3</depth>                <-- 3 = a colored image (RGB)
  </size>
  <object>
    <name>pistol</name>             <-- class name

    <bndbox>                        <-- box position around object
      <xmin>471</xmin>              
      <ymin>207</ymin>
      <xmax>613</xmax>
      <ymax>359</ymax>
    </bndbox>
  </object>
</annotation>
```

## Training process

I installed and ran Tensorboard on the Google Colab notebook to check how the model is behaving in real time.
The training process took around 14 hours of training using the GPU provided by Google Colab. It also included a lot of hyperpermeters tuning such as:
1. Image augmentation: randomly rotate, flip, resize, adjust color contrast any many more.
2. Changing the learning rate
3. Adding some regularization such as drop out.

## Inferencing the Trained Model

While training, Tensorboard offers live feed back on how the model is doing. Perhaps, one of the most important ones to look for is Mean Average Precision (mAP).

To understand mAP better, we need to know what Intersection Over Union(IOU) is:

![Image description](https://github.com/AlaaSenjab/Real-Time-Weapon-Detection/blob/master/Demo/IOU.png)


The IOU will yield to a percentage and depending on what we set our threshold to be, everything below that percentage is considered False Positive, and above it is considered True Positive.

![Image description](https://github.com/AlaaSenjab/Real-Time-Weapon-Detection/blob/master/Demo/mAP%40.5IOU.png)


## Conclusion

Reducing police response time is critical in situations where gun fire might occur. This project aimed to minimize this response time by implementing a weapon detection method while using a live camera. Sometimes calling 911 cannot be done until several minutes after an incident occur. Even though some further tests need to be done specifically to reduce false positives, this methodology could be a live savior for many people.
This project can be generalized to work on many other industries and locations. For example, public parks, gas stations, banks, etc.
