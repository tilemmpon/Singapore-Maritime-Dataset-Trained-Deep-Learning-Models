# Singapore-Maritime-Dataset-Trained-Deep-Learning-Models
This repository contains the training configurations for several Deep Learning models trained on the _Singapore Maritime Dataset_ (SMD) and links to the trained - ready to use - models. This can be considered as a model zoo for the Singapore Maritime Dataset. 

## Software frameworks used for training
The models were selelcted and trained using two separate software frameworks:

- [Tensorflow object detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
- [Keras YOLOv2 implementation](https://github.com/experiencor/keras-yolo2)

## Datasets used for training

Two separate splittings of the Singapore Maritime Dataset were used for training:

- The first split (Dataset 1) was created using [this code](https://github.com/tilemmpon/Singapore-Maritime-Dataset-Frames-Ground-Truth-Generation-and-Statistics/blob/master/Singapore_dataset_frames_generation_and_histograms.ipynb). The code extracts every firth frame of each video of the SMD. Then the first 70% was used for training and the rest 30% for testing.
- The second split (Dataset 2) was created using [this code](https://github.com/tilemmpon/Singapore-Maritime-Dataset-Frames-Ground-Truth-Generation-and-Statistics/blob/master/Singapore_dataset_frames_generation_2nd_dataset.ipynb). In this case also every firth frame of each video of the SMD is extracted. However, the frames of 4 selected videos are added completely in the test part  while for the rest of the videos - as before - the first 70% of the frames is added in the training part and the rest 30% in the testing.

More more information about how the datasets used are generated please refer to the respective Jupyter notebooks linked. All selected models from both architectures were  trained on Dataset 1. The best performing models were tested also in Dataset 2 to check their performance on a more challenging splitting of the SMD.

##Models trained using Tensorflow object detection API

Several models trained on COCO dataset were selected and fine-tuned. The results can be seen below. Some information (partly adapted) from the [original repository](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) is:

In the table below, the trained models in the SMD are listed including:

* a model name,
* a download link to a tar.gz file containing the trained model,
* the dataset this model was trained on,
* model speed --- I report running time in ms using the image size according to each model's configuration (including all
  pre and post-processing), but please be
  aware that these timings depend highly on one's specific hardware
  configuration (these timings were performed using an Nvidia
  GeForce GTX 1070 with 8GB memory) and should be treated more as relative timings in
  many cases. Also note that desktop GPU timing does not always reflect mobile
  run time. For example Mobilenet V2 is faster on mobile devices than Mobilenet
  V1, but is slightly slower on desktop GPU. The times were infered using [Tensorflow_object_detection_time Jupyter notebook](https://github.com/tilemmpon/Singapore-Maritime-Dataset-Trained-Deep-Learning-Models/blob/master/NOTEBOOKS/Tensorflow_object_detection_time.ipynb).
* mAP 0.5 IOU detector performance on subset of the test part of dataset trained on.
  Here, higher is better, and I only report bounding box mAP rounded to the
  nearest integer.
* Name and link of the configuration file used for the training,
* Name and link of the pre-trained model on COCO dataset used for fine-tuning

You can un-tar each tar.gz file via, e.g.,:

```
tar -xzvf ssd_mobilenet_v1_coco.tar.gz
```

Inside the un-tar'ed directory, you will find:

* a graph proto (`graph.pbtxt`)
* a checkpoint
  (`model.ckpt.data-00000-of-00001`, `model.ckpt.index`, `model.ckpt.meta`)
* a frozen graph proto with weights baked into the graph as constants
  (`frozen_inference_graph.pb`) to be used for out of the box inference
    (try this out in the Jupyter notebook!)
* a config file (`pipeline.config`) which was used to generate the graph.

Some remarks on frozen inference graphs:

* These frozen inference graphs were generated using the
  [v1.9.0](https://github.com/tensorflow/tensorflow/tree/v1.9.0)
  release version of Tensorflow and I do not guarantee that these will work
  with other versions.

### Trained models list

| Model name  | Dataset trained| Speed (ms) | mAP @ 0.5 IOU | training configuration |pre-trained model used|
| ------------ | :--------------: | :--------------: | :--------------: | :-------------: | :-------------: |
| [ssd_mobilenet_v2_smd](https://mega.nz/#!ijhU1IzQ!w4InM1iJsUtUKXXJYcB7aAM8K1fDVjuXwgLD_cSHhdM) |Dataset 1| 28.6 | 65 | [ssd_inception_v2_smd.config](https://github.com/tilemmpon/Singapore-Maritime-Dataset-Trained-Deep-Learning-Models/blob/master/tensorflow_training_configurations/ssd_inception_v2_smd.config) |[ssd_inception_v2_coco](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz)|
| [ssd_inception_v2_smd](https://mega.nz/#!W2hEjKQR!R7gVeAa9Vq5yVYLaWdT87df02R9pSUNjVgb9PWayyQQ) |Dataset 1| 23.9 | 59 | [ssd_mobilenet_v2_smd.config](https://github.com/tilemmpon/Singapore-Maritime-Dataset-Trained-Deep-Learning-Models/blob/master/tensorflow_training_configurations/ssd_mobilenet_v2_smd.config) | [ssd_mobilenet_v2_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz)|
| [ssd_mobilenet_v1_fpn_smd](https://mega.nz/#!zjhEAIoJ!_CVzuP0GJ2FMuP4mlO7Fu5WLT0rnDwahNxD87FBGpIE) |Dataset 1| 52 | 70 | [ssd_mobilenet_v1_fpn_smd.config](https://github.com/tilemmpon/Singapore-Maritime-Dataset-Trained-Deep-Learning-Models/blob/master/tensorflow_training_configurations/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_smd.config) |[ssd_mobilenet_v1_fpn_coco ](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz) |
| [ssd_resnet_50_fpn_smd](https://mega.nz/#!Lq4ygQgB!HNiwxfGyrntAeIHrGGu1kOffxWnwY5LjgjThddbB1-E) |Dataset 1| 70.8 | 71 | [ssd_resnet_50_fpn_smd.config](https://github.com/tilemmpon/Singapore-Maritime-Dataset-Trained-Deep-Learning-Models/blob/master/tensorflow_training_configurations/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_smd.config) |[ssd_resnet_50_fpn_coco ](http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz) |
| [faster_rcnn_resnet50_smd](https://mega.nz/#!mmoACCwL!vV6ocwCvRiSGOnmy6aqA4_HEgolLvsG2zC75VERQLzA) |Dataset 1| 173.7 | 74 | [faster_rcnn_resnet50_smd.config](https://github.com/tilemmpon/Singapore-Maritime-Dataset-Trained-Deep-Learning-Models/blob/master/tensorflow_training_configurations/faster_rcnn_resnet50_smd.config) |[faster_rcnn_resnet50_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz)|
| [faster_rcnn_resnet101_smd](https://mega.nz/#!OvpQhQIS!At-LiNYDZi50K-3L31zGtzQEEwc6b5V9P4jw2JsZfr8) |Dataset 1| 203.6 | 64 | [faster_rcnn_resnet101_smd.config](https://github.com/tilemmpon/Singapore-Maritime-Dataset-Trained-Deep-Learning-Models/blob/master/tensorflow_training_configurations/faster_rcnn_resnet101_smd.config) | [faster_rcnn_resnet101_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz)|
| [faster_rcnn_inception_v2_smd](https://mega.nz/#!euwQSCpT!p6nzkhW73bV7QG7wT15MoljspW1XM3pUZ9djwcgzQkk) |Dataset 1| 76.5 | 76 | [faster_rcnn_inception_v2_smd.config](https://github.com/tilemmpon/Singapore-Maritime-Dataset-Trained-Deep-Learning-Models/blob/master/tensorflow_training_configurations/faster_rcnn_inception_v2_smd.config) |  [faster_rcnn_inception_v2_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz)|
| [faster_rcnn_inception_resnet_v2_atrous_smd](https://mega.nz/#!K6hS1CxI!nPaJF4E_ZljgL63e6bA_0qWqeUuK-7TDvX4I2MhB4ho) |Dataset 1| 745.5 | 54 | [faster_rcnn_inception_resnet_v2_atrous_smd.config](https://github.com/tilemmpon/Singapore-Maritime-Dataset-Trained-Deep-Learning-Models/blob/master/tensorflow_training_configurations/faster_rcnn_inception_resnet_v2_atrous_smd.config) |  [faster_rcnn_inception_resnet_v2_atrous_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz)|
| [ssd_mobilenet_v1_fpn_smd_2](https://mega.nz/#!3zhClCQK!BXwCpuyq6VAb4QAYQc286wkwk4Wm_3kmjT6hBVYbDMs) |Dataset 2| 52.2 | 61 | [ssd_mobilenet_v1_fpn_smd_2.config](https://github.com/tilemmpon/Singapore-Maritime-Dataset-Trained-Deep-Learning-Models/blob/master/tensorflow_training_configurations/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_smd_2.config) |[ssd_mobilenet_v1_fpn_coco ](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz) |
| [ssd_resnet_50_fpn_smd_2](https://mega.nz/#!Lq4ygQgB!HNiwxfGyrntAeIHrGGu1kOffxWnwY5LjgjThddbB1-E) |Dataset 2| 71.2 | 62 | [ssd_resnet_50_fpn_smd_2.config](https://github.com/tilemmpon/Singapore-Maritime-Dataset-Trained-Deep-Learning-Models/blob/master/tensorflow_training_configurations/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_smd_2.config) |[ssd_resnet_50_fpn_coco ](http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz) |
| [faster_rcnn_resnet50_smd_2](https://mega.nz/#!TvhElY7K!vfMqbMY9e5YHSpMRrw7mdXd5iXa-YCdtQU0ae-EtD7E) |Dataset 2| 176.1 | 69 | [faster_rcnn_resnet50_smd_2.config](https://github.com/tilemmpon/Singapore-Maritime-Dataset-Trained-Deep-Learning-Models/blob/master/tensorflow_training_configurations/faster_rcnn_resnet50_smd_2.config) |[faster_rcnn_resnet50_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz)|
| [faster_rcnn_inception_v2_smd_2](https://mega.nz/#!y2xQjSYB!6vXFc1jHlBFZp2RLcpCzPWNh9xNPDd1qjgrxtmqPHh4) |Dataset 2| 78.2 | 67 | [faster_rcnn_inception_v2_smd_2.config](https://github.com/tilemmpon/Singapore-Maritime-Dataset-Trained-Deep-Learning-Models/blob/master/tensorflow_training_configurations/faster_rcnn_inception_v2_smd_2.config) |  [faster_rcnn_inception_v2_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz)|


## Models trained using Keras YOLOv2 implementation

Using this framework the following back-ends were used with YOLOv2 for the training:

- Darknet-19 (Full YOLOv2)
- Tiny Darknet (Tiny YOLOv2)
- SqueezeNet (SqueezeNet YOLOv2)

In the table below, the trained models in the SMD are listed including:

* a model name,
* a download link to a .h5 file containing the trained model,
* the dataset this model was trained on,
* model speed --- I report running time in ms using the image size according to each model's configuration (including all
  pre and post-processing), but please be
  aware that these timings depend highly on one's specific hardware
  configuration (these timings were performed using an Nvidia
  GeForce GTX 1070 with 8GB memory) and should be treated more as relative timings in
  many cases.The times were infered using [Keras_YOLO_time_of_inference Jupyter notebook](https://github.com/tilemmpon/Singapore-Maritime-Dataset-Trained-Deep-Learning-Models/blob/master/NOTEBOOKS/Keras_YOLO_time_of_inference.ipynb).
* mAP 0.5 IOU detector performance on subset of the test part of dataset trained on.
  Here, higher is better, and I only report bounding box mAP rounded to the
  nearest integer.
* Name and link of the configuration file used for the training,
* Name and link of the back-end model used for feature extraction.

### Trained models list

| Model name  | Dataset trained| Speed (ms) | mAP @ 0.5 IOU | training configuration |back-end used|
| ------------ | :--------------: | :--------------: | :--------------: | :-------------: | :-------------: |
| [full_yolo_v2_smd](https://mega.nz/#!e2QBHILa!OZ-_fykXbaBmza26aJCqNwxoXEETkY_dN86pLr5bWyg) |Dataset 1| 40.6 | 55 | [config_full_yolo.json](https://github.com/tilemmpon/Singapore-Maritime-Dataset-Trained-Deep-Learning-Models/blob/master/YOLOv2_training_configurations/config_full_yolo.json) |[full_yolo_backend.h5](https://mega.nz/#!3yYzkaDD!kSFJVXtaOaOsZHC_xoxl8ZaYRkES5xx0-3iW6RyBlzs)|
| [tiny_yolo_v2_smd](https://mega.nz/#!CvY3RSqZ!w-bMmo1UnxVI1NkMHCvvbYYPqabgGtl0SI-JRH6ryWc) |Dataset 1| 29.2 | 43 | [config_tiny_yolo.json](https://github.com/tilemmpon/Singapore-Maritime-Dataset-Trained-Deep-Learning-Models/blob/master/YOLOv2_training_configurations/config_tiny_yolo.json) |[tiny_yolo_backend.h5](https://onedrive.live.com/?authkey=%21AM2OzK4S4RpT%2DSU&cid=5FDEBAB7450CDD92&id=5FDEBAB7450CDD92%21107&parId=5FDEBAB7450CDD92%21121&o=OneUp)|
| [squeezenet_yolo_v2_smd](https://mega.nz/#!ijRRXSbZ!mShj3Z5h918ihc1SoaBRmBw_ZIJSlaEczRZeVWz6MV8) |Dataset 1| 47.8 | 27 | [config_squeezenet.json](https://github.com/tilemmpon/Singapore-Maritime-Dataset-Trained-Deep-Learning-Models/blob/master/YOLOv2_training_configurations/config_squeezenet.json) |[squeezenet_backend.h5](https://onedrive.live.com/?authkey=%21AM2OzK4S4RpT%2DSU&cid=5FDEBAB7450CDD92&id=5FDEBAB7450CDD92%21111&parId=5FDEBAB7450CDD92%21121&o=OneUp)|
| [full_yolo_v2_smd_2](https://mega.nz/#!evIDhIwJ!VMXgiEGlEPGPGRbRPrKsc5JPG_BKQ4aS3yrCh4cQwfY) |Dataset 2| 41.9 | 33 | [config_full_yolo_2.json](https://github.com/tilemmpon/Singapore-Maritime-Dataset-Trained-Deep-Learning-Models/blob/master/YOLOv2_training_configurations/config_full_yolo_2.json) |[full_yolo_backend.h5](https://mega.nz/#!3yYzkaDD!kSFJVXtaOaOsZHC_xoxl8ZaYRkES5xx0-3iW6RyBlzs)|

## Example inferred images

Here are some detection example of the trained models for the dataset 1. The generation of the inferred images for Keras YOLO v2 implementations was performed using [Keras_YOLO_prediction_and_save_video_and_images Jupyter notebook](https://github.com/tilemmpon/Singapore-Maritime-Dataset-Trained-Deep-Learning-Models/blob/master/NOTEBOOKS/Keras_YOLO_prediction_and_save_video_and_images.ipynb). For Tensorflow the [Jupyter Notebook provided in the tutorial](https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb) was used.

### Example results for faster_rcnn_inception_v2_smd trained on dataset1

![MVI_1469_VIS_frame470](https://github.com/tilemmpon/Singapore-Maritime-Dataset-Trained-Deep-Learning-Models/blob/master/example_inferred_images/faster_rcnn_inception_v2_dataset1/MVI_1469_VIS_frame470.jpg)

![MVI_1520_NIR_frame490](https://github.com/tilemmpon/Singapore-Maritime-Dataset-Trained-Deep-Learning-Models/blob/master/example_inferred_images/faster_rcnn_inception_v2_dataset1/MVI_1520_NIR_frame490.jpg)

![MVI_1578_VIS_frame490](https://github.com/tilemmpon/Singapore-Maritime-Dataset-Trained-Deep-Learning-Models/blob/master/example_inferred_images/faster_rcnn_inception_v2_dataset1/MVI_1578_VIS_frame490.jpg)

### Example results for ssd_resnet_50_fpn_smd trained on dataset1

![MVI_1468_NIR_frame245](https://github.com/tilemmpon/Singapore-Maritime-Dataset-Trained-Deep-Learning-Models/blob/master/example_inferred_images/ssd_resnet_50_fpn_dataset1/MVI_1468_NIR_frame245.jpg)

![MVI_1474_VIS_frame425](https://github.com/tilemmpon/Singapore-Maritime-Dataset-Trained-Deep-Learning-Models/blob/master/example_inferred_images/ssd_resnet_50_fpn_dataset1/MVI_1474_VIS_frame425.jpg)

![MVI_1486_VIS_frame620](https://github.com/tilemmpon/Singapore-Maritime-Dataset-Trained-Deep-Learning-Models/blob/master/example_inferred_images/ssd_resnet_50_fpn_dataset1/MVI_1486_VIS_frame620.jpg)

## Citing

If the Singapore Maritime Dataset is used please cite it as:
D. K. Prasad, D. Rajan, L. Rachmawati, E. Rajabaly, and C. Quek, 
"Video Processing from Electro-optical Sensors for Object Detection and 
Tracking in Maritime Environment: A Survey," IEEE Transactions on Intelligent 
Transportation Systems (IEEE), 2017. 

If models/code/figures/results from this repo are used please add a reference to the repository.

## Contribution

To report an issue use the GitHub issue tracker. Please provide as much information as you can.

Contributions (like new trained models etc.) are always welcome. Open an issue to contact me. The preferred method of contribution is through a github pull request.
