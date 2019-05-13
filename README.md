# Singapore-Maritime-Dataset-Trained-Deep-Learning-Models
This repository contains the training configurations for several Deep Learning models trained on the _Singapore Maritime Dataset_ (SMD) and links to the trained - ready to use - models. This can be considered as a model zoo for the Singapore Maritime Dataset. 

### Software frameworks used for training
The models were selelcted and trained using two separate software frameworks:

- [Tensorflow object detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
- [Keras YOLOv2 implementation](https://github.com/experiencor/keras-yolo2)

### Datasets used for training

Two separate splittings of the Singapore Maritime Dataset were used for training:

- The first split (Dataset 1) was created using [this code](https://github.com/tilemmpon/Singapore-Maritime-Dataset-Frames-Ground-Truth-Generation-and-Statistics/blob/master/Singapore_dataset_frames_generation_and_histograms.ipynb). The code extracts every firth frame of each video of the SMD. Then the first 70% was used for training and the rest 30% for testing.
- The second split (Dataset 2) was created using [this code](https://github.com/tilemmpon/Singapore-Maritime-Dataset-Frames-Ground-Truth-Generation-and-Statistics/blob/master/Singapore_dataset_frames_generation_2nd_dataset.ipynb). In this case also every firth frame of each video of the SMD is extracted. However, the frames of 4 selected videos are added completely in the test part  while for the rest of the videos - as before - the first 70% of the frames is added in the training part and the rest 30% in the testing.

More more information about how the datasets used are generated please refer to the respective Jupyter notebooks linked. All selected models from both architectures were  trained on Dataset 1. The best performing models were tested also in Dataset 2 to check their performance on a more challenging splitting of the SMD.

### Models trained using Tensorflow object detection API

Several models trained on COCO dataset were selected and fine-tuned. The results can be seen below. Some information from the [original repository](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) is:
