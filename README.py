ECFRNet: Effective corner feature representations network for image corner detection

The code is tested on windows

### Requirement
Python package:

tensorflow=1.14.0, tqdm, cv2, pytorch==1.2.0, skimage, glob

### Usage

ECFRNet catalogue ：

Main Folder：

best checkpoint：Contains optimal weights for training 

checkpoint：Weight saving folder during training 

data---datasets： ImageNet datasets and test image are placed in here
    ---train_pair：The training data set is placed here ,and named as ECFRNet_test_block.mat and ECFRNet_train_block.mat

HardNet:HardNet++ descriptor generated File is placed here

model：Network model is placed here


#### Get the ImageNet dataset

Download data from http://www.image-net.org/

The download image is palced in‘./data/datasets/ImageNet’

1.Making training set

cd‘./ECFRNet/'：
     Rcorner.m
     get_training_pair.m

1.Traning ECFRNet：

cd‘./ECFRNet/'：

      python ECFRNet_train.py

2.Testing ECFRNet：

2.1 If only the corner features are extracted.

cd‘./ECFRNet/'：

      python ECFRNet_test.py # the 2*N*M corner prediction coordinates are generated 

      extractor_corner.m #The robust corners are extracted by votting and NMS

2.2 If the corner features and descriptors are extracted(HardNet is used to extract the descriptor ).

cd‘./ECFRNet/'：

      python ECFRNet_test.py # the 2*N*M corner prediction coordinates are generated 

      extractor_all.m #The robust corners are extracted by votting and NMS, and the descriptors are extratcted by HardNet

3. If you use our optimal model, please copy the optimal weights under the best checkpoint file to the checkpoint before testing. 

      Take the 'lab.png' image as an example:

      python ECFRNet_test.py, extractor_corner.m, draw.m  #The corners with high robustness are extracted and displayed on the original image,.
