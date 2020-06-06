# MSDS19040_COVID19_DLSpring2020

Fine tuned deep learning Experiments on COVID DataSet using state of the art NN architecture ResNet-18 and VGG-16 with and without focal loss 

Covid Images Dataset:
![](Images/covid-dataset.png)

#### ResNet 18:
##### Load pretrained CNN model and fine-tune FC Layers
![](Images/resnet18_test_cm.png)
![](Images/resnet18_valid_cm.png)

##### Fine-tune the CNN and FC layers of the network
![](Images/t2_resnet18_test_cm.png)
![](Images/t2_resnet18_valid_cm.png)


#### VGG 16:
##### Load pretrained CNN model and fine-tune FC Layers
![](Images/vgg16_test_cm.png)
![](Images/vgg16_test_cm.png)

##### Fine-tune the CNN and FC layers of the network
![](Images/t2_vgg16_test_cm.png)
![](Images/t2_vgg16_test_cm.png)


## Detailed Analysis

Discussion: 

In multi-class classification, a balanced dataset with labels that are equally distributed. If one class have more samples than another, it can be an imbalanced dataset. This imbalance causes following problem: 

Training is inefficient as most samples are easy examples that contribute no useful learning signal and the easy examples can overwhelm training and lead to degenerate models. 

Secondly it may cause biasness in dataset, to get rid of this problem we use focal loss working so our system can detect less sampled labels, in our case it is COVID patients. 

 
