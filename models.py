## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # Convolutional layers
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        ## output size = (W-F)/S +1 = (224-5)/1 +1 = 220
        # the output Tensor for one image, will have the dimensions: (32, 220, 220)
        # after one pool layer, this becomes (32, 110, 110)
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.dropout1 = nn.Dropout(p=0.2)
        
        # 32 input channels/feature maps, 64 output channels/feature maps, 3x3 square convolution kernel
        ## output size = (W-F)/S +1 = (110-3)/1 +1 = 108
        # the output Tensor for one image, will have the dimensions: (64, 108, 108)
        # after one pool layer, this becomes (64, 54, 54)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout(p=0.3)
        
        # 64 input channels/feature maps, 128 output channels/feature maps, 3x3 square convolution kernel
        ## output size = (W-F)/S +1 = (54-3)/1 +1 = 52
        # the output Tensor for one image, will have the dimensions: (128, 52, 52)
        # after one pool layer, this becomes (128, 26, 26)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.bn2 = nn.BatchNorm2d(128)
        self.dropout3 = nn.Dropout(p=0.4)
        
        # 128 input channels/feature maps, 256 output channels/feature maps, 1x1 square convolution kernel
        ## output size = (W-F)/S +1 = (26-1)/1 +1 = 26
        # the output Tensor for one image, will have the dimensions: (256, 26, 26)
        # after one pool layer, this becomes (256, 13, 13)
        self.conv4 = nn.Conv2d(128, 256, 1)
        self.bn3 = nn.BatchNorm2d(256)
        self.dropout4 = nn.Dropout(p=0.5)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)
        
        # 256 outputs * the 13*13 filtered/pooled map size
        self.fc1 = nn.Linear(256*13*13, 13*136)
        
        # dropout with p=0.4
        self.fc1_drop = nn.Dropout(p=0.4)
        
        # finally, create 136 output channels (2 for each of the 68 keypoint (x, y) pairs)
        self.fc2 = nn.Linear(13*136, 136)
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        # four conv/relu/BatchNorm + pool layers
        x = self.dropout1(self.pool(F.elu(self.conv1(x))))
        x = self.dropout2(self.pool(self.bn1(F.elu(self.conv2(x)))))
        x = self.dropout3(self.pool(self.bn2(F.elu(self.conv3(x)))))
        x = self.dropout4(self.pool(self.bn3(F.elu(self.conv4(x)))))
        
        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        
        # two linear layers with dropout in between
        x = F.elu(self.fc1(x))
        x = self.fc1_drop(x)
        x = self.fc2(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
