import torch
import torch.nn as nn


class MyModel(nn.Module):
    # You can use pre-existing models but change layers to recieve full credit.
    def __init__(self):
        super(MyModel, self).__init__()
        #############################################################################
        # TODO: Initialize the network weights                                      #
        #############################################################################
        self.conv_one = nn.Conv2d(3,32,3,1,1)
        self.relu1 = nn.ReLU()
        self.pooling_one = nn.MaxPool2d(2,2)

        self.conv_two = nn.Conv2d(32,64,3,1,1)
        self.relu2 = nn.ReLU()
        self.pooling_two = nn.MaxPool2d(2,2)

        self.conv_three = nn.Conv2d(64,128,3,1,1)
        self.relu3 = nn.ReLU()
        self.pooling_three = nn.MaxPool2d(2,2)

        self.conv_four = nn.Conv2d(128,256,3,1,1)
        self.relu4 = nn.ReLU()
        self.pooling_four = nn.MaxPool2d(2,2)

        self.fully_one = nn.Linear(256*2*2,1024)
        self.relu5 = nn.ReLU()
        self.fully_two = nn.Linear(1024,10)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        outs = self.conv_one(x)
        outs = self.relu1(outs)
        outs = self.pooling_one(outs)

        outs = self.conv_two(outs)
        outs = self.relu2(outs)
        outs = self.pooling_two(outs)

        outs = self.conv_three(outs)
        outs = self.relu3(outs)
        outs = self.pooling_three(outs)

        outs = self.conv_four(outs)
        outs = self.relu4(outs)
        outs = self.pooling_four(outs)

        outs = outs.view(outs.size(0), -1)
        outs = self.fully_one(outs)
        outs = self.relu5(outs)

        outs = self.fully_two(outs)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return outs