import torch
import torch.nn as nn

class VanillaCNN(nn.Module):
    def __init__(self):
        super(VanillaCNN, self).__init__()
        #############################################################################
        # TODO: Initialize the Vanilla CNN                                          #
        #       Conv: 7x7 kernel, stride 1 and padding 0                            #
        #       Max Pooling: 2x2 kernel, stride 2                                   #
        #############################################################################

        self.conv_one = nn.Conv2d(3,32,7,1,0)
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool2d(2, 2)
        self.fully = nn.Linear(32*13*13,10)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################


    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        out = self.conv_one(x)
        out = self.relu(out)
        out = self.pooling(out)
        out = out.view(out.size(0), -1)
        outs = self.fully(out)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return outs