import torch
import torch.nn as nn

class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_size, num_classes):
        '''
        :param input_dim: input feature dimension
        :param hidden_size: hidden dimension
        :param num_classes: total number of classes
        '''
        super(TwoLayerNet, self).__init__()
        #############################################################################
        # TODO: Initialize the TwoLayerNet, use sigmoid activation between layers   #
        #############################################################################
        self.l_one = nn.Linear(input_dim,hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.l_two = nn.Linear(hidden_size,num_classes)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        out = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        x = x.view(x.size(0), -1)
        out = self.l_one(x)
        out = self.sigmoid(out)
        out = self.l_two(out)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return out