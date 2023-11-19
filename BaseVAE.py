from typing import List, Any

from torch import nn

from abc import abstractmethod
# This is a base class for all VAEs (Variational Autoencoders)
# This is an abstract class, so it cannot be instantiated
# This class inherits from nn.Module
from torch._C._te import Tensor


class BaseVAE(nn.Module): #iherit from nn.Module

    # create a constructor
    def __init__(self):
        super(BaseVAE, self).__init__()

    # define the forward pass
    #start with the encoder
    def encode(selfs, input: Tensor) -> List[Tensor]: #Return Type: List[Tensor] (a list of tensors)
        raise NotImplementedError
    '''
    Creating an an encoder function that takes in an input tensor and returns a list of tensors
    '''

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size:int, current_dic, **kwargs) -> Tensor:
        raise NotImplementedError


    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    '''
    abstract method that are not implemented in the base class but must be implemented in the child class
    '''
    @abstractmethod
    def forward(self,*inputs:Tensor,**kwargs) -> Tensor:
        pass

    def loss_function(self, *input, **kwargs) -> Tensor:
        pass









