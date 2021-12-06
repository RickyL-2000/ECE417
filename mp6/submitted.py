from typing import Any, Tuple

from torch import Tensor, matmul, norm, randn, zeros, zeros_like, cat, stack, transpose, split
from torch.nn import BatchNorm1d, Conv1d, Module, ModuleList, Parameter, ReLU, Sequential, Sigmoid, Tanh, LSTMCell, GRUCell
import torch.nn.functional as F
# from torch.autograd import Variable
from torch.nn.parameter import Parameter

from torchtyping import TensorType

################################################################################

class LineEar(Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        """
        Sets up the following Parameters:
            self.weight - A Parameter holding the weights of the layer,
                of size (output_size, input_size).
            self.bias - A Parameter holding the biases of the layer,
                of size (output_size,).
        You may also set other instance variables at this point, but these are not strictly necessary.
        """
        super(LineEar, self).__init__()
        self.weight = Parameter(zeros((output_size, input_size)))
        self.bias = Parameter(zeros(output_size))
        # raise NotImplementedError("You need to implement this!")

    def forward(self,
                inputs: TensorType["batch", "input_size"]) -> TensorType["batch", "output_size"]:
        """
        Performs forward propagation of the inputs.
        Input:
            inputs - the inputs to the cell.
        Output:
            outputs - the outputs from the cell.
        Note that all dimensions besides the last are preserved
            between inputs and outputs.
        """
        return matmul(inputs, self.weight.T) + self.bias
        raise NotImplementedError("You need to implement this!")

class EllEssTeeEmmCell(Module):
    def __init__(self, input_size: int, hidden_size: int, bidirectional: bool=False) -> None:
        super(EllEssTeeEmmCell, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.bidirectional = bidirectional

        self.weight_ih = Parameter(zeros((4*hidden_size, input_size)))
        self.weight_hh = Parameter(zeros((4*hidden_size, hidden_size)))
        self.bias_ih = Parameter(zeros(4*hidden_size))
        self.bias_hh = Parameter(zeros(4*hidden_size))

        self.i_act = Sigmoid()
        self.f_act = Sigmoid()
        self.g_act = Tanh()
        self.o_act = Sigmoid()
        self.h_act = Tanh()

    def forward(self, x: TensorType["batch", "input_size"],
                h_old: TensorType["batch", "hidden_size"],
                c_old: TensorType["batch", "hidden_size"]) -> TensorType["batch", "output_size"]:
        W_ii = self.weight_ih[:self.hidden_size, :]     # [hidden_size, input_size]
        W_if = self.weight_ih[self.hidden_size: 2*self.hidden_size, :]
        W_ig = self.weight_ih[2*self.hidden_size: 3*self.hidden_size, :]
        W_io = self.weight_ih[3*self.hidden_size:]

        W_hi = self.weight_hh[:self.hidden_size, :]     # [hidden_size, hidden_size]
        W_hf = self.weight_hh[self.hidden_size: 2*self.hidden_size, :]
        W_hg = self.weight_hh[2*self.hidden_size: 3*self.hidden_size, :]
        W_ho = self.weight_hh[3*self.hidden_size:]

        b_ii = self.bias_ih[:self.hidden_size]
        b_if = self.bias_ih[self.hidden_size: 2*self.hidden_size]
        b_ig = self.bias_ih[2*self.hidden_size: 3*self.hidden_size]
        b_io = self.bias_ih[3*self.hidden_size]

        b_hi = self.bias_hh[:self.hidden_size]
        b_hf = self.bias_hh[self.hidden_size: 2*self.hidden_size]
        b_hg = self.bias_hh[2*self.hidden_size: 3*self.hidden_size]
        b_ho = self.bias_hh[3*self.hidden_size]

        I = self.i_act(matmul(x, W_ii.T) + b_ii + matmul(h_old, W_hi.T) + b_hi)   # ["batch", "hidden_size"]
        F = self.f_act(matmul(x, W_if.T) + b_if + matmul(h_old, W_hf.T) + b_hf)
        G = self.g_act(matmul(x, W_ig.T) + b_ig + matmul(h_old, W_hg.T) + b_hg)
        O = self.o_act(matmul(x, W_io.T) + b_io + matmul(h_old, W_ho.T) + b_ho)

        C = F * c_old + I * G   # ["batch", "hidden_size"]
        H = O * self.h_act(C)

        return H, C

class EllEssTeeEmm_Custom(Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, bidirectional: bool=False) -> None:
        """
        Sets up the following:
            self.forward_layers - A ModuleList of num_layers EllEssTeeEmmCell layers.
                The first layer should have an input size of input_size
                    and an output size of hidden_size,
                while all other layers should have input and output both of size hidden_size.
        
        If bidirectional is True, then the following apply:
          - self.reverse_layers - A ModuleList of num_layers EllEssTeeEmmCell layers,
                of the exact same size and structure as self.forward_layers.
          - In both self.forward_layers and self.reverse_layers,
                all layers other than the first should have an input size of two times hidden_size.
        """
        # reference: https://www.cnblogs.com/picassooo/p/13504533.html

        super(EllEssTeeEmm_Cus, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.output_size = self.hidden_size * (1 + self.bidirectional)

        self.forward_layers = ModuleList()
        self.reverse_layers = ModuleList()
        if not bidirectional:
            for i in range(num_layers):
                self.forward_layers.append(EllEssTeeEmmCell(input_size if i == 0 else hidden_size, hidden_size))
        else:
            for i in range(num_layers):
                self.forward_layers.append(EllEssTeeEmmCell(input_size if i == 0 else hidden_size*2, hidden_size))
                self.reverse_layers.append(EllEssTeeEmmCell(input_size if i == 0 else hidden_size*2, hidden_size))


        # raise NotImplementedError("You need to implement this!")

    def forward(self,
                x: TensorType["batch", "length", "input_size"]) -> TensorType["batch", "length", "output_size"]:
        """
        Performs the forward propagation of an EllEssTeeEmm layer.
        Inputs:
           x - The inputs to the cell.
        Outputs:
           output - The resulting (hidden state) output h.
               If bidirectional was True when initializing the EllEssTeeEmm layer, then the "output_size"
               of the output should be twice the hidden_size.
               Otherwise, this "output_size" should be exactly the hidden size.
        """
        B, L, input_size = x.shape

        Output = zeros((B, L, self.output_size))
        if not self.bidirectional:
            H = zeros((self.num_layers, B, L, self.hidden_size))
            for i in range(self.num_layers):
                c = zeros((B, self.hidden_size))
                h = zeros((B, self.hidden_size))
                for t in range(L):
                    if i == 0:
                        h, c = self.forward_layers[i].forward(x[:, t, :], h, c) # [B, hidden_size]
                    else:
                        h, c = self.forward_layers[i].forward(H[i-1, :, t, :], h, c)
                    H[i, :, t, :] = h # store the current layer h as output to the next layer
                    if i == self.num_layers - 1:
                        # last layer
                        Output[:, t, :] = h
        else:
            H_forward = zeros((self.num_layers, B, L, self.hidden_size))
            H_reverse = zeros((self.num_layers, B, L, self.hidden_size))
            for i in range(self.num_layers):
                c_forward = zeros((B, self.hidden_size))
                h_forward = zeros((B, self.hidden_size))
                c_reverse = zeros((B, self.hidden_size))
                h_reverse = zeros((B, self.hidden_size))
                for t in range(L):
                    if i == 0:
                        h_forward, c_forward = self.forward_layers[i].forward(x[:, t, :], h_forward, c_forward)
                        h_reverse, c_reverse = self.reverse_layers[i].forward(x[:, -1-t, :], h_reverse, c_reverse)
                    else:
                        h_forward, c_forward = self.forward_layers[i].forward(
                            cat((H_forward[i-1, :, t, :], H_reverse[i-1, :, t, :]), dim=-1), h_forward, c_forward)
                        h_reverse, c_reverse = self.reverse_layers[i].forward(
                            cat((H_forward[i-1, :, -1-t, :], H_reverse[i-1, :, -1-t, :]), dim=-1), h_reverse, c_reverse) 
                    H_forward[i, :, t, :] = h_forward
                    H_reverse[i, :, -1-t, :] = h_reverse
            # because it's bidirectional, we need to finish the loop before assign outputs
            Output[:, :, :] = cat((H_forward[-1, :, :, :], H_reverse[-1, :, :, :]), dim=2)

        return Output

        # raise NotImplementedError("You need to implement this!")

class EllEssTeeEmm(Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, bidirectional: bool=False) -> None:
        """
        Sets up the following:
            self.forward_layers - A ModuleList of num_layers EllEssTeeEmmCell layers.
                The first layer should have an input size of input_size
                    and an output size of hidden_size,
                while all other layers should have input and output both of size hidden_size.
        
        If bidirectional is True, then the following apply:
          - self.reverse_layers - A ModuleList of num_layers EllEssTeeEmmCell layers,
                of the exact same size and structure as self.forward_layers.
          - In both self.forward_layers and self.reverse_layers,
                all layers other than the first should have an input size of two times hidden_size.
        """
        # reference: https://www.cnblogs.com/picassooo/p/13504533.html

        super(EllEssTeeEmm, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.output_size = self.hidden_size * (1 + self.bidirectional)

        self.forward_layers = ModuleList()
        self.reverse_layers = ModuleList()
        for i in range(num_layers):
            self.forward_layers.append(LSTMCell(input_size if i == 0 else hidden_size*(1+bidirectional), hidden_size))
            if bidirectional:
                self.reverse_layers.append(LSTMCell(input_size if i == 0 else hidden_size*2, hidden_size))

        # raise NotImplementedError("You need to implement this!")

    def forward(self,
                x: TensorType["batch", "length", "input_size"]) -> TensorType["batch", "length", "output_size"]:
        """
        Performs the forward propagation of an EllEssTeeEmm layer.
        Inputs:
           x - The inputs to the cell.
        Outputs:
           output - The resulting (hidden state) output h.
               If bidirectional was True when initializing the EllEssTeeEmm layer, then the "output_size"
               of the output should be twice the hidden_size.
               Otherwise, this "output_size" should be exactly the hidden size.
        """
        B, L, input_size = x.shape

        if not self.bidirectional:
            H = zeros((self.num_layers, B, L, self.hidden_size))
            for i in range(self.num_layers):
                c = zeros((B, self.hidden_size))
                h = zeros((B, self.hidden_size))
                for t in range(L):
                    if i == 0:
                        h, c = self.forward_layers[i].forward(x[:, t, :], (h, c)) # [B, hidden_size]
                    else:
                        h, c = self.forward_layers[i].forward(H[i-1, :, t, :], (h, c))
                    H[i, :, t, :] = h # store the current layer h as output to the next layer
            return H[-1, :, :, :]
        else:
            H_forward = zeros((self.num_layers, B, L, self.hidden_size))
            H_reverse = zeros((self.num_layers, B, L, self.hidden_size))
            for i in range(self.num_layers):
                c_forward = zeros((B, self.hidden_size))
                h_forward = zeros((B, self.hidden_size))
                c_reverse = zeros((B, self.hidden_size))
                h_reverse = zeros((B, self.hidden_size))
                for t in range(L):
                    if i == 0:
                        h_forward, c_forward = self.forward_layers[i].forward(x[:, t, :], (h_forward, c_forward))
                        h_reverse, c_reverse = self.reverse_layers[i].forward(x[:, -1-t, :], (h_reverse, c_reverse))
                    else:
                        h_forward, c_forward = self.forward_layers[i].forward(
                            cat((H_forward[i-1, :, t, :], H_reverse[i-1, :, t, :]), dim=-1), (h_forward, c_forward))
                        h_reverse, c_reverse = self.reverse_layers[i].forward(
                            cat((H_forward[i-1, :, -1-t, :], H_reverse[i-1, :, -1-t, :]), dim=-1), (h_reverse, c_reverse)) 
                    H_forward[i, :, t, :] = h_forward
                    H_reverse[i, :, -1-t, :] = h_reverse
            # because it's bidirectional, we need to finish the loop before assign outputs
            # Output[:, :, :] = cat((H_forward[-1, :, :, :], H_reverse[-1, :, :, :]), dim=2)
            return cat((H_forward[-1, :, :, :], H_reverse[-1, :, :, :]), dim=2)
        

        # raise NotImplementedError("You need to implement this!")

class GeeArrYou(Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float=0) -> None:
        """
        Sets up the following:
            self.forward_layers - A ModuleList of num_layers GeeArrYouCell layers.
                The first layer should have an input size of input_size
                    and an output size of hidden_size,
                while all other layers should have input and output both of size hidden_size.
            self.dropout - A dropout probability, usable as the "p" value of F.dropout.
        """
        super(GeeArrYou, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout

        self.forward_layers = ModuleList()
        for i in range(num_layers):
            self.forward_layers.append(GRUCell(input_size if i == 0 else hidden_size, hidden_size))

        # raise NotImplementedError("You need to implement this!")

    def forward(self, x: TensorType["batch", "length", "input_size"]) -> TensorType["batch", "length", "hidden_size"]:
        """
        Performs the forward propagation of a GeeArrYou layer.
        Inputs:
           x - The inputs to the cell.
        Outputs:
           output - The resulting (hidden state) output h.
               Note that the input to each GeeArrYouCell (except the first) should be
                   passed through F.dropout with the dropout probability provided when
                   initializing the GeeArrYou layer.
        """

        B, L, _ = x.shape

        H = zeros((self.num_layers, B, L, self.hidden_size))
        for i in range(self.num_layers):
            h = zeros((B, self.hidden_size))
            for t in range(L):
                if i == 0:
                    h = self.forward_layers[i].forward(x[:, t, :], h)
                else:
                    h = F.dropout(h, p=self.dropout_p)
                    h = self.forward_layers[i].forward(H[i-1, :, t, :], h)
                H[i, :, t, :] = h
        return H[-1, :, :, :]

        raise NotImplementedError("You need to implement this!")

class Encoder(Module):
    """Encoder module (Figure 3 (a) in the AutoVC paper).
    """
    def __init__(self, dim_neck: int, dim_emb: int, freq: int):
        """
        Sets up the following:
            self.convolutions - the 1-D convolution layers.
                The first should have 80 + dim_emb input channels and 512 output channels,
                    while each following convolution layer should have 512 input and 512 output channels.
                    All such layers should have a 5x5 kernel, with a stride of 1,
                    a dilation of 1, and a padding of 2.
                The output of each convolution layer should be fed into a BatchNorm1d layer of 512 input features,
                and the output of each BatchNorm1d should be fed into a ReLU layer.
            self.recurrents - a bidirectional EllEssTeeEmm with two layers, an input size of 512,
                and an output size of dim_neck.
        """
        super(Encoder, self).__init__()

        self.dim_neck = dim_neck
        self.dim_emb = dim_emb

        self.convolutions = Sequential(
            Conv1d(80+dim_emb, 512, kernel_size=5, stride=1, padding=2, dilation=1),
            BatchNorm1d(512),
            ReLU(),
            Conv1d(512, 512, kernel_size=5, stride=1, padding=2, dilation=1),
            BatchNorm1d(512),
            ReLU(),
            Conv1d(512, 512, kernel_size=5, stride=1, padding=2, dilation=1),
            BatchNorm1d(512),
            ReLU()
        )

        self.recurrent = EllEssTeeEmm(input_size=512, hidden_size=dim_neck, num_layers=2, bidirectional=True)

        # raise NotImplementedError("You need to implement this!")

    def forward(self, x: TensorType["batch", "input_dim", "length"]) -> Tuple[
        TensorType["batch", "length", "dim_neck"],
        TensorType["batch", "length", "dim_neck"]
    ]:
        """
        Performs the forward propagation of the AutoVC encoder.
        After passing the input through the convolution layers, the last two dimensions
            should be transposed before passing those layers' output through the EllEssTeeEmm.
            The output from the EllEssTeeEmm should then be split *along the last dimension* into two chunks,
            one for the forward direction (the first self.recurrent_hidden_size columns)
            and one for the backward direction (the last self.recurrent_hidden_size columns).
        """
        output = self.convolutions(x)
        output = self.recurrent(transpose(output, -1, -2))

        return split(output, self.dim_neck, dim=-1)

        raise NotImplementedError("You need to implement this!")
      
class Decoder(Module):
    """Decoder module (Figure 3 (c) in the AutoVC paper, up to the "1x1 Conv").
    
    
    """
    def __init__(self, dim_neck: int, dim_emb: int, dim_pre: int) -> None:
        """
        Sets up the following:
            self.recurrent1 - a unidirectional EllEssTeeEmm with one layer, an input size of 2*dim_neck + dim_emb
                and an output size of dim_pre.
            self.convolutions - the 1-D convolution layers.
                Each convolution layer should have dim_pre input and dim_pre output channels.
                All such layers should have a 5x5 kernel, with a stride of 1,
                a dilation of 1, and a padding of 2.
                The output of each convolution layer should be fed into a BatchNorm1d layer of dim_pre input features,
                and the output of that BatchNorm1d should be fed into a ReLU.
            self.recurrent2 - a unidirectional EllEssTeeEmm with two layers, an input size of dim_pre
                and an output size of 1024.
            self.fc_projection = a LineEar layer with an input size of 1024 and an output size of 80.
        """
        super(Decoder, self).__init__()

        self.dim_neck = dim_neck
        self.dim_emb = dim_emb
        self.dim_pre = dim_pre

        self.recurrent1 = EllEssTeeEmm(input_size=2*dim_neck+dim_emb, hidden_size=dim_pre, num_layers=1, bidirectional=False)
        self.convolutions = Sequential(
            Conv1d(dim_pre, dim_pre, kernel_size=5, stride=1, padding=2, dilation=1),
            BatchNorm1d(dim_pre),
            ReLU(),
            Conv1d(dim_pre, dim_pre, kernel_size=5, stride=1, padding=2, dilation=1),
            BatchNorm1d(dim_pre),
            ReLU(),
            Conv1d(dim_pre, dim_pre, kernel_size=5, stride=1, padding=2, dilation=1),
            BatchNorm1d(dim_pre),
            ReLU()
        )
        self.recurrent2 = EllEssTeeEmm(input_size=dim_pre, hidden_size=1024, num_layers=2, bidirectional=False)
        self.fc_projection = LineEar(input_size=1024, output_size=80)

        # raise NotImplementedError("You need to implement this!")

    def forward(self, x: TensorType["batch", "input_length", "input_dim"]) -> TensorType["batch", "input_length", "output_dim"]:
        """
        Performs the forward propagation of the AutoVC decoder.
            It should be enough to pass the input through the first EllEssTeeEmm,
            the convolution layers, the second EllEssTeeEmm, and the final LineEar
            layer in that order--except that the "input_length" and "input_dim" dimensions
            should be transposed before input to the convolution layers, and this transposition
            should be undone before input to the second EllEssTeeEmm.
        """
        output = self.recurrent1(x)
        output = self.convolutions(transpose(output, -1, -2))
        output = self.recurrent2(transpose(output, -1, -2))
        output = self.fc_projection(output)

        return output

        raise NotImplementedError("You need to implement this!")
    
class Postnet(Module):
    """Post-network module (in Figure 3 (c) in the AutoVC paper,
           the two layers "5x1 ConvNorm x 4" and "5x1 ConvNorm".).
    """
    def __init__(self) -> None:
        """
        Sets up the following:
            self.convolutions - a Sequential object with five Conv1d layers, each with 5x5 kernels,
            a stride of 1, a padding of 2, and a dilation of 1:
                The first should take an 80-channel input and yield a 512-channel output.
                The next three should take 512-channel inputs and yield 512-channel outputs.
                The last should take a 512-channel input and yield an 80-channel output.
                Each layer's output should be passed into a BatchNorm1d,
                and (except for the last layer) from there through a Tanh,
                before being sent to the next layer.
        """
        super(Postnet, self).__init__()

        self.convolutions = Sequential(
            Conv1d(80, 512, kernel_size=5, stride=1, padding=2, dilation=1),
            BatchNorm1d(512),
            Tanh(),
            Conv1d(512, 512, kernel_size=5, stride=1, padding=2, dilation=1),
            BatchNorm1d(512),
            Tanh(),
            Conv1d(512, 512, kernel_size=5, stride=1, padding=2, dilation=1),
            BatchNorm1d(512),
            Tanh(),
            Conv1d(512, 512, kernel_size=5, stride=1, padding=2, dilation=1),
            BatchNorm1d(512),
            Tanh(),
            Conv1d(512, 80, kernel_size=5, stride=1, padding=2, dilation=1),
            BatchNorm1d(80),
            # ReLU(),
        )

        # raise NotImplementedError("You need to implement this!")

    def forward(self, x: TensorType["batch", "input_channels", "n_mels"]) -> TensorType["batch", "input_channels", "n_mels"]:
        """
        Performs the forward propagation of the AutoVC decoder.
        If you initialized this module properly, passing the input through self.convolutions here should suffice.
        """
        return self.convolutions(x)
        raise NotImplementedError("You need to implement this!")

class SpeakerEmbedderGeeArrYou(Module):
    """
    """
    def __init__(self, n_hid: int, n_mels: int, n_layers: int, fc_dim: int, hidden_p: float) -> None:
        """
        Sets up the following:
            self.rnn_stack - an n_layers-layer GeeArrYou with n_mels input features,
                n_hid hidden features, and a dropout of hidden_p.
            self.projection - a LineEar layer with an input size of n_hid
                and an output size of fc_dim.
        """
        super(SpeakerEmbedderGeeArrYou, self).__init__()
        raise NotImplementedError("You need to implement this!")
        
    def forward(self, x: TensorType["batch", "frames", "n_mels"]) -> TensorType["batch", "fc_dim"]:
        """
        Performs the forward propagation of the SpeakerEmbedderGeeArrYou.
            After passing the input through the RNN, the last frame of the output
            should be taken and passed through the fully connected layer.
            Each of the frames should then be normalized so that its Euclidean norm is 1.
        """
        raise NotImplementedError("You need to implement this!")
