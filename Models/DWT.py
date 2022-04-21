# Copyright (c) Tomohiko Nakamura
# All rights reserved.
# Modified for tensorflow and only haar wavelet
"""Library of discrete wavelet transform layers using fixed and trainable wavelets
"""
import tensorflow as tf

import numpy
class DWT(tf.keras.layers.Layer):
    '''Discrete wavelet transform layer using fixed wavelet
    This layer uses discrete wavelet transform (DWT) for downsampling. It enables us to downsample features without losing their entire information and causing aliasing in the feature domain.
    Attributes:
        p_weight (tf.Tensor): Prediction weight
        p_params (dict): Convolution parameters for the prediction step
        u_weight (tf.Tensor): Update weight
        u_params (dict): Convolution parameters for the update step
        scaling_factor (tf.Tensor): Scaling factor
    Referenes:
        [1] Tomohiko Nakamura, Shihori Kozuka, and Hiroshi Saruwatari, “Time-Domain Audio Source Separation with Neural Networks Based on Multiresolution Analysis,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 29, pp. 1687–1701, Apr. 2021.
    '''
    def __init__(self):
        super().__init__()
        ########
        p_weight = [1]
        self.register_buffer('p_weight', tf.constant(
            p_weight, dtype=float32)[None, None, :])
        self.p_params = dict(bias=False, strides=1, dilation_rate=1, groups=1)
        self.p_padding = (self.p_weight.shape[2]-1)//2
        #
        u_weight = [0.5]
        self.register_buffer('u_weight', tf.constant(u_weight, dtype=float32)[None, None, :])
        self.u_params = dict(bias=False, strides=1, dilation_rate=1, groups=1)
        self.u_padding = (self.u_weight.shape[2]-1)//2
        #
        scaling_factor = numpy.sqrt(2.0)
        self.register_buffer('scaling_factor', tf.constant(scaling_factor, dtype=float32))

    def split(self, x):
        '''Split step
        Args:
            x (tf.Tensor): Input feature (batch x channels x time)
        Returns:
            Tuple[tf.Tensor]: even- and odd-indexed components of `x`, each of which has (batch x channels x time/2) shape.
        '''
        even = x[:, :, ::2]
        odd = x[:, :, 1::2]
        return even, odd

    def predict(self, even):
        '''Predict odd from even
        Args:
            even (tf.Tensor): Even component (batch x ch x time)
        Return:
            tf.Tensor: Predicted odd component (batch x ch x time)
        '''
        return even*self.p_weight[0,0,0]

    def update(self, highfreq):
        '''Smooth even from prediction error
        Args:
            highfreq (tf.Tensor): Prediction error, a.k.a. (unscaled) high-frequency component (batch x ch x time)
        Return:
            tf.Tensor: Residual for smoothed even (batch x ch x time)
        '''
        return highfreq*self.u_weight[0,0,0]

    def call(self, x, no_concat=False):
        '''Forward computation 
        Args:
            x (tf.Tensor): Input feature (batch x ch x time)
            no_concat (bool): If True, return tuple of high- and low-frequency components (default: False)
        
        Return:
            tf.Tensor or Tuple[tf.Tensor, tf.Tensor]: High- and low-frequency components concatenated along the channel axis (batch x ch*2 x time/2) or tuple of them (batch x ch x time/2)
        '''
        assert x.shape[-1]%2 == 0, "Time length must be even."
        even, odd = self.split(x)
        unscaled_highfreq = odd - self.predict(even)
        unscaled_lowfreq = even + self.update(unscaled_highfreq)
        y = unscaled_highfreq/self.scaling_factor, unscaled_lowfreq*self.scaling_factor
        
        if no_concat:
            return y
        else:
            #return torch.cat(y, dim=1)
            return tf.keras.layers.Concatenate(axis=1)(y)

    def inverse(self, x, no_concat=False):
        '''Inverse computation
        Args:
            x (tf.Tensor): High- and low-frequency components concatenated along the channel axis (batch x ch*2 x time/2) or tuple of them (batch x ch x time/2)
            no_concat (bool): If True, `x` is assumed to be tuple of high- and low-frequency components (default: False)
        
        Return:
            tf.Tensor: inverse DWT of x (batch x ch x time)
        '''
        if no_concat:
            assert isinstance(x, tuple), 'If no_concat=True, x must be a tuple of high- and low-frequency components.'
            highfreq, lowfreq = x
        else:
            # First and second ones are respectively high- and low-frequency components.
            C = x.shape[1] // 2
            highfreq = x[:, :C, :]
            lowfreq = x[:, C:, :]

        unscaled_highfreq = highfreq/self.scaling_factor
        unscaled_lowfreq = lowfreq*self.scaling_factor
        even = unscaled_lowfreq - self.update(unscaled_highfreq)
        odd = unscaled_highfreq + self.predict(even)
        y = tf.stack((even, odd), axis=-1)
        y = y.reshape(y.shape[0], y.shape[1], -1)
        
        return y