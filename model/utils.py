import tensorflow as tf
from tensorflow.keras import layers
#from spectral_normalization import SpectralNormalization


class GenConv(layers.Layer):
    def __init__(self, cnum, ksize, stride=1, rate=1, name='conv',
                padding='SAME', activation=tf.nn.elu, training=True, **kwargs):
        assert padding in ['SYMMETRIC', 'SAME', 'REFELECT']
        super(GenConv, self).__init__(name=name, **kwargs)
        self.padding = padding
        self.ksize = ksize
        self.rate = rate
        self.cnum = cnum
        self.activation = activation
        if self.padding == 'SYMMETRIC' or self.padding == 'REFELECT':
            padding = 'VALID'
       
        self.conv = layers.Conv2D(cnum, ksize, stride, dilation_rate=rate,
        activation=None, padding=padding, name=name)
        #self.norm = SpectralNormalization(self.conv)


    def call(self, x):
        if self.padding == 'SYMMETRIC' or self.padding == 'REFELECT':
            p = int(self.rate*(self.ksize-1)/2)
            x = tf.pad(x, [[0,0], [p, p], [p, p], [0,0]], mode=self.padding)
        x = self.conv(x)
        # x = self.norm(x)
        if self.cnum == 3 or self.activation is None:
            return x
        x, y = tf.split(x, 2, 3)
        x = self.activation(x)
        y = tf.nn.sigmoid(y)
        x = x * y
        return x

class GenDeconv(layers.Layer):
    def __init__(self, cnum, name='upsample', padding='SAME', training=True, **kwargs):
        super(GenDeconv, self).__init__(name=name, **kwargs)
        self.upsample = layers.UpSampling2D(2)
        self.conv = GenConv(cnum, 3, 1, name=name+'_conv', padding=padding, training=training)
    
    def call(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x
        