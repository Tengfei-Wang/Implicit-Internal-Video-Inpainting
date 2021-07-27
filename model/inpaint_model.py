import numpy as np
import tensorflow as tf
from tensorflow import keras
from model.utils import GenConv, GenDeconv
from tensorflow.keras import layers

class RefineModel(keras.Model):
    def __init__(self, training=True, padding='SAME', name='inpaintingModel', **kwargs):
        # CoarseNet:
        cnum = 48
        super(RefineModel, self).__init__(name=name, **kwargs)
        self.coarse_net = keras.Sequential(
            [
                GenConv(cnum, 5, 1, training=training, padding=padding, name='conv1'),
                GenConv(2*cnum, 3, 2, training=training, padding=padding, name='conv2'),
                GenConv(2*cnum, 3, 1, training=training, padding=padding, name='conv3'),
                GenConv(4*cnum, 3, 2, training=training, padding=padding, name='conv4'),
                GenConv(4*cnum, 3, 1, training=training, padding=padding, name='conv5'),
                GenConv(4*cnum, 3, 1, training=training, padding=padding, name='conv6'),
                GenConv(4*cnum, 3, rate=2, training=training, padding=padding, name='conv7'),
                GenConv(4*cnum, 3, rate=4, training=training, padding=padding, name='conv8'),
                GenConv(4*cnum, 3, rate=8, training=training, padding=padding, name='conv9'),
                GenConv(4*cnum, 3, rate=16, training=training, padding=padding, name='conv10'),
                GenConv(4*cnum, 3, 1, training=training, padding=padding, name='conv11'),
                GenConv(4*cnum, 3, 1, training=training, padding=padding, name='conv12'),
                GenDeconv(2*cnum, training=training, padding=padding, name='conv13'),
                GenConv(2*cnum, 3, 1, training=training, padding=padding, name='conv14'),
                GenDeconv(cnum, training=training, padding=padding, name='conv15'),
                GenConv(cnum//2, 3, 1, training=training, padding=padding, name='conv16'),
                GenConv(3, 3, 1, activation=None, training=training, padding=padding, name='conv17')
            ]
        )
        # FineNet:
        # Main Conv branch
        self.conv_branch = keras.Sequential(
            [
                GenConv(cnum, 5, 1, training=training, padding=padding, name='fineconv1'),
                GenConv(cnum, 3, 2, training=training, padding=padding, name='fineconv2'),
                GenConv(2*cnum, 3, 1, training=training, padding=padding, name='fineconv3'),
                GenConv(2*cnum, 3, 2, training=training, padding=padding, name='fineconv4'),
                GenConv(4*cnum, 3, 1, training=training, padding=padding, name='fineconv5'),
                GenConv(4*cnum, 3, 1, training=training, padding=padding, name='fineconv6'),
                GenConv(4*cnum, 3, rate=2, training=training, padding=padding, name='fineconv7'),
                GenConv(4*cnum, 3, rate=4, training=training, padding=padding, name='fineconv8'),
                GenConv(4*cnum, 3, rate=8, training=training, padding=padding, name='fineconv9'),
                GenConv(4*cnum, 3, rate=16, training=training, padding=padding, name='fineconv10')
            ]
        )
        # Another Conv branch 
        self.conv_branch2 = keras.Sequential(
            [
                GenConv(cnum, 5, 1, training=training, padding=padding, name='fineconv2_1'),
                GenConv(cnum, 3, 2, training=training, padding=padding, name='fineconv2_2'),
                GenConv(2*cnum, 3, 1, training=training, padding=padding, name='fineconv2_3'),
                GenConv(4*cnum, 3, 2, training=training, padding=padding, name='fineconv2_4'),
                GenConv(4*cnum, 3, 1, training=training, padding=padding, name='fineconv2_5'),
                GenConv(4*cnum, 3, 1, training=training, padding=padding, name='fineconv2_6', activation=tf.nn.relu),
                GenConv(4*cnum, 3, 1, training=training, padding=padding, name='fineconv2_9'),
                GenConv(4*cnum, 3, 1, training=training, padding=padding, name='fineconv2_10')
            ]
        )
        # Out branch
        self.out_branch = keras.Sequential(
            [
                GenConv(4*cnum, 3, 1, training=training, padding=padding, name='outconv1'),
                GenConv(4*cnum, 3, 1, training=training, padding=padding, name='outconv2'),
                GenDeconv(2*cnum, training=training, padding=padding, name='outconv3'),
                GenConv(2*cnum, 3, 1, training=training, padding=padding, name='outconv4'),
                GenDeconv(cnum, training=training, padding=padding, name='outconv5'),
                GenConv(cnum//2, 3, 1, training=training, padding=padding, name='outconv6'),
                GenConv(3, 3, 1, activation=None, training=training, padding=padding, name='outconv7')
            ]
        )

    def call(self, x, mask): 
        """
        Args:
            x: incomplete image, [-1, 1]
            mask: mask region {0, 1}
        Returns:
            [-1, 1] as predicted image
        """
        # input
        xin = x

        # stage 1
        x = self.coarse_net(x)
        x = tf.nn.tanh(x)
        x_stage1 = x


        # stage 2
        x = x*mask + xin[:, :, :, 0:3]*(1.-mask)
        x.set_shape(xin[:, :, :, 0:3].get_shape().as_list())
        # first branch
        xnow = x
        x = self.conv_branch(xnow)
        x_hallu = x
        # second branch
        x = self.conv_branch2(xnow)
        pm = x
        # out branch
        x = tf.concat([x_hallu, pm], axis=3)
        x = self.out_branch(x)
        x = tf.nn.tanh(x)
        x_stage2 = x
        return x_stage1, x_stage2

class StackModel(keras.Model):
    def __init__(self, training=True, padding='SAME', name='inpaintingModel', **kwargs):
        # CoarseNet:
        cnum = 48
        super(StackModel, self).__init__(name=name, **kwargs)
        self.coarse_net = keras.Sequential(
            [
                GenConv(cnum, 5, 1, training=training, padding=padding, name='conv1'),
                GenConv(2*cnum, 3, 2, training=training, padding=padding, name='conv2_downsample'),
                GenConv(2*cnum, 3, 1, training=training, padding=padding, name='conv3'),
                GenConv(4*cnum, 3, 2, training=training, padding=padding, name='conv4_downsample'),
                GenConv(4*cnum, 3, 1, training=training, padding=padding, name='conv5'),
                GenConv(4*cnum, 3, 1, training=training, padding=padding, name='conv6'),
                GenConv(4*cnum, 3, rate=2, training=training, padding=padding, name='conv7_atrous'),
                GenConv(4*cnum, 3, rate=4, training=training, padding=padding, name='conv8_atrous'),
                GenConv(4*cnum, 3, rate=8, training=training, padding=padding, name='conv9_atrous'),
                GenConv(4*cnum, 3, rate=16, training=training, padding=padding, name='conv10_atrous'),
                GenConv(4*cnum, 3, 1, training=training, padding=padding, name='conv11'),
                GenConv(4*cnum, 3, 1, training=training, padding=padding, name='conv12'),
                GenDeconv(2*cnum, training=training, padding=padding, name='conv13_upsample'),
                GenConv(2*cnum, 3, 1, training=training, padding=padding, name='conv14'),
                GenDeconv(cnum, training=training, padding=padding, name='conv15_upsample'),
                GenConv(cnum//2, 3, 1, training=training, padding=padding, name='conv16'),
                GenConv(3, 3, 1, activation=None, training=training, padding=padding, name='conv17')
            ]
        )
        # FineNet:
        # Conv branch
        self.conv_branch2 = keras.Sequential(
            [
                GenConv(cnum, 5, 1, training=training, padding=padding, name='fineconv2_1'),
                GenConv(cnum, 3, 2, training=training, padding=padding, name='fineconv2_2_downsample'),
                GenConv(2*cnum, 3, 1, training=training, padding=padding, name='fineconv2_3'),
                GenConv(4*cnum, 3, 2, training=training, padding=padding, name='fineconv2_4_downsample'),
                GenConv(4*cnum, 3, 1, training=training, padding=padding, name='fineconv2_5'),
                GenConv(4*cnum, 3, 1, training=training, padding=padding, name='fineconv2_6',
                                    activation=tf.nn.relu),
                GenConv(4*cnum, 3, 1, training=training, padding=padding, name='fineconv2_9'),
                GenConv(4*cnum, 3, 1, training=training, padding=padding, name='fineconv2_10')
            ]
        )
        # Out branch
        self.out_branch = keras.Sequential(
            [
                GenConv(4*cnum, 3, 1, training=training, padding=padding, name='outconv1'),
                GenConv(4*cnum, 3, 1, training=training, padding=padding, name='outconv2'),
                GenDeconv(2*cnum, training=training, padding=padding, name='outconv3_upsample'),
                GenConv(2*cnum, 3, 1, training=training, padding=padding, name='outconv4'),
                GenDeconv(cnum, training=training, padding=padding, name='outconv5_upsample'),
                GenConv(cnum//2, 3, 1, training=training, padding=padding, name='outconv6'),
                GenConv(3, 3, 1, activation=None, training=training, padding=padding, name='outconv7')
            ]
        )

    def call(self, x, mask): 
        """
        Args:
            x: incomplete image, [-1, 1]
            mask: mask region {0, 1}
        Returns:
            [-1, 1] as predicted image
        """
        # input
        xin = x

        # stage 1
        x = self.coarse_net(x)
        x = tf.nn.tanh(x)
        x_stage1 = x

        # stage 2
        x = x*mask + xin[:, :, :, 0:3]*(1.-mask)
        x.set_shape(xin[:, :, :, 0:3].get_shape().as_list())
        # first branch
        x = self.conv_branch2(x)
        x = self.out_branch(x)
        x = tf.nn.tanh(x)
        x_stage2 = x
        return x_stage1, x_stage2



# only consist of the coarse stage
class BaseModel(keras.Model):
    def __init__(self, training=True, padding='SAME', name='inpaintingModel', **kwargs):
        # CoarseNet:
        cnum = 48
        super(BaseModel, self).__init__(name=name, **kwargs)
        self.coarse_net = keras.Sequential(
            [
                GenConv(cnum, 5, 1, training=training, padding=padding, name='conv1'),
                GenConv(2*cnum, 3, 2, training=training, padding=padding, name='conv2_downsample'),
                GenConv(2*cnum, 3, 1, training=training, padding=padding, name='conv3'),
                GenConv(4*cnum, 3, 2, training=training, padding=padding, name='conv4_downsample'),
                GenConv(4*cnum, 3, 1, training=training, padding=padding, name='conv5'),
                GenConv(4*cnum, 3, 1, training=training, padding=padding, name='conv6'),
                GenConv(4*cnum, 3, rate=2, training=training, padding=padding, name='conv7_atrous'),
                GenConv(4*cnum, 3, rate=4, training=training, padding=padding, name='conv8_atrous'),
                GenConv(4*cnum, 3, rate=8, training=training, padding=padding, name='conv9_atrous'),
                GenConv(4*cnum, 3, rate=16, training=training, padding=padding, name='conv10_atrous'),
                GenConv(4*cnum, 3, 1, training=training, padding=padding, name='conv11'),
                GenConv(4*cnum, 3, 1, training=training, padding=padding, name='conv12'),
                GenDeconv(2*cnum, training=training, padding=padding, name='conv13_upsample'),
                GenConv(2*cnum, 3, 1, training=training, padding=padding, name='conv14'),
                GenDeconv(cnum, training=training, padding=padding, name='conv15_upsample'),
                GenConv(cnum//2, 3, 1, training=training, padding=padding, name='conv16'),
                GenConv(3, 3, 1, activation=None, training=training, padding=padding, name='conv17')
            ]
        )

    def call(self, x): 
        """
        Args:
            x: incomplete image, [-1, 1]
            mask: mask region {0, 1}
        Returns:
            [-1, 1] as predicted image
        """

        # stage 1
        x = self.coarse_net(x)
        x = tf.nn.tanh(x)

        return x 


# using only the coarseNet
# concat the upsampled input
class BaseModelUp(keras.Model):
    def __init__(self, training=True, padding='SAME', name='inpaintingModel', **kwargs):
        # CoarseNet:
        cnum = 48
        super(RefineModelUpCoarse, self).__init__(name=name, **kwargs)
        self.coarse_net = keras.Sequential(
            [
                GenConv(cnum, 8, 1, training=training, padding=padding, name='conv1'),
                GenConv(2*cnum, 3, 2, training=training, padding=padding, name='conv2_downsample'),
                GenConv(2*cnum, 3, 1, training=training, padding=padding, name='conv3'),
                GenConv(4*cnum, 3, 2, training=training, padding=padding, name='conv4_downsample'),
                GenConv(4*cnum, 3, 1, training=training, padding=padding, name='conv5'),
                GenConv(4*cnum, 3, 1, training=training, padding=padding, name='conv6'),
                GenConv(4*cnum, 3, rate=2, training=training, padding=padding, name='conv7_atrous'),
                GenConv(4*cnum, 3, rate=4, training=training, padding=padding, name='conv8_atrous'),
                GenConv(4*cnum, 3, rate=8, training=training, padding=padding, name='conv9_atrous'),
                GenConv(4*cnum, 3, rate=16, training=training, padding=padding, name='conv10_atrous'),
                GenConv(4*cnum, 3, 1, training=training, padding=padding, name='conv11'),
                GenConv(4*cnum, 3, 1, training=training, padding=padding, name='conv12'),
                GenDeconv(2*cnum, training=training, padding=padding, name='conv13_upsample'),
                GenConv(2*cnum, 3, 1, training=training, padding=padding, name='conv14'),
                GenDeconv(cnum, training=training, padding=padding, name='conv15_upsample'),
                GenConv(cnum//2, 3, 1, training=training, padding=padding, name='conv16'),
                GenConv(3, 3, 1, activation=None, training=training, padding=padding, name='conv17')
            ]
        )

    def call(self, x, mask, x_coarse): 
        """
        Args:
            x: incomplete image, [-1, 1]
            mask: mask region {0, 1}
        Returns:
            [-1, 1] as predicted image
        """
        # input
        xin = x

        # stage 1
        x = self.coarse_net(x)
        x = tf.nn.tanh(x)

        return x 


class RefineModelUp(keras.Model):
    def __init__(self, training=True, padding='SAME', name='inpaintingModel', **kwargs):
        cnum = 48
        super(RefineModelUp, self).__init__(name=name, **kwargs)
        # CoarseNet:
        self.coarse_net = keras.Sequential(
            [
                GenConv(cnum, 8, 1, training=training, padding=padding, name='conv1'),
                GenConv(2*cnum, 3, 2, training=training, padding=padding, name='conv2_downsample'),
                GenConv(2*cnum, 3, 1, training=training, padding=padding, name='conv3'),
                GenConv(4*cnum, 3, 2, training=training, padding=padding, name='conv4_downsample'),
                GenConv(4*cnum, 3, 1, training=training, padding=padding, name='conv5'),
                GenConv(4*cnum, 3, 1, training=training, padding=padding, name='conv6'),
                GenConv(4*cnum, 3, rate=2, training=training, padding=padding, name='conv7_atrous'),
                GenConv(4*cnum, 3, rate=4, training=training, padding=padding, name='conv8_atrous'),
                GenConv(4*cnum, 3, rate=8, training=training, padding=padding, name='conv9_atrous'),
                GenConv(4*cnum, 3, rate=16, training=training, padding=padding, name='conv10_atrous'),
                GenConv(4*cnum, 3, 1, training=training, padding=padding, name='conv11'),
                GenConv(4*cnum, 3, 1, training=training, padding=padding, name='conv12'),
                GenDeconv(2*cnum, training=training, padding=padding, name='conv13_upsample'),
                GenConv(2*cnum, 3, 1, training=training, padding=padding, name='conv14'),
                GenDeconv(cnum, training=training, padding=padding, name='conv15_upsample'),
                GenConv(cnum//2, 3, 1, training=training, padding=padding, name='conv16'),
                GenConv(3, 3, 1, activation=None, training=training, padding=padding, name='conv17')
            ]
        )
        # FineNet:
        self.conv_branch = keras.Sequential(
            [
                GenConv(cnum, 5, 1, training=training, padding=padding, name='fineconv1'),
                GenConv(cnum, 3, 2, training=training, padding=padding, name='fineconv2_downsample'),
                GenConv(2*cnum, 3, 1, training=training, padding=padding, name='fineconv3'),
                GenConv(2*cnum, 3, 2, training=training, padding=padding, name='fineconv4_downsample'),
                GenConv(4*cnum, 3, 1, training=training, padding=padding, name='fineconv5'),
                GenConv(4*cnum, 3, 1, training=training, padding=padding, name='fineconv6'),
                GenConv(4*cnum, 3, rate=2, training=training, padding=padding, name='fineconv7_atrous'),
                GenConv(4*cnum, 3, rate=4, training=training, padding=padding, name='fineconv8_atrous'),
                GenConv(4*cnum, 3, rate=8, training=training, padding=padding, name='fineconv9_atrous'),
                GenConv(4*cnum, 3, rate=16, training=training, padding=padding, name='fineconv10_atrous')
            ]
        )
         
        self.conv_branch2 = keras.Sequential(
            [
                GenConv(cnum, 5, 1, training=training, padding=padding, name='fineconv2_1'),
                GenConv(cnum, 3, 2, training=training, padding=padding, name='fineconv2_2_downsample'),
                GenConv(2*cnum, 3, 1, training=training, padding=padding, name='fineconv2_3'),
                GenConv(4*cnum, 3, 2, training=training, padding=padding, name='fineconv2_4_downsample'),
                GenConv(4*cnum, 3, 1, training=training, padding=padding, name='fineconv2_5'),
                GenConv(4*cnum, 3, 1, training=training, padding=padding, name='fineconv2_6',
                                    activation=tf.nn.relu),
                GenConv(4*cnum, 3, 1, training=training, padding=padding, name='fineconv2_9'),
                GenConv(4*cnum, 3, 1, training=training, padding=padding, name='fineconv2_10')
            ]
        )
        # Out branch
        self.out_branch = keras.Sequential(
            [
                GenConv(4*cnum, 3, 1, training=training, padding=padding, name='outconv1'),
                GenConv(4*cnum, 3, 1, training=training, padding=padding, name='outconv2'),
                GenDeconv(2*cnum, training=training, padding=padding, name='outconv3_upsample'),
                GenConv(2*cnum, 3, 1, training=training, padding=padding, name='outconv4'),
                GenDeconv(cnum, training=training, padding=padding, name='outconv5_upsample'),
                GenConv(cnum//2, 3, 1, training=training, padding=padding, name='outconv6'),
                GenConv(3, 3, 1, activation=None, training=training, padding=padding, name='outconv7')
            ]
        )

    def call(self, x, mask, x_coarse): 
        """
        Args:
            x: incomplete image, [-1, 1]
            mask: mask region {0, 1}
        Returns:
            [-1, 1] as predicted image
        """
        # input
        xin = x
 

        # stage 1
        x = self.coarse_net(x)
        x = tf.nn.tanh(x)
        x_stage1 = x

        # stage 2
        x = x_coarse*mask + xin[:, :, :, 0:3]*(1.-mask)
        x.set_shape(xin[:, :, :, 0:3].get_shape().as_list())
        # first branch
        xnow = x
        x = self.conv_branch(xnow)
        x_hallu = x
        # second branch
        x = self.conv_branch2(xnow)
        pm = x
        # out branch
        x = tf.concat([x_hallu, pm], axis=3)
        x = self.out_branch(x)
        x = tf.nn.tanh(x)
        x_stage2 = x
        return x_stage1, x_stage2
 
 
class UnetModel(keras.Model):
    def __init__(self, training=True, name ='Seg',  **kwargs):
        super(UnetModel, self).__init__(**kwargs)
        self.conv1 = keras.Sequential([
          layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'),
          layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
          ])
     
        self.conv2 = keras.Sequential([
          layers.MaxPooling2D(pool_size=(2, 2)),
          layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'),
          layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
          ])
    
        self.conv3 = keras.Sequential([
          layers.MaxPooling2D(pool_size=(2, 2)),
          layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'),
          layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
          ])
          
        self.drop4 = keras.Sequential([
          layers.MaxPooling2D(pool_size=(2, 2)),
          layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'),
          layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'),
          layers.Dropout(0.5)
          ])
    
    
        self.up6 = keras.Sequential([
          layers.MaxPooling2D(pool_size=(2, 2)),
          layers.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'),
          layers.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'),
          layers.Dropout(0.5),
          layers.UpSampling2D(size = (2,2)),
          layers.Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
          ])
    
        self.up7 = keras.Sequential([
          layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'),
          layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'),
          layers.UpSampling2D(size = (2,2)),
          layers.Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
          ])
    
        self.up8 = keras.Sequential([
          layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'),
          layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'),
          layers.UpSampling2D(size = (2,2)),
          layers.Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
          ])
    
        self.up9 = keras.Sequential([    
          layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'),
          layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'),
          layers.UpSampling2D(size = (2,2)),
          layers.Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
          ])
    
        self.conv10 = keras.Sequential([  
          layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'),
          layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'),
          layers.Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'),
          layers.Conv2D(1, 1, activation = 'sigmoid')
          ])      
        
        

    def call(self, x): 
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        drop4 = self.drop4(conv3)
        up6 = self.up6(drop4)
        merge6 = tf.concat([drop4,up6], axis = 3)
        up7 = self.up7(merge6)
        merge7 = tf.concat([conv3,up7], axis = 3)
        up8 = self.up8(merge7)
        merge8 = tf.concat([conv2,up8], axis = 3)
        up9 = self.up9(merge8)
        merge9 = tf.concat([conv1,up9], axis = 3)
        output = self.conv10(merge9)

        return output