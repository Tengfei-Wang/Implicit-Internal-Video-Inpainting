import tensorflow as tf
from tensorflow import keras

# https://www.tensorflow.org/tutorials/generative/style_transfer
def vgg_layers(layer_names):
    vgg = keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    return keras.Model([vgg.input], outputs)


def perceptual_loss(img1, img2):
    vgg = vgg_layers( ['block4_conv2'])
    vgg.trainable = False
    img1_ = keras.applications.vgg19.preprocess_input((img1+1)*127.5)
    img2_ = keras.applications.vgg19.preprocess_input((img2+1)*127.5)
    img1_features =  vgg(img1_)
    img2_features =  vgg(img2_)
    loss = tf.reduce_mean(tf.square(img1_features[-1] - img2_features[-1]))
    return loss
    
    
def contextual_loss(img1, img2) :
    vgg = vgg_layers( ['block4_conv2'])
    vgg.trainable = False
    
    img1_ = keras.applications.vgg19.preprocess_input((img1+1)*127.5)
    img2_ = keras.applications.vgg19.preprocess_input((img2+1)*127.5)
    img1_features =  vgg(img1_)
    img2_features =  vgg(img2_)    
    
    img1_features = tf.transpose(img1_features, perm=[0,3,1,2])
    img2_features = tf.transpose(img2_features, perm=[0,3,1,2])
  
    dist = compute_cosine_distance(img1_features, img2_features)
   
    d_min = tf.reduce_min(dist, axis=2, keepdims=True)
    d_tilde = dist / (d_min + 1e-5)
    w = tf.math.exp((1 - d_tilde)*2)
    cx_ij = w / tf.reduce_sum(w, axis=2, keepdims=True)
    cx = tf.reduce_mean(tf.reduce_max(cx_ij, axis=1), axis=1)
  
    cx_loss = tf.reduce_mean(-tf.math.log(cx))
    return cx_loss
  

def compute_cosine_distance(x,y) :
    N, C, H, W = x.shape
    y_mu = tf.reduce_mean(y, axis=[0,2,3], keepdims=True)
  
    x_centered = x - y_mu
    y_centered = y - y_mu
  
    x_normalized = x_centered / tf.norm(x_centered, ord=2, axis=1, keepdims=True)
    y_normalized = y_centered / tf.norm(y_centered, ord=2, axis=1, keepdims=True)
  
    x_normalized = tf.reshape(x_normalized, [N,C,-1])
    y_normalized = tf.reshape(y_normalized, [N,C,-1])
  
    x_normalized = tf.transpose(x_normalized, perm=[0,2,1])

    cosine_sim = tf.matmul(x_normalized, y_normalized)
  
    dist = 1 - cosine_sim
  
    return dist