import numpy as np
import tensorflow as tf

def get_transform(FLAGS):
  # Generate random tansformation
  ang = np.deg2rad(1.0)
  tx = tf.random.uniform(shape = [FLAGS.batch_size, 1], minval = -2.0, maxval = 2.0, dtype = tf.float32)
  ty = tf.random.uniform(shape = [FLAGS.batch_size, 1], minval = -2.0, maxval = 2.0, dtype = tf.float32)
  r = tf.random.uniform(shape = [FLAGS.batch_size, 1], minval = -ang, maxval = ang, dtype = tf.float32)
  z = tf.random.uniform(shape = [FLAGS.batch_size, 1], minval = 0.97, maxval = 1.03, dtype = tf.float32)
  hx = tf.random.uniform(shape = [FLAGS.batch_size, 1], minval = -ang, maxval = ang, dtype = tf.float32)
  hy = tf.random.uniform(shape = [FLAGS.batch_size, 1], minval = -ang, maxval = ang, dtype = tf.float32)
  sx = FLAGS.img_shapes[0]
  sy = FLAGS.img_shapes[1]

  # Construct transformation matrix
  a = hx - r
  b = hy + r
  T1 = tf.divide(z*tf.cos(a), tf.cos(hx))
  T2 = tf.divide(z*tf.sin(a), tf.cos(hx)) 
  T3 = tf.divide(sx*tf.cos(hx)-sx*z*tf.cos(a) + 2*tx*z*tf.cos(a) - sy*z*tf.sin(a) + 2*ty*z*tf.sin(a), 2*tf.cos(hx))
  T4 = tf.divide(z*tf.sin(b), tf.cos(hy))
  T5 = tf.divide(z*tf.cos(b), tf.cos(hy))
  T6 = tf.divide(sy*tf.cos(hy)-sy*z*tf.cos(b)+2*ty*z*tf.cos(b)-sx*z*tf.sin(b)+2*tx*z*tf.sin(b), 2*tf.cos(hy))
  T7 = tf.zeros ([FLAGS.batch_size, 2], 'float32')
  T = tf.concat([T1, T2, T3, T4, T5, T6, T7], 1)
  return T