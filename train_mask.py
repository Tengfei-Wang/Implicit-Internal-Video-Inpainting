import os
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
import numpy as np
import data.dataloader as dl 
from config.config import Config
from model.inpaint_model import UnetModel

if __name__ == "__main__":
    # read config 
    FLAGS = Config('config/train_seg.yml')
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU_ID 
    

    # define the model
    model = UnetModel()
    if not FLAGS.model_restore=="":
        model.load_weights(FLAGS.model_restore)

    # define the optimizer
    optimizer = keras.optimizers.Adam(learning_rate=FLAGS.lr, beta_1=0.9, beta_2=0.999)

    # define the dataloader 
    full_ds = dl.build_dataset_seg(FLAGS.dir_video, FLAGS.dir_mask,
                                FLAGS.batch_size, FLAGS.max_epochs, FLAGS.img_shapes[0], FLAGS.img_shapes[1])
    #summary writer
    writer = tf.summary.create_file_writer(FLAGS.log_dir)

    # define the training steps and loss
    @tf.function
    def training_step(batch_data, step, shift_h, shift_w):
        batch_pos = batch_data[0]
        mask1 = batch_data[1] > 0.8
        mask1 = tf.cast(mask1, tf.float32)
        
        mask2 = []
        img_shift = []
        for i in range(batch_pos.shape[0]):
          mask2_ = tf.roll(tf.expand_dims(mask1[0], 0), (shift_h[i], shift_w[i]), axis=(1,2))  
          img_shift_ = tf.roll(tf.expand_dims(batch_pos[0],0), (shift_h[i], shift_w[i]), axis=(1,2))  
          mask2.append(mask2_)
          img_shift.append(img_shift_)
          
        mask2 = tf.concat(mask2, axis = 0)
        img_shift = tf.concat(img_shift, axis = 0)

        input = batch_pos*(1.-mask2) + img_shift*mask2
       
        with tf.GradientTape() as tape:
            output= model(input)  
            loss = tf.reduce_mean(keras.losses.binary_crossentropy(mask2*( 1. - mask1), output*( 1. - mask1), from_logits=False))
            #loss =  tf.reduce_mean(tf.abs(output - mask2)*( 1. - mask1))

        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        output_mask = output > 0.5
        output_mask = tf.cast(output_mask, tf.float32)
        viz_img = [ input, tf.concat([mask2, mask2, mask2], axis = 3), tf.concat([output_mask, output_mask, output_mask], axis = 3)]
        viz_img_concat = tf.concat(viz_img, axis=2)  

        # a work around here / since there is bug in tf image summary until tf 2.3
#        if step % FLAGS.summary_iters == 0:
#            with tf.device("cpu:0"):
#                with writer.as_default():
#                    tf.summary.image('input_output', viz_img_concat, step=step, max_outputs=5)
#                    tf.summary.scalar('loss', loss, step=step)
        return loss

    # start training
    for step, batch_data in enumerate(full_ds):
        shift_h = np.random.randint(FLAGS.img_shapes[0], size = FLAGS.batch_size)
        shift_w = np.random.randint(FLAGS.img_shapes[1], size = FLAGS.batch_size)
        step = tf.convert_to_tensor(step, dtype=tf.int64)
        shift_h = tf.convert_to_tensor(shift_h, dtype=tf.int64)
        shift_w = tf.convert_to_tensor(shift_w, dtype=tf.int64)
        losses = training_step(batch_data, step, shift_h, shift_w)

        if step % FLAGS.print_iters == 0:
            print("Step:", step, "Loss", losses)

#        if step % FLAGS.summary_iters == 0:
#            writer.flush()

        if step % FLAGS.model_iters == 0:
            model.save_weights("%s/checkpoint_%d"%(FLAGS.log_dir, step.numpy()))

        if step >= FLAGS.max_iters:
            break
            
    print('finished!')

