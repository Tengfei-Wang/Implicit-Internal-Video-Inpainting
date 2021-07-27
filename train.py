import os
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa

import data.dataloader as dl 
from config.config import Config
from model.inpaint_model import RefineModel, BaseModel, StackModel
from loss import ambiguity_loss, stable_loss

if __name__ == "__main__":
    # read config 
    FLAGS = Config('config/train.yml')
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU_ID 
    
    #mirrored_strategy = tf.distribute.MirroredStrategy()
    #with mirrored_strategy.scope():
    
    # define the model
    if FLAGS.coarse_only:
        model = BaseModel()
    else:
        if FLAGS.use_refine:
            model = RefineModel()
        else: 
            model = StackModel()

    if not FLAGS.model_restore=="":
        model.load_weights(FLAGS.model_restore)
        
        
    # define the optimizer
    optimizer = keras.optimizers.Adam(learning_rate=FLAGS.lr, beta_1=0.9, beta_2=0.999)

    # define the dataloader 
    full_ds = dl.build_dataset_video(FLAGS.dir_video, FLAGS.dir_mask, FLAGS.dir_mask, 
                                FLAGS.batch_size, FLAGS.max_epochs, FLAGS.img_shapes[0], FLAGS.img_shapes[1])
    #dist_full_ds = mirrored_strategy.experimental_distribute_dataset(full_ds)

    #summary writer
    writer = tf.summary.create_file_writer(FLAGS.log_dir)

    # define the training steps and loss
    def training_step(batch_data, step):       
        batch_pos = batch_data[0]
        mask1 = batch_data[2] 
        mask2 = batch_data[1] 
        shift_h = tf.random.uniform(shape=[], maxval=mask1.shape[1], dtype=tf.int64)
        shift_w = tf.random.uniform(shape=[], maxval=mask1.shape[2], dtype=tf.int64)
        mask1 = tf.roll(mask1, (shift_h, shift_w), axis=(1,2))  
        mask = tf.cast(
            tf.logical_or(
                tf.cast(mask1, tf.bool),
                tf.cast(mask2, tf.bool),
            ),
            tf.float32
        )
        batch_incomplete = batch_pos*(1.-mask)
        xin = batch_incomplete
        x = tf.concat([xin, mask], axis=3)

        # stabilization loss
        if FLAGS.stabilization_loss:
            T = stable_loss.get_transform(FLAGS)

            # Perform transformation
            T_batch_pos = tfa.image.transform(batch_pos, T, interpolation = 'BILINEAR')
            Tmask = tfa.image.transform(mask, T, interpolation = 'NEAREST')
            Tmask2 = tfa.image.transform(mask2, T, interpolation = 'NEAREST')
            Tmask_n = tf.cast(
                tf.logical_or(
                    tf.cast(mask2, tf.bool),
                    tf.cast(Tmask2, tf.bool),),
                tf.float32)
            
            Tx = tf.concat([T_batch_pos*(1-Tmask), Tmask], axis=3)
 

        with tf.GradientTape(persistent=True) as tape:
            if not FLAGS.coarse_only:
                x1, x2 = model(x, mask) 
                loss = FLAGS.l1_loss_alpha * tf.reduce_mean(tf.abs(batch_pos - x1)*(1-mask2))
                loss += FLAGS.l1_loss_alpha * tf.reduce_mean(tf.abs(batch_pos - x2)*(1-mask2))
                
                if FLAGS.stabilization_loss:
                    Tx1, Tx2 = model(Tx, Tmask)
                    loss += FLAGS.stabilization_loss_alpha * tf.reduce_mean(tf.abs((Tx2 - x2)-(T_batch_pos-batch_pos)) * (1-Tmask_n))
                    loss += FLAGS.stabilization_loss_alpha * tf.reduce_mean(tf.abs((Tx1 - x1)-(T_batch_pos-batch_pos)) * (1-Tmask_n))
                    
                if FLAGS.ambiguity_loss:
                    #loss += FLAGS.ambiguity_loss_alpha*ambiguity_loss.perceptual_loss((1-mask2)*x2, (1-mask2)*batch_pos)
                    loss +=  FLAGS.ambiguity_loss_alpha*ambiguity_loss.contextual_loss((1-mask2[::-1,:,:,:])*x2, (1-mask2[::-1,:,:,:])*batch_pos[::-1,:,:,:])                     

            else:
                x1 = model(x) 
                loss = FLAGS.l1_loss_alpha * tf.reduce_mean(tf.abs(batch_pos - x1)*(1-mask2))
                x2 = x1
        

        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))


        # add summary
        batch_complete = x2*mask + batch_incomplete*(1.-mask)
        viz_img = [batch_pos, batch_incomplete, x1, x2, batch_complete]
        viz_img_concat = (tf.concat(viz_img, axis=2) + 1) / 2.0  

        # a work around here / since there is a bug in tf image summary until tf 2.3
        if step % FLAGS.summary_iters == 0:
            with tf.device("cpu:0"):
                with writer.as_default():
                    tf.summary.image('input_input_x1_x2_output', viz_img_concat, step=step, max_outputs=6)
                    tf.summary.scalar('loss', loss, step=step)
        
        return loss
    

#    @tf.function
#    def distributed_train_step(dataset_inputs, step):
#        per_replica_losses = mirrored_strategy.experimental_run_v2(training_step, args=(dataset_inputs, step,))
#        return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    # start training
    for step, batch_data in enumerate(full_ds):
        step = tf.convert_to_tensor(step, dtype=tf.int64)
        losses = training_step(batch_data, step)

        if step % FLAGS.print_iters == 0:
            print("Step:", step, "Loss", losses)

        if step % FLAGS.summary_iters == 0:
            writer.flush()

        if step % FLAGS.model_iters == 0:
            model.save_weights("%s/checkpoint_%d"%(FLAGS.log_dir, step.numpy()))

        if step >= FLAGS.max_iters:
            break
    print('finished!')