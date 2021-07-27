import os
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
import numpy as np
import pathlib

import data.dataloader as dl 
from config.config import Config
from model.inpaint_model import UnetModel

if __name__ == "__main__":
    FLAGS = Config('config/test_seg.yml')
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU_ID 
    
    test_dir = FLAGS.test_dir
    pathlib.Path(test_dir).mkdir(parents=True, exist_ok=True)

    model = UnetModel()
    model.load_weights(FLAGS.model_restore)
    test_ds = dl.build_dataset_video(FLAGS.dir_video, FLAGS.dir_mask,  FLAGS.dir_mask, 1, 1, FLAGS.img_shapes[0], FLAGS.img_shapes[1])
 
    @tf.function
    def testing_step(batch_data, filepath):
        batch_pos = batch_data[0]
        output = model(batch_pos)
        
        mask = output[0]> 0.5   
        mask = tf.cast(mask, tf.uint8)*255
        out_image = tf.image.encode_png(mask )
        tf.io.write_file(filepath, out_image)


    for step, batch_data in enumerate(test_ds):
        print(step)
        filepath = "%s/%04d.png" % (test_dir, step)
        filepath = tf.convert_to_tensor(filepath)
        testing_step(batch_data, filepath)
