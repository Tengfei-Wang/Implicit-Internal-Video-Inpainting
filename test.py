import os
import tensorflow as tf
import pathlib

import data.dataloader as dl 
from config.config import Config
from model.inpaint_model import RefineModel, BaseModel, StackModel



if __name__ == "__main__":
    FLAGS = Config('config/test.yml')
    os.environ["CUDA_VISIBLE_DEVICES"]= FLAGS.GPU_ID
    test_dir = FLAGS.test_dir  
    pathlib.Path(test_dir).mkdir(parents=True, exist_ok=True)

    if FLAGS.coarse_only:
        model = BaseModel()
    else:
        if FLAGS.use_refine:
            model = RefineModel()
        else: 
            model = StackModel()

    model.load_weights(FLAGS.model_restore)
    test_ds = dl.build_dataset_video(FLAGS.dir_video, FLAGS.dir_mask, FLAGS.dir_mask, 1, 1, FLAGS.img_shapes[0], FLAGS.img_shapes[1])

    @tf.function
    def testing_step(batch_data):
        batch_pos = batch_data[0]
        mask = batch_data[1]
        mask = tf.cast(tf.cast(mask, tf.bool), tf.float32)
        batch_incomplete = batch_pos*(1.-mask)
        xin = batch_incomplete
 
        x = tf.concat([xin, mask], axis=3)
        if FLAGS.coarse_only:
            x2 = model(x, mask)
        else:
            x1, x2 = model(x, mask)
        batch_complete = x2*mask + batch_incomplete*(1.-mask)

        # write image
        batch_complete = (batch_complete + 1) / 2.0 * 255
        batch_complete = tf.cast(batch_complete[0], tf.uint8)
        out_image = tf.io.encode_jpeg(batch_complete, format='rgb')
        out_gt = tf.io.encode_jpeg(tf.cast((batch_pos[0] + 1) / 2.0 * 255, tf.uint8), format='rgb')
        return out_image, out_gt


    for step, batch_data in enumerate(test_ds):
        print(step)
        filepath = "%s/%04d.jpg" % (test_dir, step)

        out_image, out_gt = testing_step(batch_data)
        if (step == 0):
            print(model.summary())

        tf.io.write_file(filepath, out_image)