import tensorflow as tf
import glob

def decode_img(img, img_height, img_width, channels=3):
  img = tf.image.decode_jpeg(img, channels=channels)
  return tf.image.resize(img, [img_height, img_width]) 

def process_img(file_path, img_height, img_width):
  img = tf.io.read_file(file_path)
  img = decode_img(img, img_height, img_width)
  img = img / 127.5 - 1
  return img 

def process_mask(file_path, img_height, img_width):
  img = tf.io.read_file(file_path)
  img = decode_img(img, img_height, img_width, channels=1)
  return img / 255.0 


def build_dataset_video(image_folder, mask_folder, mask_folder_a, batch_size, epochs, img_height, img_width):
    # convert to tensor
    img_height = tf.constant(img_height, dtype=tf.int64)
    img_width = tf.constant(img_width, dtype=tf.int64)

    # image dataset
    image_count = len(glob.glob(image_folder+"/*"))
    image_ds = tf.data.Dataset.list_files(str(image_folder+'/*'), shuffle=False)
    image_ds = image_ds.map(lambda file_path: process_img(file_path, img_height, img_width), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    image_ds.cache()
    image_ds = image_ds.repeat(epochs)

    # mask dataset
    mask_ds = tf.data.Dataset.list_files(str(mask_folder+'/*'), shuffle=False)
    mask_ds = mask_ds.map(lambda file_path: process_mask(file_path, img_height, img_width), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    mask_ds.cache()
    mask_ds = mask_ds.repeat(epochs)

    # mask dataset for augmentation
    mask_ds_a = tf.data.Dataset.list_files(str(mask_folder_a+'/*'), shuffle=False)
    mask_ds_a = mask_ds_a.map(lambda file_path: process_mask(file_path, img_height, img_width), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    mask_ds_a.cache()
    mask_ds_a = mask_ds_a.repeat(epochs)
    mask_ds_a = mask_ds_a.shuffle(image_count, reshuffle_each_iteration=True)

    # make full dataset
    full_ds = tf.data.Dataset.zip((image_ds, mask_ds, mask_ds_a))

    # configuration
    full_ds = full_ds.batch(batch_size)
    full_ds = full_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return full_ds 


def build_dataset_seg(image_folder, mask_folder,  batch_size, epochs, img_height, img_width):
    # convert to tensor
    img_height = tf.constant(img_height, dtype=tf.int64)
    img_width = tf.constant(img_width, dtype=tf.int64)

    # image dataset
    image_count = len(glob.glob(image_folder+"/*"))
    image_ds = tf.data.Dataset.list_files(str(image_folder+'/00000.jpg'), shuffle=False)
    image_ds = image_ds.map(lambda file_path: process_img(file_path, img_height, img_width), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    image_ds.cache()
    image_ds = image_ds.repeat(epochs)

    # mask dataset
    mask_ds = tf.data.Dataset.list_files(str(mask_folder+'/00000.png'), shuffle=False)
    mask_ds = mask_ds.map(lambda file_path: process_mask(file_path, img_height, img_width), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    mask_ds.cache()
    mask_ds = mask_ds.repeat(epochs)

    # make full dataset
    full_ds = tf.data.Dataset.zip((image_ds, mask_ds))

    # configuration
    full_ds = full_ds.batch(batch_size)
    full_ds = full_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return full_ds 
    

def build_dataset_up(image_folder, mask_folder, coarse_result_folder, mask_folder_a, batch_size, 
                     epochs, img_height, img_width, crop_height, crop_width, testing=False):
    # Convert to tensor
    img_height = tf.constant(img_height, dtype=tf.int64)
    img_width = tf.constant(img_width, dtype=tf.int64)
    crop_height = tf.constant(crop_height, dtype=tf.int64)
    crop_width = tf.constant(crop_width, dtype=tf.int64)

    # image dataset
    image_ds = tf.data.Dataset.list_files(str(image_folder+'/*'), shuffle=False)
    image_ds = image_ds.map(lambda file_path: process_img(file_path, img_height, img_width), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    image_ds.cache()
    image_ds = image_ds.repeat(epochs)

    # mask dataset
    mask_ds = tf.data.Dataset.list_files(str(mask_folder+'/*'), shuffle=False)
    mask_ds = mask_ds.map(lambda file_path: process_mask(file_path, img_height, img_width), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    mask_ds.cache()
    mask_ds = mask_ds.repeat(epochs)

    # mask dataset for augmentation
    mask_ds_a = tf.data.Dataset.list_files(str(mask_folder_a+'/*'), shuffle=False)
    mask_ds_a = mask_ds_a.map(lambda file_path: process_mask(file_path, crop_height, crop_width), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    mask_ds_a.cache()
    mask_ds_a = mask_ds_a.repeat(epochs)
    mask_count = len(glob.glob(mask_folder_a+"/*"))
    mask_ds_a = mask_ds_a.shuffle(mask_count, reshuffle_each_iteration=True)

    # coarse dataset
    image_c_ds = tf.data.Dataset.list_files(str(coarse_result_folder+'/*'), shuffle=False)
    image_c_ds = image_c_ds.map(lambda file_path: process_img(file_path, img_height, img_width), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    image_c_ds.cache()
    image_c_ds = image_c_ds.repeat(epochs)

    # make full dataset
    full_ds = tf.data.Dataset.zip((image_ds, mask_ds, mask_ds_a, image_c_ds))

    # configuration
    full_ds = full_ds.batch(batch_size)
    full_ds = full_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return full_ds
