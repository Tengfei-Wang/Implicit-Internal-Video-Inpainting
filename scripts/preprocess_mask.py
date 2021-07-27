import cv2
import numpy as np
import pathlib
import glob
import argparse

def read_mask(path):
    mask = cv2.imread(path, 0)
    kernel = np.ones((12,12),np.uint8)
    mask = cv2.dilate(mask,kernel,iterations = 1)
    mask[mask >0]=1.
    return mask*255 

def generate_fixed_mask(height, width, height_ratio, width_ratio):
    mask = np.zeros((height, width))
    h_start = int((1-height_ratio) / 2.0 * height)
    h_end = int(h_start + height_ratio * height) 
    w_start = int((1-width_ratio) / 2.0 * width)
    w_end = int(w_start + width_ratio * width) 
    return mask*255


#annotation_path = "inputs/annotations/bmx-trees"
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_path', required=True, help='directory path for Annotation or Mask')
    parser.add_argument('--generate_fix_mask', action='store_true', help='generate fixed rectangular mask')
    args = parser.parse_args()

    fixed = args.generate_fix_mask 
    annotation_path = args.annotation_path
    annotation_list = sorted(glob.glob("%s/*"%annotation_path))
    
    # create a new mask folder
    if fixed:
        mask_folder = "inputs/masks/%s_fixed"%(annotation_path.split("/")[-1])
    else:
        mask_folder = "inputs/masks/%s"%(annotation_path.split("/")[-1])
    
    img_folder = "inputs/videos/%s"%(annotation_path.split("/")[-1])
    pathlib.Path(mask_folder).mkdir(parents=True, exist_ok=True)
    
    # write black and white masks
    for annt_path in annotation_list:
        img_path = "%s/%s.jpg" % (img_folder, annt_path.split("/")[-1].split(".")[0])
        mask_path = "%s/%s" % (mask_folder, annt_path.split("/")[-1])
        print(mask_path)
        if fixed:
            mask = generate_fixed_mask(480, 952, 0.25, 0.25)
        else:
            mask = read_mask(annt_path)
        cv2.imwrite(mask_path, mask)    
    
if __name__ == "__main__":
    main()
