import skimage.io as io
import numpy as np
import cv2 

def resize_img(img, scale_factor):
    new_size = (np.round(np.array(img.shape[:2]) * scale_factor)).astype(int)
    new_img = cv2.resize(img, (new_size[1], new_size[0]))
    # This is scale factor of [height, width] i.e. [y, x]
    actual_factor = [new_size[0] / float(img.shape[0]),
                     new_size[1] / float(img.shape[1])]
    return new_img, actual_factor

def crop(img, bbox, bgval=0):
    '''
    Crops a region from the image corresponding to the bbox.
    If some regions specified go outside the image boundaries, the pixel values are set to bgval.
    Args:
        img: image to crop
        bbox: bounding box to crop
        bgval: default background for regions outside image        
    '''
    bbox = [int(round(c)) for c in bbox]
    bwidth = bbox[2] - bbox[0] + 1
    bheight = bbox[3] - bbox[1] + 1

    im_shape = np.shape(img)
    im_h, im_w = im_shape[0], im_shape[1]

    nc = 1 if len(im_shape) < 3 else im_shape[2]
    
    img_out = np.ones((bheight, bwidth, nc))*bgval
    x_min_src = max(0, bbox[0])
    x_max_src = min(im_w, bbox[2]+1)
    y_min_src = max(0, bbox[1])
    y_max_src = min(im_h, bbox[3]+1)
    
    x_min_trg = x_min_src - bbox[0]
    x_max_trg = x_max_src - x_min_src + x_min_trg
    y_min_trg = y_min_src - bbox[1]
    y_max_trg = y_max_src - y_min_src + y_min_trg
    img_out[y_min_trg:y_max_trg, x_min_trg:x_max_trg, :] = img[y_min_src:y_max_src, x_min_src:x_max_src, :]
    return img_out

def preprocess_image(img_path, img_size=256):
    img = io.imread(img_path) / 255.
    # Scale the max image size to be img_size
    scale_factor = float(img_size) / np.max(img.shape[:2])
    img, _ = resize_img(img, scale_factor)
    if len(img.shape) == 2:
        img = img.reshape(img.shape+(1,))
    # Crop img_size x img_size from the center
    center = np.round(np.array(img.shape[:2]) / 2).astype(int)
    # img center in (x, y)
    center = center[::-1]
    bbox = np.hstack([center - img_size / 2., center + img_size / 2.])

    img = crop(img, bbox, bgval=1.)

    # Transpose the image to 3xHxW
    img = np.transpose(img, (2, 0, 1))
    if img.shape[1] == 1+img_size:
        img = img[:, :img_size, :img_size]
    return img