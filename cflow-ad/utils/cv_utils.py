import cv2 as cv
import numpy as np
from utils import score_utils

def histogram_equalization(img):
    image_ycrcb = cv.cvtColor(img, cv.COLOR_RGB2YCR_CB)
    # apply histogram equalization on this channel
    y_channel = image_ycrcb[:, :, 0]
    cr_channel = image_ycrcb[:, :, 1]
    cb_channel = image_ycrcb[:, :, 2]
    # local histogram equalization
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(y_channel)
    equalized_image = cv.merge([equalized, cr_channel, cb_channel])
    equalized_image = cv.cvtColor(equalized_image, cv.COLOR_YCR_CB2RGB)
    return equalized_image


def get_otsu_threshold_output(img):
    otsu_threshold, otsu_image = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return otsu_threshold, otsu_image


def split_channel(img):
    img_b, img_g, img_r = cv.split(img)
    return img_b, img_g, img_r


def get_gray_scale(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

def is_color_image(img):
    return (len(img.shape)==3) and (3 in img.shape)

def apply_clahe(img):
    clahe =cv.createCLAHE(clipLimit=5)
    return clahe.apply(img)
def perform_morphology(img):
    res = img
    struct_ele = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    # Morphology transform works only on gray scale
    if is_color_image(img):
        res = get_gray_scale(res)
    # Perform morpho
    for i in range(2):
        res = cv.dilate(res, struct_ele, iterations=2)
        res = cv.erode(res, struct_ele, iterations=2)
    return res

def denoise_color_image(image):
    denoised_image = cv.fastNlMeansDenoisingColored(image, None, 4, 10, 7, 21)
    return denoised_image


def handle_processing_step(c, img, process_type, process_value):
    res = img
    if process_type == 'denoise':
        if process_value == 'true':
            res = histogram_equalization(res)
    if process_type == 'hist_eq':
        if process_value == 'true':
            res = histogram_equalization(res)
    if process_type == 'clahe':
        if process_value == 'true':
            res = apply_clahe(res)
    if process_type == 'morph':
        c_b, c_g, c_r = split_channel(res)
        if process_value == 'red':
            res = perform_morphology(c_r)
        elif process_value == 'green':
            res = perform_morphology(c_g)
        elif process_value == 'blue':
            res = perform_morphology(c_b)
    if process_type == 'otsu':
        if process_value == 'true':
            res = get_otsu_threshold_output(res)
    return res

def handle_image_processing(c, img):
    # Image processing pipeline based on the config object
    B,C,H,W = img.shape
    res_list = np.array([]) 
    img = score_utils.t2np(img)
    processing_stage_string = c.image_processing
    if not processing_stage_string:
        return img
    stages = processing_stage_string.split('|')
    # Batch processing
    for stage in stages:
        p_type, p_value = stage.split(':')
        for i in range(B):
            im = img[i]
            im = im.reshape(H,W,C)
            im_res =handle_processing_step(c, im, p_type, p_value)
            if len(im_res.shape) == 2:
                im_res = cv.cvtColor(im_res, cv.COLOR_GRAY2RGB)
            else:
                im_res = im_res.reshape(C,H,W)
            res_list = np.append(res_list, im_res)
    res_list = score_utils.np2t(res_list)
    res_list = res_list.reshape(B,C,H,W)
    return res_list
