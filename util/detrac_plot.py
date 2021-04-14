import numpy as np
import cv2
import PIL
from PIL import Image
from torch.utils import data
import matplotlib.pyplot as plt


def pil_to_cv(pil_im):
    """ convert PIL image to cv2 image"""
    open_cv_image = np.array(pil_im) 
    # Convert RGB to BGR 
    return open_cv_image[:, :, ::-1] 


def plot_text(im,offset,cls,idnum,class_colors,class_dict):
    """ Plots filled text box on original image, 
        utility function for plot_bboxes_2
        im - cv2 image
        offset - to upper left corner of bbox above which text is to be plotted
        cls - string
        class_colors - list of 3 tuples of ints in range (0,255)
        class_dict - dictionary that converts class strings to ints and vice versa
    """
    
    text = "{}: {}".format(idnum,class_dict[cls])
    
    font_scale = 1.0
    font = cv2.FONT_HERSHEY_PLAIN
    
    # set the rectangle background to white
    rectangle_bgr = class_colors[cls]
    
    # get the width and height of the text box
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
    
    # set the text start position
    text_offset_x = int(offset[0])
    text_offset_y = int(offset[1])
    # make the coords of the box with a small padding of two pixels
    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width - 2, text_offset_y - text_height - 2))
    cv2.rectangle(im, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
    cv2.putText(im, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0, 0, 0), thickness=1)


def plot_bboxes_2d(im,label,ignored_regions = []):
    """ Plots rectangular bboxes on image and returns image
    im - cv2 or PIL style image (function converts to cv2 style image)
    label - for one frame, in the form output by parse_label_file 
    bbox_im -  cv2 im with bboxes and labels plotted
    """
    
    # check type and convert PIL im to cv2 im if necessary
    assert type(im) in [np.ndarray, 
                       PIL.PngImagePlugin.PngImageFile,
                       PIL.JpegImagePlugin.JpegImageFile], "Invalid image format"
    if type(im) != np.ndarray:
        im = pil_to_cv(im)
    cv_im = im.copy() 
    
    class_colors = [
            (255,150,0),
            (255,100,0),
            (255,50,0),
            (0,255,150),
            (0,255,100),
            (0,255,50),
            (0,100,255),
            (0,50,255),
            (200,200,200) #ignored regions
            ]
    
    for det in label:
        bbox = det['bbox'].astype(int)
        cls = det['class']
        idnum = det['id']
        
        cv2.rectangle(cv_im,(bbox[0],bbox[1]),(bbox[2],bbox[3]), class_colors[cls], 1)
        plot_text(cv_im,(bbox[0],bbox[1]),cls,idnum,class_colors,class_dict)
        
    for region in ignored_regions:
        bbox = region.astype(int)
        cv2.rectangle(cv_im,(bbox[0],bbox[1]),(bbox[2],bbox[3]), class_colors[-1], 1)
    return cv_im

# for conversion of UA Detrac class labels
class_dict = {
    'Sedan':0,
    'Hatchback':1,
    'Suv':2,
    'Van':3,
    'Police':4,
    'Taxi':5,
    'Bus':6,
    'Truck-Box-Large':7,
    'MiniVan':8,
    'Truck-Box-Med':9,
    'Truck-Util':10,
    'Truck-Pickup':11,
    'Truck-Flatbed':12,
    
    0:'Sedan',
    1:'Hatchback',
    2:'Suv',
    3:'Van',
    4:'Police',
    5:'Taxi',
    6:'Bus',
    7:'Truck-Box-Large',
    8:'MiniVan',
    9:'Truck-Box-Med',
    10:'Truck-Util',
    11:'Truck-Pickup',
    12:'Truck-Flatbed'
    }
