"""
Derek Gloudemans - August 4, 2020
Adapted from https://github.com/yhenon/pytorch-retinanet - train.py

This file provides a dataset class for working with the UA-detrac tracking dataset.
Provides:
    - plotting of 2D bounding boxes
    - training/testing loader mode (random images from across all tracks) using __getitem__()
    - track mode - returns a single image, in order, using __next__()
"""

import os,sys
import numpy as np
import random 
import math
random.seed = 0

import cv2
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import functional as F

import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

try:
    sys.path.insert(0,os.getcwd)
    from detrac.detrac_plot import pil_to_cv, plot_bboxes_2d
except:
    from detrac_plot import pil_to_cv, plot_bboxes_2d,plot_text



class LocMulti_Dataset(data.Dataset):
    """
    Creates an object for referencing the UA-Detrac 2D object tracking dataset
    and returning single object images for localization. Note that this dataset
    does not automatically separate training and validation data, so you'll 
    need to partition data manually by separate directories
    """
    
    def __init__(self, image_dir, label_dir, cs =224):
        """ initializes object
        image dir - (string) - a directory containing a subdirectory for each track sequence
        label dir - (string) - a directory containing a label file per sequence
        """
        self.classes = 13
        self.class_dict = {
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
            "None":13,
            
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
            12:'Truck-Flatbed',
            13:"None"
            }
        
        self.cs = cs
        self.im_tf = transforms.Compose([
                transforms.RandomApply([
                    transforms.ColorJitter(brightness = 0.6,contrast = 0.6,saturation = 0.5)
                        ]),
                transforms.ToTensor(),
                # transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=(0.485,0.456,0.406)),
                # transforms.RandomErasing(p=0.2, scale=(0.02, 0.07), ratio=(0.3, 3.3), value=(0.485,0.456,0.406)),
                # transforms.RandomErasing(p=0.2, scale=(0.02, 0.05), ratio=(0.3, 3.3), value=(0.485,0.456,0.406)),
                # transforms.RandomErasing(p=0.1, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=(0.485,0.456,0.406)),
                # transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=(0.485,0.456,0.406)),

                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
                ])

        # for denormalizing
        self.denorm = transforms.Normalize(mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                           std = [1/0.229, 1/0.224, 1/0.225])
        
        # stores files for each image
        dir_list = next(os.walk(image_dir))[1]
        track_list = [os.path.join(image_dir,item) for item in dir_list]
        track_list.sort()
        
        # parse labels and store in dict keyed by track name
        label_list = {}
        for item in os.listdir(label_dir):
            name = item.split("_v3.xml")[0]
            out =  self.parse_labels(os.path.join(label_dir,item))
            label_list[name] = out
        
        # for storing data
        self.all_data = []
        
        # parse and store all labels and image names in a list such that
        # all_data[i] returns dict with image name, label and other stats
        # track_offsets[i] retuns index of first frame of track[i[]]
        for i in range(len(track_list)):

            images = [os.path.join(track_list[i],frame) for frame in os.listdir(track_list[i])]
            images.sort() 
            labels,metadata = label_list[track_list[i].split("/")[-1]]
            
            # each iteration of the loop gets one image
            print (len(images),len(labels))
            for j in range(len(images)):
                try:
                    image = images[j]
                    label = labels[j]
                                        

                    self.all_data.append((image,label,metadata))

                except:
                    # this occurs because Detrac was dumbly labeled and they didn't include empty annotations for frames without objects
                    # parse_labels corrects this mostly, except for trailing frames
                    # so we just pass because there are no objects or labels anyway
                    self.all_data.append((images[j],[],metadata))
                    #print("Error: tried to load label {} for track {} but it doesnt exist. Labels is length {}".format(j,track_list[i],len(labels))) 
                
                    
        # in case it is later important which files are which
        self.track_list = track_list
        self.label_list = label_list
        


    def __len__(self):
        """ returns total number of frames in all tracks"""
        return len (self.all_data)
       # return self.total_num_frames
    
    def __getitem__(self,index):
        """ returns item indexed from all frames in all tracks from training
        or testing indices depending on mode
        """
    
        # load image and get label        
        cur = self.all_data[index]
        im = Image.open(cur[0])
        label = cur[1]
        ignored = cur[2]['ignored_regions']
        if len(ignored) > 0:
            ignored = torch.from_numpy(np.stack(ignored))
        
        
        if len(label) > 0:
            bboxes = []
            classes = []
            for item in label:
                bboxes.append(torch.from_numpy(item["bbox"]))
                classes.append(torch.tensor(item["class_num"]))
            bboxes = torch.stack(bboxes)
            classes = torch.stack(classes).double().unsqueeze(1)
            
            
            y = torch.cat((bboxes,classes),dim= 1)
            
        else:
            y = torch.zeros([1,5])
            y[0,4] = -1
            
        # randomly flip
        FLIP = np.random.rand()
        if FLIP > 0.5:
            im= F.hflip(im)
            # reverse coords and also switch xmin and xmax
            new_y = torch.clone(y)
            new_y[:,0] = im.size[0] - y[:,2]
            new_y[:,2] = im.size[0] - y[:,0]
            y = new_y
            
            if len(ignored) > 0:
                new_ig = torch.clone(ignored)
                new_ig[:,0] = im.size[0] - ignored[:,2]
                new_ig[:,2] = im.size[0] - ignored[:,0]
                ignored = new_ig
        
        # # convert image and label to tensors
        im_t = transforms.ToTensor()(im)
        
        # # mask out ignored regions with random pixels
        for region in ignored:
            r =  torch.normal(0.485,0.229,[int(region[3])-int(region[1]),int(region[2])-int(region[0])])
            g =  torch.normal(0.456,0.224,[int(region[3])-int(region[1]),int(region[2])-int(region[0])])
            b =  torch.normal(0.406,0.225,[int(region[3])-int(region[1]),int(region[2])-int(region[0])])
            rgb = torch.stack([r,g,b])
            im_t[:,int(region[1]):int(region[3]),int(region[0]):int(region[2])] = rgb 
          
        im = transforms.ToPILImage()(im_t)
        

        
        # use one object to define center
        if torch.sum(y) != 0:
            idx = np.random.randint(len(y))
            box = y[idx]
            centx = (box[0] + box[2])/2.0
            centy = (box[1] + box[3])/2.0
            noise = np.random.normal(0,20,size = 2)
            centx += noise[0]
            centy += noise[1]
            
            size = max(box[3]-box[1],box[2] - box[0])
            size_noise = max( -(size*2/4) , np.random.normal(size/2,size/4))
            size += size_noise
            
            if size < 50:
                size = 50
        else:
            size = max(50,np.random.normal(100,25))
            centx = np.random.randint(100,500)
            centy = np.random.randint(100,500)
        try:
            minx = int(centx - size/2)
            miny = int(centy - size/2)
            maxx = int(centx + size/2)
            maxy = int(centy + size/2)
        
        except TypeError:
            print(centx,centy,size)
        
        im_crop = F.crop(im,miny,minx,maxy-miny,maxx-minx)
        del im 
        
        if im_crop.size[0] == 0 or im_crop.size[1] == 0:
            print("Oh no! {} {} {}".format(centx,centy,size))
            raise Exception
            
        # shift labels if there is at least one object
   
        if torch.sum(y) != 0:
            y[:,0] = y[:,0] - minx
            y[:,1] = y[:,1] - miny
            y[:,2] = y[:,2] - minx
            y[:,3] = y[:,3] - miny

        crop_size = im_crop.size
        im_crop = F.resize(im_crop, (self.cs,self.cs))

        y[:,0] = y[:,0] * self.cs/crop_size[0]
        y[:,2] = y[:,2] * self.cs/crop_size[0]
        y[:,1] = y[:,1] * self.cs/crop_size[1]
        y[:,3] = y[:,3] * self.cs/crop_size[1]
        
        # remove all labels that aren't in crop
        if torch.sum(y) != 0:
            keepers = []
            for i,item in enumerate(y):
                if item[0] < self.cs-15 and item[2] > 0+15 and item[1] < self.cs-15 and item[3] > 0+15:
                    keepers.append(i)
            y = y[keepers]
        if len(y) == 0:
            y = torch.zeros([1,5])
            y[0,4] = -1
        
        im_t = self.im_tf(im_crop)

        
        return im_t, y,ignored
    
    def parse_labels(self,label_file):
        """
        Returns a set of metadata (1 per track) and a list of labels (1 item per
        frame, where an item is a list of dictionaries (one dictionary per object
        with fields id, class, truncation, orientation, and bbox
        """
        
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
            "None":13,
            
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
            12:'Truck-Flatbed',
            13:"None"
            }
        
        
        tree = ET.parse(label_file)
        root = tree.getroot()
        
        # get sequence attributes
        seq_name = root.attrib['name']
        
        # get list of all frame elements
        #frames = root.getchildren()
        frames = list(root)
        # first child is sequence attributes
        seq_attrs = frames[0].attrib
        
        # second child is ignored regions
        ignored_regions = []
        for region in frames[1]:
            coords = region.attrib
            box = np.array([float(coords['left']),
                            float(coords['top']),
                            float(coords['left']) + float(coords['width']),
                            float(coords['top'])  + float(coords['height'])])
            ignored_regions.append(box)
        frames = frames[2:]
        
        # rest are bboxes
        all_boxes = []
        frame_counter = 1
        for frame in frames:
            while frame_counter < int(frame.attrib['num']):
                # this means that there were some frames with no detections
                all_boxes.append([])
                frame_counter += 1
            
            frame_counter += 1
            frame_boxes = []
            #boxids = frame.getchildren()[0].getchildren()
            boxids = list(list(frame)[0])
            for boxid in boxids:
                #data = boxid.getchildren()
                data = list(boxid)
                coords = data[0].attrib
                stats = data[1].attrib
                bbox = np.array([float(coords['left']),
                                float(coords['top']),
                                float(coords['left']) + float(coords['width']),
                                float(coords['top'])  + float(coords['height'])])
                det_dict = {
                        'id':int(boxid.attrib['id']),
                        'class':stats['vehicle_type'],
                        'class_num':class_dict[stats['vehicle_type']],
                        'color':stats['color'],
                        'orientation':float(stats['orientation']),
                        'truncation':float(stats['truncation_ratio']),
                        'bbox':bbox,
                        'frame':int(frame.attrib['num'])
                        }
                
                frame_boxes.append(det_dict)
            all_boxes.append(frame_boxes)
        
        sequence_metadata = {
                'sequence':seq_name,
                'seq_attributes':seq_attrs,
                'ignored_regions':ignored_regions
                }
        return all_boxes, sequence_metadata
    
    
    def num_classes(self):
        return self.classes
    
    def label_to_name(self,num):
        return class_dict[num]
        
    def load_annotations(self,idx):
        """
        Loads labels in format for mAP evaluation 
        list of arrays, one [n_detections x 4] array per class
        """
        annotation = [[] for i in range(self.classes)]
        
        label = self.all_data[idx][1]
        
        for obj in label:
            cls = int(obj['class_num'])
            annotation[cls].append(obj['bbox'].astype(float))
        
        for idx in range(len(annotation)):
            if len(annotation[idx]) > 0:
                annotation[idx] = np.stack(annotation[idx])
            else:
                annotation[idx] = np.empty(0)
        return annotation
    
    def show(self,index):
        """ plots all frames in track_idx as video
            SHOW_LABELS - if True, labels are plotted on sequence
            track_idx - int    
        """
        mean = np.array([0.485, 0.456, 0.406])
        stddev = np.array([0.229, 0.224, 0.225])
        
        im,label,_ = self[index]
        
        im = self.denorm(im)
        cv_im = np.array(im) 
        cv_im = np.clip(cv_im, 0, 1)
        
        # Convert RGB to BGR 
        cv_im = cv_im[::-1, :, :]         
        
        cv_im = np.moveaxis(cv_im,[0,1,2],[2,0,1])

        cv_im = cv_im.copy()

        class_colors = [
            (255,150,0),
            (255,100,0),
            (255,50,0),
            (0,255,150),
            (0,255,100),
            (0,255,50),
            (0,100,255),
            (0,50,255),
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
        
        if torch.sum(label) != 0:
            for bbox in label:
                bbox = bbox.int().data.numpy()
                cv2.rectangle(cv_im,(bbox[0],bbox[1]),(bbox[2],bbox[3]), class_colors[bbox[4]], 1)
                plot_text(cv_im,(bbox[0],bbox[1]),bbox[4],0,class_colors,class_dict)
        
        
        # for region in metadata["ignored_regions"]:
        #     bbox = region.astype(int)
        #     cv2.rectangle(cv_im,(bbox[0],bbox[1]),(bbox[2],bbox[3]), class_colors[-1], 1)
       
    
        cv2.imshow("Frame",cv_im)
        cv2.waitKey(0) 
        cv2.destroyAllWindows()

def collate(inputs):
    """
    Recieves list of tuples and returns a tensor for each item in tuple, except metadata
    which is returned as a single list
    """
    im = [] # in this dataset, always [3 x W x H]
    label = [] # variable length
    metadata = []
    max_labels = 0
    
    for batch_item in inputs:
        im.append(batch_item[0])
        label.append(batch_item[1])
        metadata.append(batch_item[2])
        
        # keep track of image with largest number of annotations
        if len(batch_item[1]) > max_labels:
            max_labels = len(batch_item[1])
        
    # collate images        
    ims = torch.stack(im)
    
    # collate labels
    labels = torch.zeros([len(label),max_labels,5]) - 1
    for idx in range(len(label)):
        num_objs = len(label[idx])
        
        labels[idx,:num_objs,:] = label[idx]
        
    return ims,labels,metadata   
        
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
            "None":13,
            
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
            12:'Truck-Flatbed',
            13:"None"
            }

def collate(inputs):
    """
    Recieves list of tuples and returns a tensor for each item in tuple, except metadata
    which is returned as a single list
    """
    im = [] # in this dataset, always [3 x W x H]
    label = [] # variable length
    metadata = []
    max_labels = 0
    
    for batch_item in inputs:
        im.append(batch_item[0])
        label.append(batch_item[1])
        metadata.append(batch_item[2])
        
        # keep track of image with largest number of annotations
        if len(batch_item[1]) > max_labels:
            max_labels = len(batch_item[1])
        
    # collate images        
    ims = torch.stack(im)
    
    # collate labels
    labels = torch.zeros([len(label),max_labels,5]) - 1
    for idx in range(len(label)):
        num_objs = len(label[idx])
        
        labels[idx,:num_objs,:] = label[idx]
        
    return ims,labels,metadata       
    
if __name__ == "__main__":
    #### Test script here
    try: 
        test
    except:
        
        image_dir = "/home/worklab/Desktop/detrac/DETRAC-train-data"
        label_dir = "/home/worklab/Desktop/detrac/DETRAC-Train-Annotations-XML-v3"
        test = LocMulti_Dataset(image_dir,label_dir)
    for i in range(10):
        idx = np.random.randint(0,len(test))
        test.show(idx)
        test.show(idx)
    
    cv2.destroyAllWindows()