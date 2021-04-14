#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 11:33:03 2020

@author: worklab
"""


import os
import numpy as np
import random 
import time
random.seed = 0

import cv2
from PIL import Image
import torch

from torchvision.transforms import functional as F
import torch.multiprocessing as mp


class FrameLoader():
    
    def __init__(self,track_directory,device,buffer_size = 9,downsample = 1,s=1,show = True):
        
        """
        Parameters
        ----------
        track_directory : str
            Path to frames 
        device : torch.device
            specifies where frames should be loaded to , memory or GPU
        det_step : int
            How often full detection is performed, necessary for image pre-processing
        init_frames : int
            Number of frames used to initialize Kalman filter every det_step frames
        cutoff : int, optional
            If not None, only this many frames are loaded into memory. The default is None.
    
        """
        try:
            files = []
            for item in [os.path.join(track_directory,im) for im in os.listdir(track_directory)]:
                files.append(item)
                files.sort()    
        
            
            self.files = files
            self.downsample = downsample
            self.s = s
            
            #manager = mp.Manager()
            
            #self.det_step = det_step
            self.init_frames = init_frames
            self.device = device
        
            # create shared queue
            #mp.set_start_method('spawn')
            ctx = mp.get_context('spawn')
            self.queue = ctx.Queue()
            #self.cache = ctx.Value(torch.Tensor)
            self.frame_idx = -1
            
            self.worker = ctx.Process(target=load_to_queue, args=(self.queue,files,device,buffer_size,self.downsample))
            self.worker.start()
            time.sleep(5)
        
        except: # file is a video
            sequence = track_directory
            
            self.sequence = sequence
            self.s = s
            
            manager = mp.Manager()
            
            #self.det_step = det_step
            self.device = device
        
            # create shared queue
            #mp.set_start_method('spawn')
            ctx = mp.get_context('spawn')
            self.queue = ctx.Queue()
            
            self.frame_idx = -1
            
            cap = cv2.VideoCapture(sequence)
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.len = length
            cap.release()
            time.sleep(1)
            
            self.worker = ctx.Process(target=load_to_queue_video, args=(self.queue,sequence,device,buffer_size,self.s,show))
            self.worker.start()
            time.sleep(5)
        
    def __len__(self):
        """
        Description
        -----------
        Returns number of frames in the track directory
        """
        try:
            return self.len
        except:   
            return len(self.files)
    
    def __next__(self):
        """
        Description
        -----------
        Returns next frame and associated data unless at end of track, in which
        case returns -1 for frame num and None for frame

        Returns
        -------
        frame_num : int
            Frame index in track
        frame : tuple of (tensor,tensor,tensor)
            image, image dimensions and original image

        """
        
        if self.frame_idx < len(self) -1:
        
            frame = self.queue.get(timeout = 5)
            self.frame_idx = frame[0]
            return frame
        
        else:
            self.worker.terminate()
            self.worker.join()
            return [-1,None,None,None]

def load_to_queue(image_queue,files,device,queue_size,downsample):
    """
    Description
    -----------
    Whenever necessary, loads images, moves them to GPU, and adds them to a shared
    multiprocessing queue with the goal of the queue always having a certain size.
    Process is to be called as a worker by FrameLoader object
    
    Parameters
    ----------
    image_queue : multiprocessing Queue
        shared queue in which preprocessed images are put.
    files : list of str
        each str is path to one file in track directory
    det_step : int
        specifies number of frames between dense detections 
    init_frames : int
        specifies number of dense detections before localization begins
    device : torch.device
        Specifies whether images should be put on CPU or GPU.
    queue_size : int, optional
        Goal size of queue, whenever actual size is less additional images will
        be processed and added. The default is 5.
    """
    
    frame_idx = 0    
    while frame_idx < len(files):
        
        if image_queue.qsize() < queue_size:
            
            # load next image
            with Image.open(files[frame_idx]) as im:
             
              # if frame_idx % det_step.value < init_frames:   
              #     # convert to CV2 style image
              #     open_cv_image = np.array(im) 
              #     im = open_cv_image.copy() 
              #     original_im = im[:,:,[2,1,0]].copy()
              #     # new stuff
              #     dim = (im.shape[1], im.shape[0])
              #     im = cv2.resize(im, (1920,1080))
              #     im = im.transpose((2,0,1)).copy()
              #     im = torch.from_numpy(im).float().div(255.0).unsqueeze(0)
              #     dim = torch.FloatTensor(dim).repeat(1,2)
              #     dim = dim.to(device,non_blocking = True)
              # else:
                  # keep as tensor
              original_im = np.array(im)[:,:,[2,1,0]].copy()
              im = F.resize(im,(int(im.size[1]//downsample),int(im.size[0]//downsample)))
              im = F.to_tensor(im)
              im = F.normalize(im,mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
              dim = None
                 
              # store preprocessed image, dimensions and original image
              im = im.to(device)
              frame = (frame_idx,im,dim,original_im)
             
              # append to queue
              image_queue.put(frame)
             
            frame_idx += 1
    
    # neverending loop, because if the process ends, the tensors originally
    # initialized in this function will be deleted, causing issues. Thus, this 
    # function runs until a call to self.next() returns -1, indicating end of track 
    # has been reached
    while True:  
           time.sleep(5)
        
def load_to_queue_video(image_queue,sequence,device,queue_size,s,show):
    
    cap = cv2.VideoCapture(sequence)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_idx = 0    
    while frame_idx < length:
        
        if image_queue.qsize() < queue_size:
            
            if frame_idx % s == 0: # don't skip
            
                # load next image from videocapture object
                ret,original_im = cap.read()
                if ret == False:
                    frame = (-1,None,None,None)
                    image_queue.put(frame)       
                    break
                else:
                    #original_im = cv2.resize(original_im,(1920,1080))
                    im = F.to_tensor(original_im)
                    im = F.normalize(im,mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
                    # store preprocessed image, dimensions and original image
                    im = im.to(device)
                    dim = None
                    if show:
                        frame = (frame_idx,im,dim,original_im)
                    else:
                        frame = (frame_idx,im,dim,None)
                    # append to queue
                    image_queue.put(frame)       
                    frame_idx += 1
            
            else: # load dummy info, won't be used
                # load next image from videocapture object
                ret = cap.grab()
                if ret == False:
                    frame = (-1,None,None,None)
                    image_queue.put(frame)       
                    break
                else:
                    frame = (frame_idx,None,None,None)
                    image_queue.put(frame)       
                    frame_idx += 1
                    
    # neverending loop, because if the process ends, the tensors originally
    # initialized in this function will be deleted, causing issues. Thus, this 
    # function runs until a call to self.next() returns -1, indicating end of track 
    # has been reached
    while True:  
           time.sleep(5)
    
    
        
if __name__ == "__main__":
    
    track_dir = "/home/worklab/Desktop/detrac/DETRAC-all-data"
    label_dir = "/home/worklab/Desktop/detrac/DETRAC-Train-Annotations-XML-v3"
    track_list = [os.path.join(track_dir,item) for item in os.listdir(track_dir)]  
    label_list = [os.path.join(label_dir,item) for item in os.listdir(label_dir)] 
    track_dict = {}
    for item in track_list:
        id = int(item.split("MVI_")[-1])
        track_dict[id] = {"frames": item,
                          "labels": None}
    for item in label_list:
        id = int(item.split("MVI_")[-1].split("_v3.xml")[0])
        track_dict[id]['labels'] = item
        
    path = track_dict[40962]['frames']     
    test = FrameLoader(path,torch.device("cuda:0"),det_step = 10, init_frames = 3)

    all_time = 0
    print(test.queue.qsize())
    count = 0
    while True:
        start = time.time()
        num, frame = next(test)
        
        if num > 0:
            all_time += (time.time() - start)
        
        time.sleep(0.03)
       
        if frame is not None:
            out = frame[0] + 1
        
        if num == -1:
            break
        count += 1
        print(count, all_time/count)