#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 09:58:20 2020

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

class OutputWriter():
    
    def __init__(self,output_file):
            
        
        # create shared queue
        ctx = mp.get_context('spawn')
        self.queue = ctx.Queue()
        
        self.worker = ctx.Process(target=write_frames, args=(self.queue,output_file))
        self.worker.start()
        
    def __call__(self,frame):
        self.queue.put(frame)
        
def write_frames(queue,output_file):
    
    frame = 0
    
    while True:
        try:
            im = queue.get(timeout = 10)
            
            cv2.imwrite(os.path.join(output_file,"{}.png".format(str(frame).zfill(5))),im*255)
            frame += 1
            
        except:
            break
        
