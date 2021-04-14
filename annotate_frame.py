#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 21:29:21 2021

@author: worklab
"""


# using openCV, draw ignored regions for each frame. 
# Press d to remove all ignored regions
# otherwise they carry over from frame to frame
# Press 2 to advance frames, 1 to reverse frames
import numpy as np
import os
import cv2
import csv
import torch
import argparse
import _pickle as pickle
import cv2 as cv
from PIL import Image
from torchvision.transforms import functional as F

def get_frames(in_dir,out_dir):
    
    for video in os.listdir(in_dir):
        if ".mp4" not in video:
            continue
        
        out_path = os.path.join(out_dir,video.split(".mp4")[0])
        os.mkdir(out_path)
        
        video = os.path.join(in_dir,video)
        
        cap = cv2.VideoCapture(video)
        for i in range(50):
            ret,original_im = cap.read()
            
            
            frame_path = os.path.join(out_path,str(i).zfill(5)+".png")
            cv2.imwrite(frame_path,original_im)
            
        cap.release()

def plot_outputs(output_file,im_dir,ignore = None):
    
    if ignore is not None:
        with open(ignore,"rb") as f:
            ignored = pickle.load(f)
    else:
        ignored = None        
    
    frame_dict = {}
    
    with open(output_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            frame,obj_idx,left,top,width,height,conf,cls,visibility,_ = row
            frame = int(frame)
            obj_idx = int(obj_idx)
            left = float(left)
            top = float(top)
            width = float(width)
            height = float(height)
            conf = float(conf)
            cls = 0#torch.tensor(int(cls)).int()
            visibility = float(visibility)
            bbox = torch.tensor([left,top,left+width,top+height]).int()
            
            if frame not in frame_dict.keys():
                frame_dict[frame] = []
            
            frame_dict[frame].append([bbox,cls,conf,visibility,obj_idx])

    ims = [os.path.join(im_dir,item) for item in os.listdir(im_dir)]
    ims.sort()
    idx_colors = np.random.rand(10000,3)
    idx_colors[0] = np.array([0,1,0])
    idx_colors[1] = np.array([0,0,1])
    
    ff = 0
    for im_name in ims:
        
        frame = int(im_name.split("/")[-1].split(".jpg")[0])
        # open image
        
        im = Image.open(im_name)
        im = F.to_tensor(im)
        cv_im = np.array(im) 
        cv_im = np.clip(cv_im, 0, 1)
        
        # Convert RGB to BGR 
        cv_im = cv_im[::-1, :, :]         
        
        cv_im = np.moveaxis(cv_im,[0,1,2],[2,0,1])

        cv_im = cv_im.copy()

        
        im_idx = int(im_name.split("/")[-1].split(".jpg")[0])
        label = frame_dict[im_idx]
        
        for obj in label:
            bbox = obj[0]
            bbox = bbox.int().data.numpy()
            cls = obj[1]
            
            # check whether
            if ignored is not None:
                for region in ignored[frame]:
                    area = (bbox[2] - bbox[0])     * (bbox[3] - bbox[1])
                    xmin = max(region[0],bbox[0])
                    xmax = min(region[2],bbox[2])
                    ymin = max(region[1],bbox[1])
                    ymax = min(region[3],bbox[3])
                    intersection = max(0,(xmax - xmin)) * max(0,(ymax - ymin))
                    overlap = intersection  / (area)
                    
                    if overlap > 0.5:
                        cls = 1
            
            cv2.rectangle(cv_im,(bbox[0],bbox[1]),(bbox[2],bbox[3]), idx_colors[cls], 1)
            #plot_text(cv_im,(bbox[0],bbox[1]),obj_idx,0,class_colors,self.class_dict)
        
        
        # for region in metadata["ignored_regions"]:
        #     bbox = region.astype(int)
        #     cv2.rectangle(cv_im,(bbox[0],bbox[1]),(bbox[2],bbox[3]), class_colors[-1], 1)
       
    
        cv2.imshow("Frame",cv_im)
        cv2.waitKey(ff) 
        if ff == 0:
            ff = 1
    cv2.destroyAllWindows()


class Frame_Labeler():
    
    def __init__(self,directory,classes = ["Car Source","Truck Source", "Sink","Ignore", "Ignore for source"]):
        self.frame = 1
        
        self.directory = directory
        lis = os.listdir(directory)
        self.frames = [os.path.join(directory,item) for item in lis]
        self.frames.sort()
        
        
        self.frame_boxes = {}
        
        self.load_annotations()
        
        self.cur_image = cv2.imread(self.frames[0])
        
        self.start_point = None # used to store click temporarily
        self.clicked = False 
        self.new = None # used to store a new box to be plotted temporarily
        self.cont = True
        self.define_direction = False
        
        # classes
        self.cur_class = 0
        self.n_classes = len(classes)
        self.class_names = classes
        self.colors = (np.random.rand(self.n_classes,3))*255
        self.colors[0] = np.array([0,255,0])
        self.colors[1] = np.array([255,0,0])
        self.colors[2] = np.array([0,0,255])
        self.colors[3] = np.array([0,0,0])

        self.plot_boxes()
        self.changed = False

    def load_annotations(self):
        try:
            self.cur_frame_boxes = []
            name = "_fov_annotations/{}.csv".format(self.frames[self.frame-1].split("/")[-1].split(".")[0])
            with open(name,"r") as f:
                read = csv.reader(f)
                for row in read:
                    count = 0
                    for item in row:
                        if len(item) > 0:
                            count += 1
                    row = row[:count]
                    if len(row) in [5,6]:
                        row = [int(float(item)) for item in row]
                    elif len(row) > 6:
                        row = [int(float(item)) for item in row[:5]] + [float(item) for item in row[5:]]
                    self.cur_frame_boxes.append(np.array(row))
                    
        except FileNotFoundError:
            self.cur_frame_boxes = []
        
    def plot_boxes(self):
        self.cur_image = cv2.imread(self.frames[self.frame-1])

        last_source_idx = 0
        for idx, box in enumerate(self.cur_frame_boxes):
            self.cur_image = cv2.rectangle(self.cur_image,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),self.colors[int(box[4])],2)
            label = "{} {}".format(self.class_names[int(box[4])],idx+1)
            
            if box[4] in [0,1]:
                last_source_idx = idx+1
            elif box[4] == 2: # if sink, find index of associated source
                letter = idx - last_source_idx
                letters = ["a","b","c","d","e","f","g","h"]
                letter = letters[letter]
                label = "{} {}{} (Movement {})".format(self.class_names[int(box[4])],last_source_idx,letter,box[5])
            
            self.cur_image = cv2.putText(self.cur_image, label, (int(box[0]),int(box[1])-5), cv2.FONT_HERSHEY_PLAIN,1, [0,0,0], 3)
            self.cur_image = cv2.putText(self.cur_image, label, (int(box[0]),int(box[1])-5), cv2.FONT_HERSHEY_PLAIN,1, [225,255,255], 1)
            
            if len(box) > 7:
                self.cur_image = cv2.line(self.cur_image,(int(box[0]),int(box[1])),(int(box[0]) + int(box[5]*box[7]),int(box[1]) + int(box[6]*box[7])),self.colors[int(box[4])],2)
                
                
    def toggle_class(self):
        self.cur_class = (self.cur_class + 1) % self.n_classes
        print("Active Class: {}".format(self.class_names[self.cur_class]))
        
    def on_mouse(self,event, x, y, flags, params):
       
    
       if event == cv.EVENT_LBUTTONDOWN and not self.clicked:
         self.start_point = (x,y)
         self.clicked = True
         self.changed = True
         
       elif event == cv.EVENT_LBUTTONUP:
          if not self.define_direction:
              if self.cur_class == 2:
                  movement = int(input("Source-sink movement:"))
                  box = np.array([self.start_point[0],self.start_point[1],x,y,self.cur_class,movement]).astype(int)
              else:
                  box = np.array([self.start_point[0],self.start_point[1],x,y,self.cur_class]).astype(int)
                  
              self.cur_frame_boxes.append(box)
              self.new = box
              self.clicked = False
            
              if self.cur_class in [0,1]:
                  self.define_direction = True
                  
            
        
          elif self.define_direction:
              box = self.cur_frame_boxes[-1]
              x0,y0 = self.start_point
              magnitude = np.sqrt((x-x0)**2 + (y-y0)**2)
              x_comp = (x-x0)/magnitude
              y_comp = (y-y0)/magnitude
            
              line = np.array([x_comp,y_comp,magnitude])
              new_box = np.concatenate((box,line))
              self.cur_frame_boxes[-1] = new_box
              
              self.clicked = False
              self.new = new_box
              self.define_direction = False
              self.cur_class = 2


              
    
    def next(self):
        self.save_frame()
        self.clicked = False
        self.define_direction = False
        
        # store current boxes
        self.frame_boxes[self.frame] = self.cur_frame_boxes
        
        if self.frame == len(self.frames):
            print("Last Frame.")    
            
        else:
            self.frame += 1
            self.load_annotations()
            
            # load image and plot existing boxes
            self.cur_image = cv2.imread(self.frames[self.frame-1])
            self.plot_boxes()
            self.changed = False
                
    def prev(self):
        self.save_frame()
        self.clicked = False
        self.define_direction = False
        
        if self.frame == 1:
            print("On first frame. Cannot go to previous frame")
        else:
            self.frame_boxes[self.frame] = self.cur_frame_boxes
            self.frame -= 1
            
            self.load_annotations()
            
            # load image and plot existing boxes
            self.cur_image = cv2.imread(self.frames[self.frame-1])
            self.plot_boxes()
            self.changed = False
                
            
    def quit(self):
        self.frame_boxes[self.frame] = self.cur_frame_boxes
        
        cv2.destroyAllWindows()
        self.cont = False
        print("Images are from {}".format(self.directory))
        name = input("Save file name (Enter q to discard):")
        if name == "q":
            print("Labels discarded")
        else:
            with open(name,"wb") as f:
                pickle.dump(self.frame_boxes,f)
            print("Saved boxes as file {}".format(name))
        
    def save_frame(self):
        if len(self.cur_frame_boxes) == 0 or not self.changed:
            return
        name = "_fov_annotations/{}.csv".format(self.frames[self.frame-1].split("/")[-1].split(".")[0])
        if os.path.exists(name): # don't overwrite
            overwrite = input("Overwrite existing file? (y/n)")
            if overwrite != "y":
                return
        
        outputs = []
        for item in self.cur_frame_boxes:
            output = list(item)
            outputs.append(output)
        
        with open(name,"w") as f:
            writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            writer.writerows(outputs)
        print("Saved boxes as file {}".format(name))
        
        name = name.split(".csv")[0] + ".png"
        cv2.imwrite(name,self.cur_image)
        
        
    def clear(self):
        self.cur_frame_boxes = []
        self.cur_image = cv2.imread(self.frames[self.frame-1])

    def undo(self):
        self.clicked = False
        self.define_direction = False
        self.define_magnitude = False
        
        self.cur_frame_boxes = self.cur_frame_boxes[:-1]
        self.cur_image = cv2.imread(self.frames[self.frame-1])
        self.plot_boxes()
        
        
    def run(self):  
        self.frame_boxes[self.frame] = self.cur_frame_boxes

        cv2.namedWindow("window")
        cv.setMouseCallback("window", self.on_mouse, 0)
           
        while(self.cont): # one frame
        
           if self.new is not None:
               self.plot_boxes()
                    
           self.new = None
               
           cv2.imshow("window", self.cur_image)
           title = "{} toggle class (1), switch frame (8-9), clear all (c), undo(u), delete(d),  quit (q), switch sequence (8-9)".format(self.class_names[self.cur_class])
           cv2.setWindowTitle("window",str(title))
           
           key = cv2.waitKey(1)
           if key == ord('9'):
                self.next()
           elif key == ord('8'):
                self.prev()
           elif key == ord('c'):
                self.clear()
           elif key == ord("q"):
                self.quit()
           elif key == ord("1"):
                self.toggle_class()
           elif key == ord("u"):
               self.undo()
           elif key == ord("d"):
               self.remove()
     
    def remove(self):
        try:
            idx = int(input("Enter source to be deleted:")) -1
        except:
            return
        
        # removes a single source and all associated sinks
        if idx < 0 or idx > len(self.cur_frame_boxes):
            return
        
        # get source after idx
        next_source_idx = None
        for i in range(idx+1,len(self.cur_frame_boxes)):
            if self.cur_frame_boxes[i][4] in [0,1]:
                next_source_idx = i
                break
            
        print("Deleting boxes {} to {}".format(idx+1,next_source_idx))
        # remove from idx to next_source_idx
        keep = []
        if next_source_idx is not None:
            for i in range(len(self.cur_frame_boxes)):
                if i < idx or i >= next_source_idx:
                    keep.append(self.cur_frame_boxes[i])
        else:
            for i in range(len(self.cur_frame_boxes)):
                if i < idx:
                    keep.append(self.cur_frame_boxes[i])
        self.cur_frame_boxes = keep
        self.plot_boxes()
              
            
if __name__ == "__main__":
      
     #add argparse block here so we can optinally run from command line
     #add argparse block here so we can optinally run from command line
     try:
        parser = argparse.ArgumentParser()
        parser.add_argument("directory",help = "directory where frame images are stored")
        parser.add_argument("-classes", help = "list separated by commas",default = "Class1,Class2")


        args = parser.parse_args()
        
        dir = args.directory
        classes = args.classes
        
        frame_labeler = Frame_Labeler(dir,classes)
        frame_labeler.run()

     except:
        dir = "/home/worklab/Data/cv/AIC21_Track1_Vehicle_Counting_full/AIC21_Track1_Vehicle_Counting/screen_shot_with_roi_and_movement"
            
        test = Frame_Labeler(dir)
        test.run()
            
        #get_frames("/home/worklab/Data/cv/AIC21_Track1_Vehicle_Counting_full/AIC21_Track1_Vehicle_Counting/Dataset_A","frames_from_each")