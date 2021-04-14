
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 15:46:21 2020

@author: worklab
"""


import numpy as np
import random 
import time
import csv

random.seed = 0
import cv2
import torch
from torchvision.ops import roi_align
import matplotlib.pyplot  as plt
from scipy.optimize import linear_sum_assignment
from torchvision.ops import nms


# filter and frame loader
from util_track.mp_loader import FrameLoader
from util_track.kf import Torch_KF
from util_track.mp_writer import OutputWriter



class LBT_Count():
    
    def __init__(self,
                 track_dir,
                 video_id,
                 localizer,
                 kf_params,
                 config_file,
                 cam_annotations,
                 class_dict,
                 movement_conversions,
                 PLOT = True,
                 device_id = 0):
        """
         Parameters
        ----------
        track_dir : str
            path to directory containing ordered track images
        detector : object detector with detect function implemented that takes a frame and returns detected object
        localizer : CNN object localizer
        kf_params : dictionary
            Contains the parameters to initialize kalman filters for tracking objects
        det_step : int optional
            Number of frames after which to perform full detection. The default is 1.
        init_frames : int, optional
            Number of full detection frames before beginning localization. The default is 3.
        fsld_max : int, optional
            Maximum dense detection frames since last detected before an object is removed. 
            The default is 1.
        matching_cutoff : int, optional
            Maximum distance between first and second frame locations before match is not considered.
            The default is 100.
        iou_cutoff : float in range [0,1], optional
            Max iou between two tracked objects before one is removed. The default is 0.5.       
        ber : float, optional
            How much bounding boxes are expanded before being fed to localizer. The default is 1.
        PLOT : bool, optional
            If True, resulting frames are output. The default is True. 
        """
        
        self.sequence_name = track_dir
        self.sequence_id = video_id
        
        cam_num = self.sequence_name.split("/cam_")[-1][:2]
        try:
             self.cam_num = int(cam_num)
        except:
             self.cam_num = int(cam_num[0])
        
        self.PLOT = PLOT
        
        self.output_file = "outputs/counts_{}.txt".format(video_id)
        
        # CUDA
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:{}".format(device_id) if use_cuda else "cpu")
        torch.cuda.set_device(device_id)
        
        torch.cuda.empty_cache() 
        
        # store localizer
        self.localizer = localizer.to(self.device)
        self.localizer.eval()
        self.localizer.training = False
        
        # store class_dict
        self.class_dict = class_dict
        self.movement_conversions = movement_conversions
        
        # store camera annotations
        self.cam = cam_annotations
        self.sources = torch.stack([item[0] for item in self.cam["car"]]+ [item[0] for item in self.cam["truck"]])
        self.sinks = [item[1] for item in self.cam["car"]]+ [item[1] for item in self.cam["truck"]]
        self.last_sink_obj = [[-1 for idx in item[1]] for item in self.cam["car"]] + [[-1 for idx in item[1]] for item in self.cam["truck"]]

        self.source_box_classes = torch.tensor([0 for item in self.cam["car"]]+ [1 for item in self.cam["truck"]]).to(self.device)
        self.avg_source_obj_sizes = np.zeros(len(self.sources))

        
        # parse config file here
        params = self.parse_config_file(config_file)[0]
        
        
        #store parameters
        self.s                = params["skip_step"]
        self.fsld_max         = params["fsld_max"]
        self.ber              = params["ber"]
        self.sink_iou         = params['sink_iou']
        self.ignore_iou       = params["ignore_iou"]
        self.conf_new         = params["conf_new"]
        self.iou_new_nms      = params["iou_new"]
        self.iou_overlap      = params["iou_overlap"]
        self.conf_loc         = params["conf_loc"]
        self.iou_loc          = params["iou_loc"]
        self.cs = cs          = params["cs"]
        self.W                = params["W"]
        self.t_cutoff         = params["t_cutoff"]
        self.keep_classes     = params["keep_classes"]
        R_red                 = params["R_red"]
        OUT                   = params["output_video_path"]
        
        
        self.s = 1
        print("Warning: s is overridden on line 134!")
       
        # store filter params
        kf_params["R"] /= R_red
        self.state_size = kf_params["Q"].shape[0]
        self.filter = Torch_KF(torch.device("cpu"),INIT = kf_params)
       
        fls = 1 if self.PLOT else self.s
        self.loader = FrameLoader(track_dir,self.device,downsample = 1,s = fls,show = self.PLOT)
        
        # create output image writer
        if OUT is not None:
            self.writer = OutputWriter(OUT)
        else:
            self.writer = None
        
        time.sleep(1)
        self.n_frames = len(self.loader)
    
        self.next_obj_id = 0             # next id for a new object (incremented during tracking)
        self.fsld = {}                   # fsld[id] stores frames since last detected for object id
    
        self.all_tracks = {}             # stores states for each object
        self.all_classes = {}            # stores class evidence for each object
        self.all_confs = {}
        self.all_sources = {}
        self.all_first_frames = {}
        # for keeping track of what's using time
        self.time_metrics = {            
            "load":0,
            "predict":0,
            "pre_localize and align":0,
            "localize":0,
            "post_localize":0,
            "detect":0,
            "parse":0,
            "match":0,
            "update":0,
            "add and remove":0,
            "store":0,
            "plot":0,
            "blackout":0,
            }
        
        self.idx_colors = np.random.rand(100,3)
    
        self.n_objs = [] # for getting avg objs per frame
        
        
    def parse_config_file(self,config_file):
        all_blocks = []
        current_block = {}
        with open(config_file, 'r') as f:
            for line in f:
                # ignore empty lines and comment lines
                if line is None or len(line.strip()) == 0 or line[0] == '#':
                    continue
                strip_line = line.split("#")[0].strip()
    
                if '==' in strip_line:
                    pkey, pval = strip_line.split('==')
                    pkey = pkey.strip()
                    pval = pval.strip()
                    
                    # parse out non-string values
                    try:
                        pval = int(pval)
                    except ValueError:
                        try:    
                            pval = float(pval)    
                        except ValueError:
                            pass 
                    if pval == "None":
                        pval = None
                    elif pval == "True":
                        pval = True
                    elif pval == "False":
                        pval = False
                    elif type(pval) == str and "," in pval:
                        pval = [int(item) for item in pval.split(",")]
                        
                    current_block[pkey] = pval
                    
                else:
                    raise AttributeError("""Got a line in the configuration file that isn't a block header nor a 
                    key=value.\nLine: {}""".format(strip_line))
            # add the last block of the file (if it's non-empty)
            all_blocks.append(current_block)
            
        return all_blocks
      
    
    def crop_tracklets(self,boxes,frame,ber = None):
        """
        Crops relevant areas from frame based on a priori (pre_locations) object locations
        """
        if ber is None:
            ber = self.ber
            
        #box_ids = []
        #box_list = []
        
        # # convert to array
        # for id in pre_locations:
        #     box_ids.append(id)
        #     box_list.append(pre_locations[id][:4])
        # boxes = np.array(box_list)
        # boxes = pre_locations
    
        temp = np.zeros(boxes.shape)
        temp[:,0] = (boxes[:,0] + boxes[:,2])/2.0
        temp[:,1] = (boxes[:,1] + boxes[:,3])/2.0
        temp[:,2] =  boxes[:,2] - boxes[:,0]
        temp[:,3] = (boxes[:,3] - boxes[:,1])/(temp[:,2] + 1e-07)
        boxes = temp
    
        # first row of zeros is batch index (batch is size 0) for ROI align
        new_boxes = np.zeros([len(boxes),5]) 

        # use either s or s x r for both dimensions, whichever is smaller,so crop is square
        box_scales = np.max(np.stack((boxes[:,2],boxes[:,2]*boxes[:,3]),axis = 1),axis = 1) #/2.0
            
        #expand box slightly
        box_scales = box_scales * ber# box expansion ratio
        
        new_boxes[:,1] = boxes[:,0] - box_scales/2
        new_boxes[:,3] = boxes[:,0] + box_scales/2 
        new_boxes[:,2] = boxes[:,1] - box_scales/2 
        new_boxes[:,4] = boxes[:,1] + box_scales/2 
        
        torch_boxes = torch.from_numpy(new_boxes).float().to(self.device)
    
        # crop using roi align 
        crops = roi_align(frame.unsqueeze(0),torch_boxes,(self.cs,self.cs))
        
        return crops,new_boxes,box_scales
    
          
            
    def local_to_global(self,reg_out,box_scales,new_boxes):
        """
        reg_out - tensor of shape [n_crops, n_anchors, 4]
        box_scales - tensor of shape [n_crops]
        new_boxes - tensor of shape n_crops,4
        """
        detections = reg_out
        # detections = (reg_out* 224*wer - 224*(wer-1)/2)
        # detections = detections.data.cpu()
        n_anchors = detections.shape[1]
        
        box_scales = torch.from_numpy(box_scales).to(self.device).unsqueeze(1).repeat(1,n_anchors)
        new_boxes = torch.from_numpy(new_boxes).to(self.device).unsqueeze(1).repeat(1,n_anchors,1)
        
        # add in original box offsets and scale outputs by original box scales
        detections[:,:,0] = detections[:,:,0]*box_scales/self.cs + new_boxes[:,:,1]
        detections[:,:,2] = detections[:,:,2]*box_scales/self.cs + new_boxes[:,:,1]
        detections[:,:,1] = detections[:,:,1]*box_scales/self.cs + new_boxes[:,:,2]
        detections[:,:,3] = detections[:,:,3]*box_scales/self.cs + new_boxes[:,:,2]

        # convert into xysr form 
        # output = np.zeros([len(detections),4])
        # output[:,0] = (detections[:,0] + detections[:,2]) / 2.0
        # output[:,1] = (detections[:,1] + detections[:,3]) / 2.0
        # output[:,2] = (detections[:,2] - detections[:,0])
        # output[:,3] = (detections[:,3] - detections[:,1]) / output[:,2]
        
        return detections
    
    
    def remove_overlaps(self):
        """
        Checks IoU between each set of tracklet objects and removes the newer tracklet
        when they overlap more than iou_cutoff (likely indicating a tracklet has drifted)
        """
        if self.iou_overlap > 0:
            removals = []
            locations = self.filter.objs()
            for i in locations:
                for j in locations:
                    if i != j:
                        iou_metric = self.iou(locations[i],locations[j])
                        if iou_metric > self.iou_overlap:
                            # determine which object has been around longer
                            if self.all_first_frames[j] > self.all_first_frames[i]:
                                removals.append(j)
                            else:
                                removals.append(i)
            removals = list(set(removals))
            self.filter.remove(removals)
   
    
    def remove_anomalies(self,max_scale= 1200):
        """
        Removes all objects with negative size or size greater than max_size
        """
        removals = []
        locations = self.filter.objs()
        for i in locations:
            if (locations[i][2]-locations[i][0]) > max_scale or (locations[i][2]-locations[i][0]) < 0:
                removals.append(i)
            elif (locations[i][3] - locations[i][1]) > max_scale or (locations [i][3] - locations[i][1]) < 0:
                removals.append(i)
        self.filter.remove(removals)         
    
    def iou(self,a,b):
        """
        Description
        -----------
        Calculates intersection over union for all sets of boxes in a and b
    
        Parameters
        ----------
        a : tensor of size [batch_size,4] 
            bounding boxes
        b : tensor of size [batch_size,4]
            bounding boxes.
    
        Returns
        -------
        iou - float between [0,1]
            average iou for a and b
        """
        
        area_a = (a[2]-a[0]) * (a[3]-a[1])
        area_b = (b[2]-b[0]) * (b[3]-b[1])
        
        minx = max(a[0], b[0])
        maxx = min(a[2], b[2])
        miny = max(a[1], b[1])
        maxy = min(a[3], b[3])
        
        intersection = max(0, maxx-minx) * max(0,maxy-miny)
        union = area_a + area_b - intersection
        iou = intersection/union
        
        return iou

    def plot(self,im,detections,post_locations,all_classes,class_dict,title = None):
        """
        Description
        -----------
        Plots the detections and the estimated locations of each object after 
        Kalman Filter update step
    
        Parameters
        ----------
        im : cv2 image
            The frame
        detections : tensor [n,4]
            Detections output by either localizer or detector (xysr form)
        post_locations : tensor [m,4] 
            Estimated object locations after update step (xysr form)
        all_classes : dict
            indexed by object id, where each entry is a list of the predicted class (int)
            for that object at every frame in which is was detected. The most common
            class is assumed to be the correct class        
        class_dict : dict
            indexed by class int, the string class names for each class
        frame : int, optional
            If not none, the resulting image will be saved with this frame number in file name.
            The default is None.
        """
        
        im = im.copy()/255.0
    
        #plot ignored regions
        for box in self.cam["ignore"]:
            color = (0.1,0.1,0.1) #colors[int(obj.cls)]
            c1 =  (int(box[0]),int(box[1]))
            c2 =  (int(box[2]),int(box[3]))
            im = cv2.rectangle(im,c1,c2,color,-1)
            
        #plot ignored regions
        for i,box in enumerate(self.sources):
            if self.source_box_classes[i] == 0:
                color = (0.1,0.9,0.1) #colors[int(obj.cls)]
            else:
                color = (0.9,0.1,0.1)
            c1 =  (int(box[0]),int(box[1]))
            c2 =  (int(box[2]),int(box[3]))
            im = cv2.rectangle(im,c1,c2,color,1)
            
        for s_idx, source in enumerate(self.sinks):
            for b_idx, box in enumerate(source):
                if self.last_sink_obj[s_idx][b_idx] == title:
                    color = (1,1,1)
                elif self.last_sink_obj[s_idx][b_idx] + box[6] > title:
                    color = (0.5,0.5,0.5)
                else:
                    color = (0.1,0.1,0.9) #colors[int(obj.cls)]
                c1 =  (int(box[0]),int(box[1]))
                c2 =  (int(box[2]),int(box[3]))
                im = cv2.rectangle(im,c1,c2,color,1)
    
        # plot detection bboxes
        for det in detections:
            bbox = det[:4]
            color = (0.4,0.4,0.7) #colors[int(obj.cls)]
            c1 =  (int(bbox[0]),int(bbox[1]))
            c2 =  (int(bbox[2]),int(bbox[3]))
            im = cv2.rectangle(im,c1,c2,color,1)
            
        # plot estimated locations
        for id in post_locations:
            # get class
            try:
                most_common = np.argmax(all_classes[id])
                cls = class_dict[most_common]
            except:
                cls = "" 
            label = "{} {}".format(cls,id)
            bbox = post_locations[id][:4]
            
            if sum(bbox) != 0: # all 0's is the default in the storage array, so ignore these
                try:
                    color = self.idx_colors[id%100]
                    c1 =  (int(bbox[0]),int(bbox[1]))
                    c2 =  (int(bbox[2]),int(bbox[3]))
                    im= cv2.rectangle(im,c1,c2,color,1)
                except:
                    pass
                
                # plot label
                text_size = 0.8
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN,text_size , 1)[0]
                c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
                cv2.rectangle(im, c1, c2,color, -1)
                cv2.putText(im, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN,text_size, [225,255,255], 1)
        
        # resize to fit on standard monitor
        #if im.shape[0] > 1920:
        im = cv2.resize(im, (1920,1080))
        cv2.imshow("frame",im)
        cv2.setWindowTitle("frame",str(title))
        cv2.waitKey(1)
        
        if self.writer is not None:
            self.writer(im)


        
    def md_iou(self,a,b):
        """
        a,b - [batch_size x num_anchors x 4]
        """
        
        area_a = (a[:,:,2]-a[:,:,0]) * (a[:,:,3]-a[:,:,1])
        area_b = (b[:,:,2]-b[:,:,0]) * (b[:,:,3]-b[:,:,1])
        
        minx = torch.max(a[:,:,0], b[:,:,0])
        maxx = torch.min(a[:,:,2], b[:,:,2])
        miny = torch.max(a[:,:,1], b[:,:,1])
        maxy = torch.min(a[:,:,3], b[:,:,3])
        zeros = torch.zeros(minx.shape,dtype=float).to(self.device)
        
        intersection = torch.max(zeros, maxx-minx) * torch.max(zeros,maxy-miny)
        union = area_a + area_b - intersection
        iou = torch.div(intersection,union)
        
        #print("MD iou: {}".format(iou.max(dim = 1)[0].mean()))
        return iou
    
    def parse_sources(self,source_confs,source_classes,source_boxes,frame_num):
        """
        Boxes output from source areas are parsed and used to initialize new objects
        source_confs - [] tensor
        source_classes - [] tensor
        source-boxes - [] tensor
        """
        #store source for each box 
        source_ids = torch.tensor([i for i in range(len(source_boxes))]).unsqueeze(1)
        source_ids = source_ids.repeat(1,len(source_boxes[0]))
        source_ids = source_ids.reshape(-1,1)
    
        # source_box_classes 
        #source_box_classes = self.source_box_classes.unsqueeze(1).repeat(1,len(source_boxes[1]))
        #source_box_classes = source_box_classes.reshape(-1,1).squeeze(1)
        
        # flatten all sources into one dim
        source_confs = source_confs.reshape(-1)
        #source_classes = source_classes.reshape(-1)
        source_boxes = source_boxes.reshape(-1,4)
        
        # remove all objects with class other than desired classes
        # keep = torch.where(source_classes == source_box_classes)[0]
        # source_confs = source_confs[keep]
        # source_classes = source_classes[keep]
        # source_boxes = source_boxes[keep]
        # source_ids = source_ids[keep]

        # remove all objects with conf lower than self.conf_new
        keep = torch.where(source_confs > self.conf_new)[0]
        source_confs = source_confs[keep]
        #source_classes = source_classes[keep]
        source_boxes = source_boxes[keep]
        source_ids = source_ids[keep]

        # perform nms on boxes
        keep = nms(source_boxes,source_confs,self.iou_new_nms)
        source_confs = source_confs[keep]
        #source_classes = source_classes[keep]
        source_boxes = source_boxes[keep]
        source_ids = source_ids[keep]
        
        # add new objects
        new_ids = []
        new_speeds = []
        cur_row = 0
        for i in range(len(source_boxes)):                
            new_ids.append(self.next_obj_id)

            self.fsld[self.next_obj_id] = 0
            self.all_tracks[self.next_obj_id] = np.zeros([self.n_frames,self.state_size])
            self.all_classes[self.next_obj_id] = np.zeros(13)
            self.all_confs[self.next_obj_id] = []
            
            # update average source obj size
            avg_source_size = self.avg_source_obj_sizes[source_ids[i]]
            this_size = (source_boxes[i,2]-source_boxes[i,0])
            if avg_source_size == 0:
                self.avg_source_obj_sizes[source_ids[i]] = self.t_cutoff
            else:
                self.avg_source_obj_sizes[source_ids[i]] *= 0.8
                self.avg_source_obj_sizes[source_ids[i]] += 0.2*this_size
            
            if this_size > 1.5*self.avg_source_obj_sizes[source_ids[i]]:
                cls = 1
                self.all_classes[self.next_obj_id][cls] += 50

            else:
                cls = 0
                self.all_classes[self.next_obj_id][cls] += 1

            #cls = source_classes[i]
            
            self.all_classes[self.next_obj_id][cls] += 1
            self.all_confs[self.next_obj_id].append(source_confs[i])
            self.all_sources[self.next_obj_id] = source_ids[i]
            self.all_first_frames[self.next_obj_id] = frame_num
            self.next_obj_id += 1
            cur_row += 1
            
            speed = self.sources[source_ids[i],5:][0]
            if cls ==1:
                speed = speed * 0.75 # reduce truck speed
                
            new_speeds.append(speed)
        if len(source_boxes) > 0:        
            speeds = torch.stack(new_speeds).to(self.device).float()
            source_boxes = torch.cat((source_boxes,speeds),dim = 1)
            self.filter.add(source_boxes,new_ids)

    def remove_sinks(self,frame_num):
        removals = []
        locations = self.filter.objs()
        for id in locations:
            bbox = locations[id][:4]
            for sink_idx,region in enumerate(self.sinks[self.all_sources[id]]):
                ox = (bbox[2] + bbox[0]) /2.0
                oy = (bbox[3] + bbox[1]) /2.0
                
                last_sink_obj = self.last_sink_obj[self.all_sources[id]][sink_idx]
                
                
                if region[0] < ox and region[2] > ox and region[1] < oy and region[3] > oy: # object center within sink
                    removals.append(id)
                    
                    # print output 
                    movement = int(region[5].item()) # movement is stored in each sink, labeled during annotation
                    #print(movement)
                    if movement != -1:
                        runtime = np.round((time.time() - self.start_time),4)
                        frame_id = frame_num + 1 # since competition is 1-indexed
                        cls = np.argmax(self.all_classes[id]) + 1
                        
                        # some fiddly truck logic
                        if bbox[2] - bbox[0] > self.t_cutoff: # big objects are trucks
                            cls = 2
                        if self.t_cutoff == -1:
                            cls = 1
                        if self.cam_num ==10:
                            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                            if area > 400**2:
                                cls = 2
                        elif self.cam_num==11:
                            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                            if area > 375**2:
                                cls = 2
                        elif self.cam_num==12:
                            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                            if area > 450**2:
                                cls = 2
                                
                        if last_sink_obj + region[6] <= frame_num: # only add object if sink is valid
                            
                            self.last_sink_obj[self.all_sources[id]][sink_idx] = frame_num
                            
                            row = [runtime,self.sequence_id,frame_id,movement,cls]
                        
                            with open(self.output_file, 'a') as f:
                                writer = csv.writer(f,delimiter = " ")
                                writer.writerow(row)
                    
                    break # move on to next object
                
        removals = list(set(removals))
        self.filter.remove(removals)

    def track(self):
        """
        Returns
        -------
        final_output : list of lists, one per frame
            Each sublist contains dicts, one per estimated object location, with fields
            "bbox", "id", and "class_num"
        frame_rate : float
            number of frames divided by total processing time
        time_metrics : dict
            Time utilization for each operation in tracking
        """    
        
        self.start_time = time.time()
        [frame_num, frame,dim,original_im] = next(self.loader)            
        self.time_metrics["load"] += time.time() - self.start_time
        
        while frame_num != -1:            
            
            #if frame_num % self.d < self.init_frames:
            # predict next object locations
            start = time.time()
            try: # in the case that there are no active objects will throw exception
                self.filter.predict()
                pre_locations = self.filter.objs()
            except:
                pre_locations = []    
                
            pre_ids = []
            pre_loc = []
            for id in pre_locations:
                pre_ids.append(id)
                pre_loc.append(pre_locations[id])
            pre_loc = np.array(pre_loc)
            
            self.time_metrics['predict'] += time.time() - start
        
            start = time.time()
            if (frame_num % self.s) == 0:
                
                # black out ignored regions
                for region in self.cam["ignore"]:
                    r =  torch.normal(0.485,0.229,[int(region[3])-int(region[1]),int(region[2])-int(region[0])],device = self.device)
                    g =  torch.normal(0.456,0.224,[int(region[3])-int(region[1]),int(region[2])-int(region[0])],device = self.device)
                    b =  torch.normal(0.406,0.225,[int(region[3])-int(region[1]),int(region[2])-int(region[0])],device = self.device)
                    rgb = torch.stack([r,g,b])
                    frame[:,region[1]:region[3],region[0]:region[2]] = rgb
                
                if len(pre_locations) > 0:
                    # get crop for each active tracklet
                    crops,new_boxes,box_scales = self.crop_tracklets(pre_loc,frame)
                    box_ids = pre_ids
                    
                # black out ignored for source only regions
                for region in self.cam["ignore_for_sources"]:
                    r =  torch.normal(0.485,0.229,[int(region[3])-int(region[1]),int(region[2])-int(region[0])],device = self.device)
                    g =  torch.normal(0.456,0.224,[int(region[3])-int(region[1]),int(region[2])-int(region[0])],device = self.device)
                    b =  torch.normal(0.406,0.225,[int(region[3])-int(region[1]),int(region[2])-int(region[0])],device = self.device)
                    rgb = torch.stack([r,g,b])
                    frame[:,region[1]:region[3],region[0]:region[2]] = rgb
                
                
                self.time_metrics["blackout"] += time.time() - start
                start = time.time()
                
                # also get crops for each source - dont expand these
                source_crops,source_new_boxes,source_box_scales = self.crop_tracklets(self.sources,frame,ber = 1.0)
                n_sources = len(source_crops)

                # concatenate
                if len(pre_locations) > 0:
                    crops = torch.cat((crops,source_crops),0)
                    new_boxes = np.concatenate((new_boxes,source_new_boxes),0)
                    box_scales = np.concatenate((box_scales,source_box_scales),0)
                else:
                    crops = source_crops
                    new_boxes = source_new_boxes
                    box_scales = source_box_scales
                
                self.time_metrics['pre_localize and align'] += time.time() - start
                
                # localize objects using localizer
                start= time.time()
                with torch.no_grad():                       
                     reg_boxes, classes = self.localizer(crops,LOCALIZE = True)
                del crops
                
                if self.PLOT:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                self.time_metrics['localize'] += time.time() - start

                start = time.time()
                reg_boxes = self.local_to_global(reg_boxes,box_scales,new_boxes)
                
                # parse retinanet bboxes
                confs,classes = torch.max(classes, dim = 2)
                
                # strip off source outputs from rest
                source_confs   =  confs[-n_sources:]
                source_classes = classes[-n_sources:]
                source_boxes   = reg_boxes[-n_sources:]
                
                confs          =  confs[:-n_sources]
                classes        = classes[:-n_sources]
                reg_boxes      = reg_boxes[:-n_sources]
                
                # use original bboxes to weight best bboxes 
                if len(pre_locations) > 0:
                    n_anchors = reg_boxes.shape[1]
                    a_priori = torch.from_numpy(pre_loc[:,:4]).to(self.device)
                    # bs = torch.from_numpy(box_scales).unsqueeze(1).repeat(1,4)
                    # a_priori = a_priori * 224/bs
                    a_priori = a_priori.unsqueeze(1).repeat(1,n_anchors,1)
                    
                    iou_score = self.md_iou(a_priori.double(),reg_boxes.double())
                    score = self.W*confs + iou_score
                    best_scores ,keep = torch.max(score,dim = 1)
                    
                    idx = torch.arange(reg_boxes.shape[0])
                    detections = reg_boxes[idx,keep,:].data.cpu()
                    cls_preds = classes[idx,keep].data.cpu()
                    confs = confs[idx,keep]
                    ious = iou_score[idx,keep]
                    
                self.time_metrics["post_localize"] += time.time() -start
                start = time.time()
                
                
                # 8a. increment fsld
                if len(pre_locations) > 0:
                   
                    for i in range(len(detections)):
                        if confs[i] < self.conf_loc or ious[i] < self.iou_loc:
                            self.fsld[box_ids[i]] += 1
                            
                        else:
                            self.fsld[box_ids[i]] = 0
                            self.all_confs[box_ids[i]].append(confs[i])
                            
                            # after some time, obj
                            if frame_num - self.all_first_frames[box_ids[i]] > self.fsld_max +2:
                                self.fsld[box_ids[i]] = -100
                            
                    # store class predictions
                    for i in range(len(cls_preds)):
                        self.all_classes[box_ids[i]][cls_preds[i].item()] += 1
                        
                    
                    # map regressed bboxes directly to objects for update step
                    self.filter.update(detections,box_ids)
                
                # remove stale objects
                if self.fsld_max != -1:
                    removals = []
                    for id in pre_ids:
                        if self.fsld[id] >= self.fsld_max:
                            removals.append(id)
                            self.fsld.pop(id,None) # remove key from fsld
                    if len(removals) > 0:
                        self.filter.remove(removals)  
        
        
                # add new objects from sources (also store source)
                self.parse_sources(source_confs,source_classes,source_boxes,frame_num)        
        
        
                # remove anomalies
                self.remove_anomalies()
                self.remove_overlaps()
                
                # remove objects that overlap sufficiently with sinks, and add to list of counted movements
                self.remove_sinks(frame_num)
                
                self.time_metrics['update'] += time.time() - start

                
                  
                
            # get all object locations and store in output dict
            start = time.time()
            try:
                post_locations = self.filter.objs()
            except:
                post_locations = {}
            self.n_objs.append(len(post_locations))
            for id in post_locations:
                try:
                   self.all_tracks[id][frame_num,:] = post_locations[id][:self.state_size]   
                except IndexError:
                    print("Index Error")
            self.time_metrics['store'] += time.time() - start  
            
            
            # 10. Plot
            start = time.time()
            if self.PLOT:
                self.plot(original_im,[],post_locations,self.all_classes,self.class_dict,title = frame_num)
               
            
            # print speed            
            fps = round(frame_num/(time.time() - self.start_time),2)
            fps_noload = round(frame_num/(time.time()-self.start_time-self.time_metrics["load"] - self.time_metrics["plot"]),2)
            print("\rTracking frame {} of {}. {} FPS ({} FPS without loading)".format(frame_num,self.n_frames,fps,fps_noload), end = '\r', flush = True)
            self.time_metrics['plot'] += time.time() - start

            # load next frame  
            start = time.time()
            [frame_num, frame,dim,original_im] = next(self.loader) 
            if self.PLOT:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            self.time_metrics["load"] += time.time() - start

        # clean up at the end
        print("Total sequence time : {}s".format(np.round((time.time() - self.start_time),2)))
        
        # computed_time = 0
        # for key in self.time_metrics:
        #     print(key,self.time_metrics[key])
        #     computed_time += self.time_metrics[key]
        # print("Computed total time: {}".format(computed_time))
        
        self.end_time = time.time()
        cv2.destroyAllWindows()
        
        total_time = self.end_time - self.start_time
        torch.cuda.empty_cache()
        
        n_objs = sum(self.n_objs)/len(self.n_objs)
        
        return total_time,n_objs
        