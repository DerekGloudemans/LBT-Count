import argparse
import os,sys
import random 
import time
import _pickle as pickle
random.seed = 0
import csv
import torch

# imports from this repo
detector_path = os.path.join(os.getcwd(),"model","py_ret_det_multigpu")
sys.path.insert(0,detector_path)
from model.py_ret_det_multigpu.retinanet.model import resnet50 
from lbt_count import LBT_Count
from lbt_count_draw import LBT_Count_Draw

def sort_human(l):
    cam_nums = [str(item.split("/cam")[-1].split("_")[1].split(".")[0]).zfill(2) for item in l]
    decorators = [item.split("_")[-1].split(".mp4")[0] for item in l]
    decorators = [item if len(item) > 2 else "" for item in decorators ]
    keys = [cam_nums[i] + decorators[i] for i in range (len(l))]
    zipped = zip(keys, l)
    zipped = sorted(zipped)
    l = [item[1] for item in zipped]
    return l

def load_annotations(annotation_file):
    """
    annotation_file - (string) path to .csv annotations of source,
    sink and ignored regions for each camera field of view
    
    loads csv into a dict object of relevant camera FOV regions
    """
    with open(annotation_file,"r") as f:
        csv_reader = csv.reader(f, delimiter=',')
        item_num = 1
        
        car_sources = []
        truck_sources = []
        ignores = []
        ignore_for_sources = []
        
        last_source_typ = None
        for item in csv_reader: # x y x y cls x_dir y dir magnitude
            item = [float(a) if len(a) > 0 else None for a in item]
            
            minx = max(min(item[0],item[2]),0)
            maxx = max(item[0],item[2])
            miny = max(min(item[1],item[3]),0)
            maxy = max(item[1],item[3])
            typ = item[4]
            
            try:
                movement = int(item[5])
            except:
                pass
            
            if typ in [0,1]: # source
                # create xyxy speed init
                speed = torch.tensor([item[5]*item[7],item[6]*item[7],item[5]*item[7],item[6]*item[7]])
                item = torch.tensor([minx,miny,maxx,maxy,item_num])
                item = torch.cat((item,speed),0)
                if typ == 0:
                    car_sources.append([item,[]])
                    last_source_typ = 0
                elif typ == 1:
                    truck_sources.append([item,[]])
                    last_source_typ = 1
            elif typ == 2:
                try:
                    item = torch.tensor([minx,miny,maxx,maxy,item_num,movement,item[6]])
                except:
                    item = torch.tensor([minx,miny,maxx,maxy,item_num,movement,-1])
                if last_source_typ == 0:
                    car_sources[-1][1].append(item)
                elif last_source_typ == 1:
                    truck_sources[-1][1].append(item)
            elif typ == 3:
                item = torch.tensor([minx,miny,maxx,maxy]).int()
                ignores.append(item)
            elif typ == 4:
                item = torch.tensor([minx,miny,maxx,maxy]).int()
                ignore_for_sources.append(item)
                
            item_num += 1
            
        cam_annotations = {"car"  :car_sources,
                           "truck":truck_sources,
                           "ignore":ignores,
                           "ignore_for_sources":ignore_for_sources
                           }
        
        return cam_annotations

if __name__ == "__main__":
    
     try:
        default_dir = "/home/worklab/Data/cv/AIC21_Track1_Vehicle_Counting_full/AIC21_Track1_Vehicle_Counting/Dataset_A"
        parser = argparse.ArgumentParser()
        parser.add_argument("-directory",help = "Should contain video files cam_1.mp4,... etc.",default = default_dir)
        parser.add_argument("-config_directory",help = "Should contain cam_1.config, etc. one config per camera",default = default_config)
        parser.add_argument("-gpu",help = "gpu idx from 0-3", type = int,default = 0)
        parser.add_argument("--show",action = "store_true")
        parser.add_argument("-range",type = str, default = "1-100")
        parser.add_argument("-v",action = "store_true")
        parser.add_argument("-draw",action = "store_true")

        args = parser.parse_args()
        input_dir = args.directory
        
        GPU_ID = args.gpu
        SHOW = args.show
        low = int(args.range.split("-")[0])
        high = int(args.range.split("-")[1])
        subset = [i for i in range(low,high)]
        VERBOSE = args.v
        DRAW = args.draw
        
     except:
         config_dir = "./counting_config"
         input_dir = "/home/worklab/Data/cv/AIC21_Track1_Vehicle_Counting_full/AIC21_Track1_Vehicle_Counting/Dataset_A"
         GPU_ID = 0
         SHOW = True
         subset = [i for i in range(1,100)]
         VERBOSE = True
         DRAW = False
      
     # classes for localizer
     class_dict = { "car":0,
                    "truck":1,
                    "motorcycle":2,
                    "trailer":3,
                    "other":4,
                    0:"car",
                    1:"truck",
                    2:"motorcycle",
                    3:"trailer",
                    4:"other"
                    }
     
     # parameter configs
     config_dir = "./counting_config"
 
     # get localizer
     loc_cp = "./_config/localizer_retrain_112.pt"
     localizer = resnet50(num_classes=5,device_id = GPU_ID)
     cp = torch.load(loc_cp)
     localizer.load_state_dict(cp) 
           
     # get filter
     filter_state_path = "./_config/filter_params_tuned_112.cpkl"
     with open(filter_state_path ,"rb") as f:
             kf_params = pickle.load(f)
              
     # get and sort .mp4 files
     sequences = [os.path.join(input_dir,item) for item in os.listdir(input_dir)]
     videos = []
     for item in sequences:
         if ".mp4" in item:
             videos.append(sequences)
     sequences = videos
     sequences = sort_human(sequences)

     # override sequences
#     sequences = ['/home/worklab/Data/cv/AIC21_Track1_Vehicle_Counting_full/AIC21_Track1_Vehicle_Counting/Dataset_A/cam_1.mp4',
#                     '/home/worklab/Data/cv/AIC21_Track1_Vehicle_Counting_full/AIC21_Track1_Vehicle_Counting/Dataset_A/cam_1_dawn.mp4',
#                     '/home/worklab/Data/cv/AIC21_Track1_Vehicle_Counting_full/AIC21_Track1_Vehicle_Counting/Dataset_A/cam_1_rain.mp4',
#                     '/home/worklab/Data/cv/AIC21_Track1_Vehicle_Counting_full/AIC21_Track1_Vehicle_Counting/Dataset_A/cam_2.mp4',
#                     '/home/worklab/Data/cv/AIC21_Track1_Vehicle_Counting_full/AIC21_Track1_Vehicle_Counting/Dataset_A/cam_2_rain.mp4',
#                     '/home/worklab/Data/cv/AIC21_Track1_Vehicle_Counting_full/AIC21_Track1_Vehicle_Counting/Dataset_A/cam_3.mp4',
#                     '/home/worklab/Data/cv/AIC21_Track1_Vehicle_Counting_full/AIC21_Track1_Vehicle_Counting/Dataset_A/cam_3_rain.mp4',
#                     '/home/worklab/Data/cv/AIC21_Track1_Vehicle_Counting_full/AIC21_Track1_Vehicle_Counting/Dataset_A/cam_4.mp4',
#                     '/home/worklab/Data/cv/AIC21_Track1_Vehicle_Counting_full/AIC21_Track1_Vehicle_Counting/Dataset_A/cam_4_dawn.mp4',
#                     '/home/worklab/Data/cv/AIC21_Track1_Vehicle_Counting_full/AIC21_Track1_Vehicle_Counting/Dataset_A/cam_4_rain.mp4',
#                     '/home/worklab/Data/cv/AIC21_Track1_Vehicle_Counting_full/AIC21_Track1_Vehicle_Counting/Dataset_A/cam_5.mp4',
#                     '/home/worklab/Data/cv/AIC21_Track1_Vehicle_Counting_full/AIC21_Track1_Vehicle_Counting/Dataset_A/cam_5_dawn.mp4',
#                     '/home/worklab/Data/cv/AIC21_Track1_Vehicle_Counting_full/AIC21_Track1_Vehicle_Counting/Dataset_A/cam_5_rain.mp4',
#                     '/home/worklab/Data/cv/AIC21_Track1_Vehicle_Counting_full/AIC21_Track1_Vehicle_Counting/Dataset_A/cam_6.mp4',
#                     '/home/worklab/Data/cv/AIC21_Track1_Vehicle_Counting_full/AIC21_Track1_Vehicle_Counting/Dataset_A/cam_6_snow.mp4',
#                     '/home/worklab/Data/cv/AIC21_Track1_Vehicle_Counting_full/AIC21_Track1_Vehicle_Counting/Dataset_A/cam_7.mp4',
#                     '/home/worklab/Data/cv/AIC21_Track1_Vehicle_Counting_full/AIC21_Track1_Vehicle_Counting/Dataset_A/cam_7_dawn.mp4',
#                     '/home/worklab/Data/cv/AIC21_Track1_Vehicle_Counting_full/AIC21_Track1_Vehicle_Counting/Dataset_A/cam_7_rain.mp4',
#                     '/home/worklab/Data/cv/AIC21_Track1_Vehicle_Counting_full/AIC21_Track1_Vehicle_Counting/Dataset_A/cam_8.mp4',
#                     '/home/worklab/Data/cv/AIC21_Track1_Vehicle_Counting_full/AIC21_Track1_Vehicle_Counting/Dataset_A/cam_9.mp4',
#                     '/home/worklab/Data/cv/AIC21_Track1_Vehicle_Counting_full/AIC21_Track1_Vehicle_Counting/Dataset_A/cam_10.mp4',
#                     '/home/worklab/Data/cv/AIC21_Track1_Vehicle_Counting_full/AIC21_Track1_Vehicle_Counting/Dataset_A/cam_11.mp4',
#                     '/home/worklab/Data/cv/AIC21_Track1_Vehicle_Counting_full/AIC21_Track1_Vehicle_Counting/Dataset_A/cam_12.mp4',
#                     '/home/worklab/Data/cv/AIC21_Track1_Vehicle_Counting_full/AIC21_Track1_Vehicle_Counting/Dataset_A/cam_13.mp4',
#                     '/home/worklab/Data/cv/AIC21_Track1_Vehicle_Counting_full/AIC21_Track1_Vehicle_Counting/Dataset_A/cam_14.mp4',
#                     '/home/worklab/Data/cv/AIC21_Track1_Vehicle_Counting_full/AIC21_Track1_Vehicle_Counting/Dataset_A/cam_15.mp4',
#                     '/home/worklab/Data/cv/AIC21_Track1_Vehicle_Counting_full/AIC21_Track1_Vehicle_Counting/Dataset_A/cam_16.mp4',
#                     '/home/worklab/Data/cv/AIC21_Track1_Vehicle_Counting_full/AIC21_Track1_Vehicle_Counting/Dataset_A/cam_17.mp4',
#                     '/home/worklab/Data/cv/AIC21_Track1_Vehicle_Counting_full/AIC21_Track1_Vehicle_Counting/Dataset_A/cam_18.mp4',
#                     '/home/worklab/Data/cv/AIC21_Track1_Vehicle_Counting_full/AIC21_Track1_Vehicle_Counting/Dataset_A/cam_19.mp4',
#                     '/home/worklab/Data/cv/AIC21_Track1_Vehicle_Counting_full/AIC21_Track1_Vehicle_Counting/Dataset_A/cam_20.mp4' ]
     
       
     time_taken = 0
     for video_id,sequence in enumerate(sequences):
         
         # skip all non-videos
         if video_id + 1 not in subset:
             continue
         if ".mp4" not in sequence: 
             continue
         
         if VERBOSE: print("On sequence {}".format(sequence.split("/")[-1]))
         
         #get camera number and movement conversions 
         cam_num = sequence.split("/cam_")[-1][:2]
         try:
             cam_num = int(cam_num)
         except:
             cam_num = int(cam_num[0])
         cam_num = str(cam_num).zfill(2)
                              
         # get config path
         config = os.path.join(config_dir,"cam{}.config".format(cam_num))
         if not os.path.exists(config):
             config = os.path.join(config_dir,"default.config".format(cam_num))
         
         # parse camera annotations
         annotation_file = "./annotations/new/cam_{}.csv".format(cam_num)
         cam_annotations = load_annotations(annotation_file)
         
         if not DRAW:
             # count 
             tracker = LBT_Count( sequence,
                                  video_id+1,
                                  localizer,
                                  kf_params,
                                  config,
                                  cam_annotations,
                                  class_dict,
                                  VERBOSE = VERBOSE,
                                  PLOT = SHOW,
                                  device_id = GPU_ID)
             time,n_objs = tracker.track()
             time_taken += time
         else:
             tracker = LBT_Count_Draw( sequence,
                                  video_id+1,
                                  localizer,
                                  kf_params,
                                  config,
                                  cam_annotations,
                                  class_dict,
                                  VERBOSE = VERBOSE,
                                  PLOT = SHOW,
                                  device_id = GPU_ID)
             time = tracker.track()
             n_objs = "[unknown]"
             time_taken += time
         
         if VERBOSE: print("Took {}s. Estimated {} objs per frame".format(time,n_objs))
         #tracker.plot_movements()
     
     if VERBOSE: print("\r\nTotal time for all sequences: {}s".format(time_taken),flush = True)
