"""
Derek Gloudemans - August 4, 2020
This file contains a simple script to train a retinanet object localizer on the UA Detrac
detection dataset.
- Pytorch framework
- Resnet-50 Backbone
- Manual file separation of training and validation data
- Automatic periodic checkpointing
"""

### Imports

import os
import sys
import numpy as np
import random 

import cv2
random.seed = 0

import torch
from torch import nn, optim
from torch.utils import data
import collections

import argparse

# add relevant packages and directories to path
localizer_path = os.path.join(os.getcwd(),"model","pytorch_retinanet_detector")
sys.path.insert(0,localizer_path)
from model.pytorch_retinanet_detector.retinanet import model

detrac_util_path = os.path.join(os.getcwd(),"util_detrac")
sys.path.insert(0,detrac_util_path)
from util.ai_multi_localization_dataset import LocMulti_Dataset, class_dict,collate


# surpress XML warnings
import warnings
warnings.filterwarnings(action='once')

def plot_detections(dataset,retinanet):
    """
    Plots detections output
    """
    retinanet.training = False
    retinanet.eval()

    idx = np.random.randint(0,len(dataset))

    im,gt,meta = dataset[idx]

    im = im.to(device).unsqueeze(0).float()
    #im = im[:,:,:224,:224]


    with torch.no_grad():

        scores,labels, boxes = retinanet(im)

    if len(boxes) > 0:
        keep = []    
        for i in range(len(scores)):
            if scores[i] > 0.5:
                keep.append(i)
        boxes = boxes[keep,:]

    im = dataset.denorm(im[0])
    cv_im = np.array(im.cpu()) 
    cv_im = np.clip(cv_im, 0, 1)

    # Convert RGB to BGR 
    cv_im = cv_im[::-1, :, :]  

    im = cv_im.transpose((1,2,0))

    for box in boxes:
        box = box.int()
        im = cv2.rectangle(im,(box[0],box[1]),(box[2],box[3]),(0.7,0.3,0.2),1)
    
    for box in gt:
        box = box.int()
        im = cv2.rectangle(im,(box[0],box[1]),(box[2],box[3]),(0.0,0.3,0.2),1)
    cv2.imshow("Frame",im)
    cv2.waitKey(2000)

    retinanet.train()
    retinanet.training = True
    retinanet.module.freeze_bn()

def plot_anchors(dataset,retinanet):
    """
    Plots detections output
    """
    retinanet.training = False
    retinanet.eval()

    idx = np.random.randint(0,len(dataset))

    im,label,meta = dataset[idx]

    im = im.to(device).unsqueeze(0).float()
    #im = im[:,:,:224,:224]


    with torch.no_grad():
        anchors = retinanet(im)
    #     scores,labels, boxes = retinanet(im)

    # if len(boxes) > 0:
    #     keep = []    
    #     for i in range(len(scores)):
    #         if scores[i] > 0.5:
    #             keep.append(i)
    #     boxes = boxes[keep,:]

    im = dataset.denorm(im[0])
    cv_im = np.array(im.cpu()) 
    cv_im = np.clip(cv_im, 0, 1)

    # Convert RGB to BGR 
    cv_im = cv_im[::-1, :, :]  

    im = cv_im.transpose((1,2,0))
    count = 0
    for box in anchors[0]:
    #for box in boxes:
        if count % 211 == 0:
            box = box.int()
            im = cv2.rectangle(im,(box[0],box[1]),(box[2],box[3]),(0.7,0.3,0.2),1)
            cv2.imshow("Frame",im)
            cv2.waitKey(200)
        count += 1

    retinanet.train()
    retinanet.training = True
    retinanet.module.freeze_bn()

def eval_iou(dataset,retinanet):
    """
    Evaluates localizer output performance
    """
    retinanet.training = False
    retinanet.eval()

    all_ious = []
    for j in range(100):
        idx = np.random.randint(0,len(dataset))
    
        im,label,meta = dataset[idx]
    
        im = im.to(device).unsqueeze(0).float()
    
    
        with torch.no_grad():
            scores,labels, boxes = retinanet(im)
            scores = scores.cpu()
            labels = labels.cpu()
            boxes = boxes.cpu()
    
        if len(boxes) > 0:
            keep = []    
            for i in range(len(scores)):
                if scores[i] > 0.5:
                    keep.append(i)
            boxes = boxes[keep,:]
    
        for item in label:
            iou_max = 0
            for box in boxes:
                iou_score = iou(box.double(),item.double())
                if iou_score > iou_max:
                    iou_max = iou_score
            all_ious.append(iou_max)
    
    score = sum(all_ious) / len(all_ious)
    print("Average iou across {} examples for all objects: {}".format(j+1,score))
    
    retinanet.train()
    retinanet.training = True
    retinanet.module.freeze_bn()

def iou(a,b):
    """
    Description
    -----------
    Calculates intersection over union for all sets of boxes in a and b

    Parameters
    ----------
    a : a torch of size [batch_size,4] of bounding boxes in xyxy formulation
    b : a torch of size [batch_size,4] of bounding boxes in xyxy formulation

    Returns
    -------
    mean_iou - float between [0,1] with average iou for a and b
    """
    a = a.unsqueeze(0)
    b = b.unsqueeze(0)
    area_a = (a[:,2]-a[:,0]) * (a[:,3] - a[:,1])
    area_b = (b[:,2]-b[:,0]) * (b[:,3] - b[:,1])
    
    minx = torch.max(a[:,0], b[:,0])
    maxx = torch.min(a[:,2], b[:,2])
    miny = torch.max(a[:,1], b[:,1])
    maxy = torch.min(a[:,3], b[:,3])
    zeros = torch.zeros(minx.shape,dtype=float)
    
    intersection = torch.max(zeros, maxx-minx) * torch.max(zeros,maxy-miny)
    union = area_a + area_b - intersection
    iou = torch.div(intersection,union)
    mean_iou = torch.mean(iou)
    
    return mean_iou
    

def to_cpu(checkpoint):
    """
    """
    try:
        retinanet = model.resnet50(5)
        retinanet = nn.DataParallel(retinanet,device_ids = [0,1,2])
        retinanet.load_state_dict(torch.load(checkpoint))
    except:
        retinanet = model.resnet34(5)
        retinanet = nn.DataParallel(retinanet,device_ids = [0,1,2])
        retinanet.load_state_dict(torch.load(checkpoint))
        
    retinanet = nn.DataParallel(retinanet, device_ids = [0])
    retinanet = retinanet.cpu()
    
    new_state_dict = {}
    for key in retinanet.state_dict():
        new_state_dict[key.split("module.")[-1]] = retinanet.state_dict()[key]
        
    torch.save(new_state_dict, "cpu_{}".format(checkpoint))
    print ("Successfully created: cpu_{}".format(checkpoint))


if __name__ == "__main__":
    
    try:
        parser = argparse.ArgumentParser()
        
        parser.add_argument("detrac_data_directory",help = "Path to main UA DETRAC data directory")
        parser.add_argument("-gpus",help = "comma separated list (e.g. 0,1)", type = str,default = "0")

        args = parser.parse_args()
        detrac_path = args.detrac_data_directory
        gpus = [int(item) for item in args.gpus.split(",")] 
        
        detrac_image_dir = detrac_path + "/DETRAC-train-data"
        detrac_label_dir = detrac_path + "/DETRAC-Train-Annotations-XML-v3"
        
    except:
        detrac_image_dir = "/home/worklab/Data/cv/Detrac/DETRAC-train-data"
        detrac_label_dir = "/home/worklab/Data/cv/Detrac/DETRAC-Train-Annotations-XML-v3"
        gpus = [0]
    
    print("Starting localizer training with GPUs: {}".format(gpus))
    
    # define parameters
    depth = 50
    num_classes = 5
    
    patience = 0
    max_epochs = 50
    start_epoch = 0
    checkpoint_file = None

    # Paths to auxiliary data (if used) here
    i24_label_dir = None #"/home/worklab/Data/cv/i24_2D_October_2020/labels.csv"
    i24_image_dir = None #"/home/worklab/Data/cv/i24_2D_October_2020/ims"
    
    ###########################################################################


    # Create the model
    if depth == 18:
        retinanet = model.resnet18(num_classes=num_classes, pretrained=True)
    elif depth == 34:
        retinanet = model.resnet34(num_classes=num_classes, pretrained=True)
    elif depth == 50:
        retinanet = model.resnet50(num_classes=num_classes, pretrained=True)
    elif depth == 101:
        retinanet = model.resnet101(num_classes=num_classes, pretrained=True)
    elif depth == 152:
        retinanet = model.resnet152(num_classes=num_classes, pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    # create dataloaders
    try:
        train_data
    except:
        # datasets here defined for UA Detrac Dataset
        print("Loading data..")
        train_data = LocMulti_Dataset(detrac_image_dir,detrac_label_dir,i24_image_dir,i24_label_dir,cs = 112,mode = "train")
        val_data = LocMulti_Dataset(detrac_image_dir,detrac_label_dir,i24_image_dir,i24_label_dir,cs = 112, mode = "val")
        
        params = {'batch_size' : 32,
              'shuffle'    : True,
              'num_workers': 0,
              'drop_last'  : True,
              'collate_fn' : collate
              }
        trainloader = data.DataLoader(train_data,**params)
        testloader = data.DataLoader(val_data,**params)

   

    # CUDA
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        if torch.cuda.device_count() > 1:
            retinanet = torch.nn.DataParallel(retinanet,device_ids = gpus)
            retinanet = retinanet.to(device)
        else:
            retinanet = retinanet.to(device)

     # load checkpoint
    try:
        if checkpoint_file is not None:
            retinanet.load_state_dict(torch.load(checkpoint_file))
    except:
        retinanet.load_state_dict(torch.load(checkpoint_file))
        
    # training mode
    retinanet.train()
    retinanet.module.freeze_bn()
    retinanet.training = True
    
    # define optimizer and lr scheduler
    optimizer = optim.Adam(retinanet.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, verbose=True, mode = "min")
    
    loss_hist = collections.deque(maxlen=500)
    most_recent_mAP = 0

    print('Num training images: {}'.format(len(train_data)))

    eval_iou(val_data,retinanet)
    # main training loop
    for epoch_num in range(start_epoch,max_epochs):


        print("Starting epoch {}".format(epoch_num))
        retinanet.train()
        retinanet.module.freeze_bn()
        epoch_loss = []

        for iter_num, (im,label,ignore) in enumerate(trainloader):
            
            retinanet.train()
            retinanet.training = True
            retinanet.module.freeze_bn()    
            
            try:
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    classification_loss, regression_loss = retinanet([im.to(device).float(), label.to(device).float()])
                else:
                    classification_loss, regression_loss = retinanet([im.float(),label.float()])

                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                if iter_num % 2 == 0:
                    print(
                        'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                            epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))
                if iter_num % 10 == 0:
                    plot_detections(val_data, retinanet)
                
                if iter_num % 1000 == 0:
                    PATH = "aicity_localizer112_retinanet_epoch{}_{}.pt".format(epoch_num,iter_num)
                    torch.save(retinanet.state_dict(), PATH)

                
                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue

        print("Epoch {} training complete".format(epoch_num))
        

        scheduler.step(np.mean(epoch_loss))
        torch.cuda.empty_cache()
        #save checkpoint at the end of each epoch
        PATH = "aicity_localizer112_retinanet_epoch{}.pt".format(epoch_num)
        torch.save(retinanet.state_dict(), PATH)


    retinanet.eval()

