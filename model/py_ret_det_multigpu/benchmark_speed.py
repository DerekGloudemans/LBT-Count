#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 12:30:54 2020

@author: worklab
"""

from retinanet.model import resnet50
import time
import torch



device = torch.device("cuda:0")

detector = resnet50(13,pretrained = True)
detector = detector.to(device)
detector.eval()
detector.freeze_bn()

transfer_times = []
detect_times = []
batch_sizes = [1,2,3,5,7,10,12,16,20,24,30,40,50,60,75,90,100]

for b in batch_sizes:
    transfer_time = 0
    detect_time = 0
    for i in range(0,1000):
        data = torch.randn([b,3,960,540])
        #data = torch.randn([b,3,2000,1000])
        start = time.time()
        data = data.to(device)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        transfer_time += elapsed
        
        with torch.no_grad():
            start = time.time()
            detector(data)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            detect_time += elapsed
    
    print("Finished b = {}".format(b))
    transfer_times.append(transfer_time)
    detect_times.append(detect_time)
    
    # save = {}
    # save["transfer_times"] = transfer_times
    # save["detect_times"] = detect_times
    # save["batch_sizes"]    = batch_sizes
    # with open("detect_1080.cpkl","wb") as f:
    #     pickle.dump(save,f)