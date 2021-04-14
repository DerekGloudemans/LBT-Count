#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 17:29:07 2020

@author: worklab
"""


# 4 x Quadro RTX 6000 machine
data_paths = {
    "train_im":"/home/worklab/Data/cv/Detrac/DETRAC-train-data",
    "train_lab":"/home/worklab/Data/cv/Detrac/DETRAC-Train-Annotations-XML-v3",
    "test_im":"/home/worklab/Data/cv/Detrac/DETRAC-test-data",
    "test_lab":"/home/worklab/Data/cv/Detrac/DETRAC-Test-Annotations-XML",
    "train_partition":"/home/worklab/Data/cv/Detrac/detrac_train_partition",
    "val_partition":"/home/worklab/Data/cv/Detrac/detrac_val_partition"
    
 }

directories = ["/home/worklab/Documents/derek/tracking-by-localization/config",
               "/home/worklab/Documents/derek/tracking-by-localization/data/detrac_detections",
               "/home/worklab/Documents/derek/tracking-by-localization/_data_utils",
               "/home/worklab/Documents/derek/tracking-by-localization/_detectors",
               "/home/worklab/Documents/derek/tracking-by-localization/_eval",
               "/home/worklab/Documents/derek/tracking-by-localization/_localizers",
               "/home/worklab/Documents/derek/tracking-by-localization/_train",
               "/home/worklab/Documents/derek/tracking-by-localization/_tracker",
               
    ]

