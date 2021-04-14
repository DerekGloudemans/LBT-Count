# LBT-count

Vehicle Turning-Movement Counting using Localization-based Tracking is a fast algorithm for performing counts at intersections and along roadways. The core innovation of this algorithm is to crop existing objects from a frame and only pass these regions through a CNN for localization. A few source regions are also processed by the CNN at each frame to initialize new objects. 

This repository contains the following:
- _fov_config/ - .config files containing parameter settings for each unique camera field of view
- _fov_annotations/ - csv files containing relevant regions (sources, sinks, and ignored regions) for each unique camera field of view
- _config/ - other config files (Kalman filter parameters and localizer state dict)
- _readme_ims/ - example images used in this repository
- _output/ - default output directory for any files generated when code runs
- util/ - implementations for FrameLoader, Kalman Filter, video output writer, etc.
- model/ - [Pytorch implementation of Retinanet](https://github.com/yhenon/pytorch-retinanet), modified for multi-gpu running
- lbt_count.py - contains code for Vehicle Turning-Movement Counting using Localization-based Tracking
- lbt_count_draw.py - modified version of lbt_count.py for plotting vehicle turning movements
- annotate_frame.py - cv2-based GUI for annotating camera fields of view
- count_all.py - main file, runs lbt_count on all video sequences within an input directory
- 
