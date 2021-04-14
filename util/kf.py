#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 10:49:38 2020
@author: worklab
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import time


class Torch_KF(object):
    """
    A tensor-based Kalman Filter that evaluates many KF-tracked objects in parallel
    using tensor operations
    """
    
    def __init__(self,device,state_err = 10000, meas_err = 1, mod_err = 1, INIT = None, ADD_MEAN_Q = False, ADD_MEAN_R = False):
        """
        Parameters
        ----------
        device : torch.device
            Specifies whether tensors should be stored on GPU or CPU.
        state_err : float, optional
            Specifies the starting state covariance value along the main diagonal. The default is 1.
        meas_err : float, optional
            Specifies the measurement covariance value along the main diagonal. The default is 1.
        mod_err : float, optional
            specifies the model covariance value along the main diagonal. The default is 1.
        INIT : dictionary, optional
            A dictionary containing initialization matrices for P0, H, mu_H, Q, and mu_Q. 
            If specified, these are used instead of the diagonal values
        """
        # initialize tensors
        self.meas_size = 4
        self.state_size = 7
        self.t = 1/15.0
        self.device = device
        self.X = None
        
        self.P0 = torch.zeros(self.state_size,self.state_size) # state covariance
        self.F = torch.zeros(self.state_size,self.state_size) # dynamical model
        self.H = torch.zeros(self.meas_size,self.state_size)  # measurement model
        self.Q = torch.zeros(self.state_size,self.state_size) # model covariance
        self.R = torch.zeros(self.meas_size,self.meas_size)   # measurement covariance
        self.R2 = torch.zeros(self.meas_size,self.meas_size)   # second measurement covariance
        
        # obj_ids[a] stores index in X along dim 0 where object a is stored
        self.obj_idxs = {}
        
        if INIT is None:
            # set intial value for state covariance
            self.P0 = torch.eye(self.state_size).unsqueeze(0) * state_err
            
            # these values won't change 
            self.F    = torch.eye(self.state_size).float()
            self.F[[0,1,2],[4,5,6]] = self.t
            self.H[:4,:4] = torch.eye(4)
            self.Q    = torch.eye(self.state_size).unsqueeze(0) * mod_err                     #+ 1
            self.R    = torch.eye(self.meas_size).unsqueeze(0) * meas_err
            self.R2   = torch.eye(self.meas_size).unsqueeze(0) * meas_err
            self.mu_Q = torch.zeros([1,self.state_size])
            self.mu_R = torch.zeros([1,self.meas_size])

            
        # use INIT matrices to initialize filter    
        else:
            self.P0 = INIT["P"].unsqueeze(0) 
            self.F  = INIT["F"]
            self.H  = INIT["H"]
            self.Q  = INIT["Q"].unsqueeze(0)
            self.R  = INIT["R"].unsqueeze(0) 
            #self.R[0,2,2] *= 100 # increase uncertainty in localizer scale
            self.R2  = INIT["R2"].unsqueeze(0)  
            self.mu_Q = INIT["mu_Q"].unsqueeze(0) 
            self.mu_R = INIT["mu_R"].unsqueeze(0)
            self.mu_R2 = INIT["mu_R2"].unsqueeze(0)

            self.state_size = self.F.shape[0]
            self.meas_size  =  self.H.shape[0]
            
            #overwrite means
            if not ADD_MEAN_Q:
                self.mu_Q  = torch.zeros([1,self.state_size])
            if not ADD_MEAN_R:
                self.mu_R  = torch.zeros([1,self.meas_size])
                self.mu_R2 = torch.zeros([1,self.meas_size])

       
        # move to device
        self.F = self.F.to(device).float()
        self.H = self.H.to(device).float()
        self.Q = self.Q.to(device).float()
        self.R = self.R.to(device).float()
        self.R2 = self.R.to(device).float()
        self.P0 = self.P0.to(device).float() 
        self.mu_Q = self.mu_Q.to(device).float()
        self.mu_R = self.mu_R.to(device).float()
        self.mu_R2 = self.mu_R.to(device).float()
        
    def add(self,detections,obj_ids):
        """
        Description
        -----------
        Initializes self.X if this is the first object, otherwise adds new object to X and P 
        
        Parameters
        ----------
        detection - np array of size [n,4] 
            Specifies bounding box x,y,scale and ratio for each detection
        obj_ids - list of length n
            Unique obj_id (int) for each detection
        """
        
        newX = torch.zeros((len(detections),self.state_size)) 
        if len(detections[0]) == self.meas_size:
            try:
                newX[:,:self.meas_size] = torch.from_numpy(detections).to(self.device)
            except:
                newX[:,:self.meas_size] = detections.to(self.device)
        else: # case where velocity estimates are given
            try:
                newX = torch.from_numpy(detections).to(device)
            except:
                newX = detections.to(self.device)
                
        newP = self.P0.repeat(len(obj_ids),1,1)

        # store state and initialize P with defaults
        try:
            new_idx = len(self.X)
            self.X = torch.cat((self.X,newX), dim = 0)
            self.P = torch.cat((self.P,newP), dim = 0)
        except:
            new_idx = 0
            self.X = newX.to(self.device).float()
            self.P = newP.to(self.device)
            
        # add obj_ids to dictionary
        for id in obj_ids:
            self.obj_idxs[id] = new_idx
            new_idx = new_idx + 1
        
    
    def remove(self,obj_ids):
        """
        Description
        -----------
        Removes objects indexed by integer id so that they are no longer tracked
        
        Parameters
        ----------
        obj_ids : list of (int) object ids
        """
        if self.X is not None:
            keepers = list(range(len(self.X)))
            for id in obj_ids:
                keepers.remove(self.obj_idxs[id])
                self.obj_idxs[id] = None    
            keepers.sort()
            
            self.X = self.X[keepers,:]
            self.P = self.P[keepers,:]
        
            # since rows were deleted from X and P, shift idxs accordingly
            new_id = 0
            for id in self.obj_idxs:
                if self.obj_idxs[id] is not None:
                    self.obj_idxs[id] = new_id
                    new_id += 1
    
    def predict(self):
        """
        Description:
        -----------
        Uses prediction equations to update X and P without a measurement
        """
        
        # update X --> X = XF + mu_F--> [n,7] x [7,7] + [n,7] = [n,7]
        self.X = torch.mm(self.X,self.F.transpose(0,1)) + self.mu_Q
        
        # update P --> P = FPF^(-1) + Q --> [nx7x7] = [nx7x7] bx [nx7x7] bx [nx7x7] + [n+7x7]
        F_rep = self.F.unsqueeze(0).repeat(len(self.P),1,1)
        step1 = torch.bmm(F_rep,self.P)
        step2 = F_rep.transpose(1,2)
        step3 = torch.bmm(step1,step2)
        step4 = self.Q.repeat(len(self.P),1,1)
        self.P = step3 + step4
        
        
    def update(self,detections,obj_ids):
        """
        Description
        -----------
        Updates state for objects corresponding to each obj_id in obj_ids
        Equations taken from: wikipedia.org/wiki/Kalman_filter#Predict
        
        Parameters
        ----------
        detection - np array of size [m,4] 
            Specifies bounding box x,y,scale and ratio for each of m detections
        obj_ids - list of length m
            Unique obj_id (int) for each detection
        """
        
        # get relevant portions of X and P
        relevant = [self.obj_idxs[id] for id in obj_ids]
        X_up = self.X[relevant,:]
        P_up = self.P[relevant,:,:]
        
        # state innovation --> y = z - XHt --> mx4 = mx4 - [mx7] x [4x7]t  
        try:
            z = torch.from_numpy(detections).to(self.device)
        except:
             z = detections.to(self.device)
        y = z + self.mu_R - torch.mm(X_up, self.H.transpose(0,1))  ######### Not sure if this is right but..
        
        # covariance innovation --> HPHt + R --> [mx4x4] = [mx4x7] bx [mx7x7] bx [mx4x7]t + [mx4x4]
        # where bx is batch matrix multiplication broadcast along dim 0
        # in this case, S = [m,4,4]
        H_rep = self.H.unsqueeze(0).repeat(len(P_up),1,1)
        step1 = torch.bmm(H_rep,P_up) # this needs to be batched along dim 0
        step2 = torch.bmm(step1,H_rep.transpose(1,2))
        S = step2 + self.R.repeat(len(P_up),1,1)
        
        # kalman gain --> K = P Ht S^(-1) --> [m,7,4] = [m,7,7] bx [m,7,4]t bx [m,4,4]^-1
        step1 = torch.bmm(P_up,H_rep.transpose(1,2))
        K = torch.bmm(step1,S.inverse())
        
        # A posteriori state estimate --> X_updated = X + Ky --> [mx7] = [mx7] + [mx7x4] bx [mx4x1]
        # must first unsqueeze y to third dimension, then unsqueeze at end
        y = y.unsqueeze(-1).float() # [mx4] --> [mx4x1]
        step1 = torch.bmm(K,y).squeeze(-1) # mx7
        X_up = X_up + step1
        
        # P_updated --> (I-KH)P --> [m,7,7] = ([m,7,7 - [m,7,4] bx [m,4,7]) bx [m,7,7]    
        I = torch.eye(self.state_size).unsqueeze(0).repeat(len(P_up),1,1).to(self.device)
        step1 = I - torch.bmm(K,H_rep)
        P_up = torch.bmm(step1,P_up)
        
        # store updated values
        self.X[relevant,:] = X_up
        self.P[relevant,:,:] = P_up
        
    def update2(self,detections,obj_ids):
        """
        Description
        -----------
        Updates state for objects corresponding to each obj_id in obj_ids
        Equations taken from: wikipedia.org/wiki/Kalman_filter#Predict
        
        Parameters
        ----------
        detection - np array of size [m,4] 
            Specifies bounding box x,y,scale and ratio for each of m detections
        obj_ids - list of length m
            Unique obj_id (int) for each detection
        """
        
        # get relevant portions of X and P
        relevant = [self.obj_idxs[id] for id in obj_ids]
        X_up = self.X[relevant,:]
        P_up = self.P[relevant,:,:]
        
        # state innovation --> y = z - XHt --> mx4 = mx4 - [mx7] x [4x7]t  
        try:
            z = torch.from_numpy(detections).to(self.device)
        except:
             z = detections.to(self.device)
        y = z + self.mu_R2 - torch.mm(X_up, self.H.transpose(0,1))  ######### Not sure if this is right but..
        
        # covariance innovation --> HPHt + R --> [mx4x4] = [mx4x7] bx [mx7x7] bx [mx4x7]t + [mx4x4]
        # where bx is batch matrix multiplication broadcast along dim 0
        # in this case, S = [m,4,4]
        H_rep = self.H.unsqueeze(0).repeat(len(P_up),1,1)
        step1 = torch.bmm(H_rep,P_up) # this needs to be batched along dim 0
        step2 = torch.bmm(step1,H_rep.transpose(1,2))
        S = step2 + self.R2.repeat(len(P_up),1,1)
        
        # kalman gain --> K = P Ht S^(-1) --> [m,7,4] = [m,7,7] bx [m,7,4]t bx [m,4,4]^-1
        step1 = torch.bmm(P_up,H_rep.transpose(1,2))
        K = torch.bmm(step1,S.inverse())
        
        # A posteriori state estimate --> X_updated = X + Ky --> [mx7] = [mx7] + [mx7x4] bx [mx4x1]
        # must first unsqueeze y to third dimension, then unsqueeze at end
        y = y.unsqueeze(-1).float() # [mx4] --> [mx4x1]
        step1 = torch.bmm(K,y).squeeze(-1) # mx7
        X_up = X_up + step1
        
        # P_updated --> (I-KH)P --> [m,7,7] = ([m,7,7 - [m,7,4] bx [m,4,7]) bx [m,7,7]    
        I = torch.eye(self.state_size).unsqueeze(0).repeat(len(P_up),1,1).to(self.device)
        step1 = I - torch.bmm(K,H_rep)
        P_up = torch.bmm(step1,P_up)
        
        # store updated values
        self.X[relevant,:] = X_up
        self.P[relevant,:,:] = P_up   
        
        
    def objs(self):
        """
        Returns
        -------
        out_dict - dictionary
            Current state of each object indexed by obj_id (int)
        """
        
        out_dict = {}
        for id in self.obj_idxs:
            idx = self.obj_idxs[id]
            if idx is not None:
                out_dict[id] = self.X[idx,:].data.cpu().numpy()
        return out_dict        

if __name__ == "__main__":
    """
    A test script in which bounding boxes are randomly generated and jittered to create motion
    """
    
     # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.cuda.empty_cache()   
    
    all_trials = [3,10,30,100,300,1000]
    all_results = {"cuda:0":[],"cpu":[]}
    for device in ["cuda:0","cpu"]:
        for n_objs in all_trials:
        
            #n_objs =1000
            n_frames = 1000
            
            ids = list(range(n_objs))
            detections = np.random.rand(n_objs,4)*50
        
            colors = np.random.rand(n_objs,4)
            colors2 = colors.copy()
            colors[:,3]  = 0.2
            colors2[:,3] = 1
            filter = Torch_KF(device)
            
            start_time = time.time()
            
            filter.add(detections,ids)
            for i in range(0,n_frames):
                
                start = time.time()
                filter.predict()
                
                detections = detections + np.random.normal(0,1,[n_objs,4]) + 1
                detections[:,2:] = detections[:,2:]/50
                remove = np.random.randint(0,n_objs - 1)
                
                ids_r = ids.copy()
                del ids_r[remove]
                det_r = detections[ids_r,:]
                start = time.time()
                filter.update(det_r,ids_r)
                tracked_objects = filter.objs()
        
                if False:
                    # plot the points to visually confirm that it seems to be working 
                    x_coords = []
                    y_coords = []
                    for key in tracked_objects:
                        x_coords.append(tracked_objects[key][0])
                        y_coords.append(tracked_objects[key][1])
                    for i in range(len(x_coords)):
                        if i < len(x_coords) -1:
                            plt.scatter(det_r[i,0],det_r[i,1], color = colors2[i])
                        plt.scatter(x_coords[i],y_coords[i],s = 300,color = colors[i])
                        plt.annotate(i,(x_coords[i],y_coords[i]))
                    plt.draw()
                    plt.pause(0.0001)
                    plt.clf()
                
            total_time = time.time() - start_time
            frame_rate = n_frames/total_time
            all_results[device].append(frame_rate)
            print("Filtering {} objects for {} frames took {} sec. Average frame rate: {} on {}".format(n_objs,n_frames,total_time, n_frames/total_time,device))
            torch.cuda.empty_cache()
            
    plt.figure()   
    plt.plot(all_trials,all_results['cpu'])
    plt.plot(all_trials,all_results['cuda:0'])
    plt.xlabel("Number of filtered objects")
    plt.ylabel("Frame Rate (Hz)")
    plt.legend(["CPU","GPU"])
    plt.title("Frame Rate versus number of objects")