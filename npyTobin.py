from torchvision import datasets, transforms
import torch.utils.data
import torch
import sys
import argparse
import matplotlib.pyplot as plt
from utils_kitti import *
import numpy as np
import os
import open3d as o3d
from open3d import open3d
from tqdm import trange


parser = argparse.ArgumentParser(description='VAE training of LiDAR')
parser.add_argument('--data',       type=str,   default='',  required= True,             help='Location of the orignal LiDAR')
parser.add_argument('--folder',     type=str,   default='',  required= True,             help='Folder to save')
args = parser.parse_args()




def get_pcd_from_img_and_save(img,index,location):
    
    frame = from_polar(img.cuda()).detach().cpu().numpy()

    frame_flat = frame.reshape((3,-1))
    some_pcd = o3d.geometry.PointCloud()
    some_arr = frame_flat.T
    # some_arr = some_arr[[(-40 < x < 40) for x, y, z in some_arr]]
    some_pcd.points = o3d.utility.Vector3dVector(some_arr)

    o3d.io.write_point_cloud(location+str(index)+".pcd", some_pcd)
    del some_pcd,some_arr,frame_flat



def get_bin(img,index,location):
    
    frame = from_polar_np(img.detach().cpu().numpy())
    # print(frame.shape) # 1,3,60,512
    frame_flat = frame.reshape((3,-1))
    # some_pcd = o3d.geometry.PointCloud()
    some_arr = frame_flat.T*120
    print(some_arr.shape)
    # raise SystemError
    file = open(location+str(index)+'.bin', mode='wb')
    x=some_arr.tofile(file)
    # print(x)
    # raise SystemError




if not os.path.exists('pcd/'+args.folder+'/'):
		os.makedirs('pcd/'+args.folder+'/')

index=-1




dataset_original   	  = np.load(args.data)
original_loader    	  = torch.utils.data.DataLoader(dataset_original, batch_size=dataset_original.shape[0], shuffle=False, num_workers=4, drop_last=False)

print('Loaded')

for  i,orig in enumerate(original_loader):
    for ii in trange(orig.shape[0]):
        index+=1    
        # print(orig[ii].shape)  # 2,60,512
        get_bin(orig[ii].reshape(1,2,60,512),index,'pcd/'+args.folder+'/') 