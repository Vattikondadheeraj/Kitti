from torchvision import datasets, transforms
import torch.utils.data
import torch
import sys
import argparse
import matplotlib.pyplot as plt
from utils512 import *
import numpy as np
import os
import open3d as o3d
from open3d import open3d
from tqdm import trange



parser = argparse.ArgumentParser(description='VAE training of LiDAR')
parser.add_argument('--orig',       type=str,   default='',              help='Location of the orignal LiDAR')
parser.add_argument('--pred',       type=str,   default='',              help='Location of the predicted LiDAR')
parser.add_argument('--batch_size', type=int,   default=2048,            help='Batch size')
parser.add_argument('--thresh',     type=float, default=0.05,            help='Threshold for filtering')
parser.add_argument('--folder',     type=str,   default='',               required=True, help='Fodler name for storing pcd inside pcd folder')
parser.add_argument('--no_polar',   type=int,   default=0,                help='if True, the representation used is (X,Y,Z), instead of (D, Z), where D=sqrt(X^2+Y^2)')
parser.add_argument('--debug', action='store_true')
# args = parser.parse_args(args=['atlas_baseline=0, autoencoder=1,panos_baseline=0'])





# Encoder must be trained with all types of frames,dynmaic, static all

args = parser.parse_args()




#Helper Functions
#-----------------------------------------------------------------------------------------

def save_on_disk(folder,filtered_dy_recon):
	np.save('SaveRecons/'+folder+'/filtered',filtered_dy_recon)



def withinRange(x,y):
    if(abs(x-y)<args.thresh):
        return 1
    else:
        return 0



def get_pcd_from_img_and_save(img,index,location):
    
    frame = from_polar(img).detach().cpu().numpy()
    # frame_actual = np.array([frame_image[:29] for frame_image in frame])
    # print(frame.shape)
    #-----------------------------------------------------------------------
    #frame = frame[:,:,:,::2]  #retrun only 256 of the 256 
    #-----------------------------------------------------------------------
    #return only 384 of the 256 beams

    # listt=np.random.choice(frame.shape[3],384,replace=False)
    # listt=np.sort(listt)
    # frame = frame[:,:,:,listt]

    #-----------------------------------------------------------------------

    # print(frame.shape)
    # exit(1)
    frame_flat = frame.reshape((3,-1))
    some_pcd = o3d.geometry.PointCloud()
    some_arr = frame_flat.T * 120
    some_arr = some_arr[[(-40 < x < 40) for x, y, z in some_arr]]
    some_pcd.points = o3d.utility.Vector3dVector(some_arr)

    o3d.io.write_point_cloud(location+str(index)+".pcd", some_pcd)
    del some_pcd,some_arr,frame_flat







def get_pcd_from_non_preprocessed_npy_and_save(img,index):
    
    frame = img.cpu().numpy()     #This is alrasy in shape in (x,y, z) form 
    

    #-----------------------------------------------------------------------

    # save as image
    plt.figure()
    plt.scatter(frame[:,0], frame[:,1], s=0.7, color='k')
    plt.savefig('samplesVid/without_prep/'+str(index)+'.jpg') 

	#-----------------------------------------------------------------------

    frame_flat = frame.reshape((3,-1))
    some_pcd = o3d.PointCloud()
    some_arr = frame_flat.T
    some_pcd.points = o3d.open3d.Vector3dVector(some_arr)
    
    o3d.write_point_cloud('testModel/without_prep/'+str(index)+".pcd", some_pcd)
    del some_pcd,some_arr,frame_flat








def get_pcd_without_preprocess(file_,location):
    frames = np.load(location + file_)
    frames = frames.transpose(0,3,1,2)
    frames = frames[:,:,:,::2]
    frames = frames[:,0:3,:,:]
    for frame_num in range(frames.shape[0]):
    	frame = frames[frame_num:frame_num+1,:,:,:]
    	get_pcd_from_non_preprocessed_npy_and_save(torch.Tensor(frame.reshape([1,3,60,256])).cuda(),str(frame_num))

    	





def save_images(recons_filtered, batch_num,folder):
	i=0
	for frame_num in range(recons_filtered.shape[0]):
		if(i<args.batch_size):
			i+=1
			frame=from_polar(torch.Tensor(recons_filtered[frame_num:frame_num+1,:,:,:]).cuda()).detach().cpu().numpy()
			frame = frame.reshape((3,-1)).T*120
			frame = frame[[(-40 < x < 40) for x, y, z in frame]]
			# frame =frame.reshape([1,3,40,256])
			plt.figure()
			plt.scatter(frame[:,0], frame[:,1], s=0.7, color='k')
			plt.savefig('samplesVid/'+folder+'/'+str(frame_num+batch_num*args.batch_size)+'.jpg') 
		else:
			break



def get_bin(img,index,location):
    
    # frame = (img.detach().cpu().numpy())
    # print(frame.shape) # 1,3,60,512
    frame_flat = img.reshape((3,-1))
    # some_pcd = o3d.geometry.PointCloud()
    some_arr = frame_flat.T.reshape(-1) * 120
    # print(some_arr.shape)
    # raise SystemError
    file = open(location+str(index)+'.bin', mode='wb')
    x=some_arr.tofile(file)
    # print(x)
    # raise SystemError








dataset_original   	  = np.load(args.orig)[:]
original_loader    	  = torch.utils.data.DataLoader(dataset_original, batch_size=dataset_original.shape[0], shuffle=False, num_workers=0, drop_last=False)

dataset_predicted  	  = np.load(args.pred)[:]
predicted_loader      = torch.utils.data.DataLoader(dataset_predicted, batch_size=dataset_original.shape[0], shuffle=False, num_workers=0, drop_last=False)


batch_size=dataset_original.shape[0]
predicted_iter = iter(predicted_loader)

process_input 	  = from_polar if args.no_polar else lambda x : x


if not os.path.exists('pcd/'+str(args.folder)+'/'):
	os.makedirs('pcd/'+str(args.folder)+'/')


recons_filtered = []
originaldy=[]

fromreconStatic = 0
fromdynamic     = 0

for  i,orig in enumerate(original_loader):
	pred = next(predicted_iter)
	print("Strte")

	originaldy = orig[:,:,:,::2]
	recons     = pred[:,:,:,:]

	# print(originaldy.shape, recons.shape)
	# exit(0)
	original_temp = np.array(originaldy.detach().cpu())    	    
	recons_temp   = np.array(recons.detach().cpu())    			
	filtered_dy_recon = np.ndarray([batch_size,2,60,256])
	
	for index in trange(batch_size):
		dynamic  = original_temp[index:index+1].reshape([2,60,256])
		reconSt  = recons_temp[index:index+1].reshape([2,40,256])
		filtered = np.ndarray([2,60,256])
		# print(dynamic.shape, reconSt,shape, filtered.shape)

		for k in range(reconSt.shape[1]):
			#thresh = thresh - 0.001
			for l in range(reconSt.shape[2]):
				if(withinRange(reconSt[0][k][l],dynamic[0][k+5][l])):
					fromdynamic+=1
					filtered[:,k+5:k+6,l:l+1]=dynamic[:,k+5:k+6,l:l+1]
				else:
					fromreconStatic+=1
					filtered[:,k+5:k+6,l:l+1]=reconSt[:,k:k+1,l:l+1]
		filtered[:,45:60,:] = dynamic[:,45:60,:]
		filtered[:,0:5,:] = dynamic[:,0:5,:]
		
		filtered_dy_recon[index] = filtered
		get_bin(from_polar_np(filtered.reshape([1,2,60,256])).reshape(3,60,256),index+i*batch_size,'pcd/'+str(args.folder)+'/')
		#get_pcd_from_img_and_save(torch.Tensor(dynamic.reshape([1,2,40,256])).cuda(),index+i*batch_size,'testModel/original/')	
	# save_on_disk(str(i),filtered_dy_recon)
	recons_filtered=filtered_dy_recon


print(fromreconStatic/(fromdynamic+fromreconStatic))


if not os.path.exists('samplesVid/'+str(args.thresh)):
	os.makedirs('samplesVid/'+str(args.thresh))


i=0
for frame_num in trange(1024):
	if(i<1024):
		i+=1
		frame=from_polar(torch.Tensor(filtered_dy_recon[frame_num:frame_num+1,:,:,:]).cuda()).detach().cpu()
		plt.figure()
		plt.xlim(-0.3, 0.3)
		plt.ylim(-0.3, 0.3)
		plt.scatter(frame[:, 0], frame[:, 1], s=0.7, color='k')
		plt.savefig('samplesVid/'+str(args.thresh)+'/'+str(frame_num)+'.jpg') 
	else:
		break
                                                              
