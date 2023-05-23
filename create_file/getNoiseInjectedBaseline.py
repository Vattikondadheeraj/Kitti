import numpy as np
from tqdm import trange
import random
import argparse
import torch
from  utils512 import *
import open3d as o3d
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Noise LIDAR Injection')
parser.add_argument('--orig',       type=str,   default='',              help='Location of the orignal LiDAR')
parser.add_argument('--pred',       type=str,   default='',              help='Location of the predicted LiDAR')
parser.add_argument('--batch_size', type=int,   default=2048,            help='Batch size')
parser.add_argument('--folder',     type=str,   default='',            help='Folder name')
parser.add_argument('--thresh',     type=float, default=0.05,            help='Threshold for filtering')

args = parser.parse_args()

def withinRange(x,y,thresh):
    if(abs(x-y)<thresh):
        return 1
    else:
        return 0




def get_pcd_from_img_and_save(img,index,location):
    
    frame = from_polar(img).detach().cpu().numpy()
    
    frame_flat = frame.reshape((3,-1))
    some_pcd = o3d.geometry.PointCloud()
    some_arr = frame_flat.T * 120
    some_arr = some_arr[[(-40 < x < 40) for x, y, z in some_arr]]
    some_pcd.points = o3d.utility.Vector3dVector(some_arr)

    o3d.io.write_point_cloud(location+str(index)+".pcd", some_pcd)
    del some_pcd,some_arr,frame_flat



def plot(frame, frame_num):
    frame=from_polar(torch.Tensor(frame).cuda()).detach().cpu()
    plt.figure()
    plt.scatter(frame[:, 0], frame[:, 1], s=0.7, color='k')
    plt.savefig('samplesVid/'+str(args.thresh)+'/'+str(frame_num)+'.jpg')
    plt.close()





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














gnd_data=np.load(args.orig)
predicted_data=np.load(args.pred)

dynamic_data=gnd_data[:,:,:,::2]
reconSt_data=predicted_data[:,:,:,:]
# print(dynamic_data.shape, reconSt_data.shape)
# exit(0)
batch_size=dynamic_data.shape[0]

if not os.path.exists('samplesVid/'+str(args.thresh)):
	os.makedirs('samplesVid/'+str(args.thresh))

if not os.path.exists('pcd/'+str(args.folder)):
	os.makedirs('pcd/'+str(args.folder))


final=[]
fromdy = 0
fromrecon = 0

for indx in trange(batch_size):
    dynamic=dynamic_data[indx]
    reconSt=reconSt_data[indx]
    # filtered=np.ndarray([2,60,256])
    # print(dynamic.shape, reconSt.shape, filtered.shape) #(2, 60, 256) (2, 40, 256) (2, 60, 256)
    # exit(0)
    e_z=[]
    e_r=[]
    for k in range(reconSt.shape[1]):
        
        for l in range(reconSt.shape[2]):
            if withinRange(reconSt[0][k][l],dynamic[0][k+5][l], args.thresh):
                fromdy +=1
                pass
            else:
                fromrecon += 1
                e_r.append(reconSt[0][k][l]-dynamic[0][k+5][l])
                e_z.append(reconSt[1][k][l]-dynamic[1][k+5][l])

    idx1=random.choices(range(5,45),k=len(e_r))
    idx2=random.choices(range(0,256),k=len(e_r))

    
        
    for i in range(len(e_r)):
        dynamic[0][idx1[i]][idx2[i]] = dynamic[0][idx1[i]][idx2[i]]+e_r[i]
        dynamic[1][idx1[i]][idx2[i]] = dynamic[1][idx1[i]][idx2[i]]+e_z[i]
    #filtered[:,45:60,:]=dynamic[:,45:60,:]
    #filtered[:,:5,:]=dynamic[:,:5,:]
    final.append(dynamic)
    # print(dynamic.shape)
    get_bin(from_polar_np((dynamic.reshape([1,2,60,256]))).reshape(3,60,256),indx,'pcd/'+str(args.folder)+'/')
    # plot(dynamic.reshape([1,2,60,256]), indx)
    
print(fromrecon/(fromdy+fromrecon))
    
# np.save(args.pred[:-4]+"noise.npy",np.array(final))


i=0
for frame_num in trange(1024):
	if(i<1024):
		i+=1
		frame=from_polar(torch.Tensor(filtered_dy_recon[frame_num:frame_num+1,:,:,:]).cuda()).detach().cpu()
		plt.figure()
		plt.scatter(frame[:, 0], frame[:, 1], s=0.7, color='k')
		plt.savefig('samplesVid/'+str(args.thresh)+'/'+str(frame_num)+'.jpg') 
	else:
		break