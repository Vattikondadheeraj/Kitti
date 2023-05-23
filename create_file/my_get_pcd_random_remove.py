from torchvision import datasets, transforms
import torch.utils.data
import torch
import sys
import argparse
import matplotlib.pyplot as plt
from utils512 import * 
# from models512 import *
import shutil
import open3d as o3d
from open3d import open3d
from tqdm import trange



parser = argparse.ArgumentParser(description='VAE training of LiDAR')
parser.add_argument('--batch_size',         type=int,   default=512,             help='size of minibatch used during training')
parser.add_argument('--use_selu',           type=int,   default=0,              help='replaces batch_norm + act with SELU')
parser.add_argument('--base_dir',           type=str,   default='runs/test',    help='root of experiment directory')
parser.add_argument('--no_polar',           type=int,   default=0,              help='if True, the representation used is (X,Y,Z), instead of (D, Z), where D=sqrt(X^2+Y^2)')
parser.add_argument('--lr',                 type=float, default=1e-3,           help='learning rate value')
parser.add_argument('--z_dim',              type=int,   default=160,            help='size of the bottleneck dimension in the VAE, or the latent noise size in GAN')
parser.add_argument('--autoencoder',        type=int,   default=1,              help='if True, we do not enforce the KL regularization cost in the VAE')
parser.add_argument('--atlas_baseline',     type=int,   default=0,              help='If true, Atlas model used. Also determines the number of primitives used in the model')
parser.add_argument('--panos_baseline',     type=int,   default=0,              help='If True, Model by Panos Achlioptas used')
parser.add_argument('--kl_warmup_epochs',   type=int,   default=150,            help='number of epochs before fully enforcing the KL loss')
parser.add_argument('--data',               type=str,   default='',             help='If True, Model by Panos Achlioptas used')
parser.add_argument('--folder',             type=str,   default='',             help='number of epochs before fully enforcing the KL loss')

parser.add_argument('--debug', action='store_true')
args = parser.parse_args()



#Helper Functions
#----------------------------------------------------------------------------------------


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
    # some_arr = some_arr[[(-40 < x < 40) for x, y, z in some_arr]]
    some_pcd.points = o3d.utility.Vector3dVector(some_arr)

    o3d.io.write_point_cloud(location+str(index)+".pcd", some_pcd)
    del some_pcd,some_arr,frame_flat







def get_pcd_from_non_preprocessed_npy_and_save(img,index):
    
    frame = img.cpu().numpy()     #This is alrasy in shape in (x,y, z) form 
    # frame_actual = np.array([frame_image[:29] for frame_image in frame])
    # print(frame.shape)
    #-----------------------------------------------------------------------
    #frame = frame[:,:,:,::2]  #retrun only 256 of the 512 
    #-----------------------------------------------------------------------
    #return only 384 of the 512 beams

    # listt=np.random.choice(frame.shape[3],384,replace=False)
    # listt=np.sort(listt)
    # frame = frame[:,:,:,listt]

    #-----------------------------------------------------------------------

    # save as image
    plt.figure()
    plt.scatter(frame[:,0], frame[:,1], s=0.7, color='k')
    plt.savefig('samplesVid/without_prep/'+str(index)+'.jpg') 

	#-----------------------------------------------------------------------


    # print(frame.shape)
    # exit(1)
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
    	get_pcd_from_non_preprocessed_npy_and_save(torch.Tensor(frame.reshape([1,3,60,512])).cuda(),str(frame_num))

    	




def add_noise(batch , noise):
	batch_xyz = from_polar(batch)
	noise_tensor = torch.zeros_like(batch_xyz).normal_(0, noise)
	noise_tensor = torch.zeros_like(batch_xyz).normal_(0, noise)
	means = batch_xyz.transpose(1,0).reshape((3, -1)).mean(dim=-1)
	stds  = batch_xyz.transpose(1,0).reshape((3, -1)).std(dim=-1)
	means, stds = [x.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) for x in [means, stds]]

	# normalize data
	norm_batch_xyz = (batch_xyz - means) / (stds + 1e-9)
	# add the noise
	input = norm_batch_xyz + noise_tensor

	# unnormalize
	input = input * (stds + 1e-9) + means

	return to_polar(input)



#this function removes entries from the array and

def zero_down_entries_and_save(frames, prob, folder):
    # print(frames.shape)
    mask = np.random.choice([0, 1], size=(frames.shape[0],1,40,256), p=[prob, 1-prob])
    final_mask = np.random.choice([0, 0], size=(frames.shape[0],2,40,256), p=[prob, 1-prob])
    for i in range(frames.shape[0]):
            final_mask[i:i+1,0:1] = mask[i:i+1]
            final_mask[i:i+1,1:2] = mask[i:i+1] 
            # print(mask.shape)
    # print(np.count_nonzero(final_mask)/(frames.shape[0]*2*40*256))
    # exit(0)
    masked_frame = np.multiply(frames[:,:,5:45,:], final_mask)
    # print(masked_frame)
    # exit(0)
    frames[:,:,5:45,:] = masked_frame

    for index in trange(frames.shape[0]):
        dynamic  = frames[index:index+1]
        # print(dynamic.shape)
        get_bin(from_polar_np(dynamic.reshape([1,2,60,256])).reshape(3,60,256), index, folder)

    return frames


def saveimages(npy, folder):
    npy = from_polar_np(npy)
    if os.path.exists('samplesVid/'+folder):
        shutil.rmtree('samplesVid/'+folder)
        shutil.rmtree('pcd/'+folder)

    os.makedirs('samplesVid/'+folder)
    os.makedirs('pcd/'+folder)


    import matplotlib.pyplot as plt
    for i in trange(npy.shape[0]):
        frame = npy[i:i+1]
        plt.figure()
        plt.scatter(frame[:,0], frame[:,1], s=0.7, color='k')
        plt.savefig('samplesVid/'+folder+'/'+str(i)+'.png')
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







npyList = [5,6,9,10]
prob    = [0.0754, 0.0822, 0.0719, 0.0843]

for i in range(len(npyList)): 

    originalst = np.load(args.data  + str(npyList[i]) + '.npy')[:,:,:,::2]
    print(originalst.shape) #[:,2,60,256]
    # exit(0)

    if os.path.exists('pcd/remove' + str(npyList[i]) + '/'):
        shutil.rmtree('pcd/remove' + str(npyList[i]) + '/')

    os.makedirs('pcd/remove' + str(npyList[i]) + '/')


    masked_frame = zero_down_entries_and_save(originalst, prob[i], 'pcd/remove' + str(npyList[i]) + '/')
    # saveimages(masked_frame, 'samplesVid/' + str(npyList[i]))
