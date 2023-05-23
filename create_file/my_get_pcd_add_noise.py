from torchvision import datasets, transforms
import torch.utils.data
import torch
import sys
import argparse
import matplotlib.pyplot as plt
from utils512 import * 
#from models512 import *
import shutil
import open3d as o3d
from open3d import open3d
from tqdm import tqdm



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
parser.add_argument('--debug', action='store_true')
# args = parser.parse_args(args=['atlas_baseline=0, autoencoder=1,panos_baseline=0'])





# Encoder must be trained with all types of frames,dynmaic, static all

args = parser.parse_args()
model = VAE(args).cuda()
network=torch.load('ATM-TRACK/v6[2-40-512-Transformed]-120-Adversarial_STST_STDY_alpha-1-reconstructionPlugged-Decoderfreezed-nc2Pairs/gen_45.pth')
model.load_state_dict(network['state_dict_gen_dy'])
model.eval()



#Helper Functions
#-----------------------------------------------------------------------------------------

def save_on_disk(folder,filtered_dy_recon):
	np.save('SaveRecons/'+folder+'/filtered',filtered_dy_recon)



def withinRange(x ,y ,thresh):
    if(abs(x-y) < thresh):
        return 1
    else:
        return 0



def get_pcd_from_img_and_save(img,index,location):
    
    frame = from_polar(img).detach().cpu().numpy()
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

    # print(frame.shape)
    # exit(1)
    frame_flat = frame.reshape((3,-1))
    some_pcd = o3d.PointCloud()
    some_arr = frame_flat.T * 120
    #some_arr = some_arr[[(-40 < x < 40) for x, y, z in some_arr]]
    some_pcd.points = o3d.open3d.Vector3dVector(some_arr)

    o3d.write_point_cloud(location+str(index)+".pcd", some_pcd)
    del some_pcd,some_arr,frame_flat
    torch.cuda.empty_cache()







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
    	get_pcd_from_non_preprocessed_npy_and_save(torch.Tensor(frame.reshape([1,3,64,512])).cuda(),str(frame_num))

    	





def save_images(recons_filtered, batch_num,folder):
	i=0
	for frame_num in range(recons_filtered.shape[0]):
		if(i<args.batch_size):
			i+=1
			frame=from_polar(torch.Tensor(recons_filtered[frame_num:frame_num+1,:,:,:]).cuda()).detach().cpu().numpy()
			frame = frame.reshape((3,-1)).T*120
			#frame = frame[[(-40 < x < 40) for x, y, z in frame]]
			# frame =frame.reshape([1,3,40,512])
			plt.figure()
			plt.scatter(frame[:,0], frame[:,1], s=0.7, color='k')
			plt.savefig('samplesVid/'+folder+'/'+str(frame_num+batch_num*args.batch_size)+'.jpg') 
		else:
			break


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

def zero_down_entries_and_save(frames):
	mask = np.random.choice([0, 1], size=frames.shape, p=[1./5, 4./5])
	print(mask.shape)
	# print(mask)
	original_temp = np.multiply(frames , mask)
	print(frames.shape)
	for index in tqdm(range(frames.shape[0])):
		dynamic  = original_temp[index:index+1].reshape([2,64,512])
		get_pcd_from_img_and_save(torch.Tensor(dynamic.reshape([1,2,64,512])).cuda(),index+i*batch_size,'testModel/original/')




#-----------------------------------------------------------------------------------------
#get_pcd_without_preprocess('6.npy','/home/prashant/Desktop/test/')
#-----------------------------------------------------------------------------------------


#print the correspoding data for refrence
# dataset_corr_st   = np.load('/home/prashant/Desktop/test/3.npy')
# dataset_corr_st   = preprocess(dataset_corr_st).astype('float32')

#validation input dataset
dataset_val   	  = np.load('/home/prashant/Desktop/test/6prep-64beam.npy')
dataset_val	  =dataset_val[:]
#dataset_val   	  = preprocess(dataset_val).astype('float32')
val_loader    	  = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)
process_input 	  = from_polar if args.no_polar else lambda x : x
# print 
batch_size=args.batch_size




recons_filtered=[]
originaldy=[]

fromreconStatic = 0
fromdynamic     = 0
thresh = 0.05

for i, img in enumerate(val_loader): 

	# if(os.path.exists('SaveRecons/'+str(i))):
	# 	shutil.rmtree('SaveRecons/'+str(i))

	# os.makedirs('SaveRecons/'+str(i))
	
	img = img.cuda()
	originaldy = img

	mask = np.random.choice([0, 1], size=originaldy.shape, p=[1./5, 4./5])
	original_temp = np.multiply(originaldy.detach().cpu() , mask)
	original_temp = original_temp.cuda()
	original_temp = original_temp.type(torch.FloatTensor)

	recon, kl_cost,z = model(process_input(original_temp[:,:,5:45,:]).cuda())
	recons     = recon[:,:,:40,:]

	
	# original_temp = np.array(originaldy.detach().cpu())
	#print(np.count_nonzero(original_temp))
	
	recons_temp   = np.array(recons.detach().cpu())    			
	filtered_dy_recon = np.ndarray([batch_size,2,64,512])
	
	for index in tqdm(range(batch_size)):
		dynamic  = original_temp[index:index+1].reshape([2,64,512])
		reconSt  = recons_temp[index:index+1].reshape([2,40,512])
		filtered = np.zeros_like(dynamic)

		for k in range(reconSt.shape[1]):
			#thresh = thresh - 0.001
			for l in range(reconSt.shape[2]):
				if(withinRange(reconSt[0][k][l],dynamic[0][k+5][l],thresh)):
					fromdynamic+=1
					filtered[:,k+5:k+6,l:l+1]=dynamic[:,k+5:k+6,l:l+1]
				else:
					fromreconStatic+=1
					filtered[:,k+5:k+6,l:l+1]=reconSt[:,k:k+1,l:l+1]
		filtered[:,45:64,:] = dynamic[:,45:64,:]
		filtered[:,0:5,:] = dynamic[:,0:5,:]
		

		filtered_dy_recon[index] = filtered
		get_pcd_from_img_and_save(torch.Tensor(filtered.reshape([1,2,64,512])).cuda(),index+i*batch_size,'testModel/pcd0.05/')
		#get_pcd_from_img_and_save(torch.Tensor(dynamic.reshape([1,2,64,512])).cuda(),index+i*batch_size,'testModel/original/')	
	print(fromreconStatic/(fromreconStatic+fromdynamic))
	recons_filtered=filtered_dy_recon
	#torch.cuda.empty_cache()
	save_images(recons_filtered,i,'filtered0.05')
	torch.cuda.empty_cache()
	#save_images(original_temp,i,'original')
	#save_images(recons_temp,i,'reconstructed')
	#GC.collect()









# i=0
# for frame_num in range(originaldy.shape[0]):
# 	if(i<2048):
# 		i+=1
# 		frame=from_polar(originaldy[frame_num:frame_num+1,:,:,:]).detach().cpu()
# 		plt.figure()
# 		plt.scatter(frame[:, 0], frame[:, 1], s=0.7, color='k')
# 		plt.savefig('samplesVid/original/'+str(frame_num)+'.jpg') 
# 	else:
# 		break
                                                              



# print(fromreconStatic)
# print(fromdynamic)



	




	
	
