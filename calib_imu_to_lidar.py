import numpy as np
import pandas as pd
import argparse


parser = argparse.ArgumentParser(description="Calibrate GT pose for imu shift")
parser.add_argument("--pose_file", help="pose path file in KITTI format")
parser.add_argument("--save", help="Location to save")

args = parser.parse_args()


 


imu_to_lidar_transform = np.array([7.027555e-03, -9.999753e-01, 2.599616e-05, -7.137748e-03, -2.254837e-03, -4.184312e-05, -9.999975e-01, -7.482656e-02, 9.999728e-01, 7.027479e-03, -2.255075e-03, -3.336324e-01, 0, 0, 0, 1]).reshape(4,4)

imu_to_lidar_transform = np.linalg.inv(imu_to_lidar_transform)
    

f = open(args.save,"w+")


poses = pd.read_csv(args.pose_file, sep=" ", header=None)

for pose in poses.iterrows():
    # print(type(pose[1]))      #padas.core.series.Series
    x = list(pose[1])
    x.append(0)
    x.append(0)
    x.append(0)
    x.append(1)
    pose_matrix = np.array(x).reshape(4,4)
    transformed_pose = np.dot(pose_matrix, imu_to_lidar_transform).reshape(-1,16)
    f.write(str(transformed_pose[0][0]))
    x = [ f.write(' ' + str(transformed_pose[0][i])) for i in range(1, 12) ]
    f.write('\n')
    print(transformed_pose[0])
    # exit(0)
    # f.writeline(transformed_pose.resh)
    




x = -y 
z= x
y = z

