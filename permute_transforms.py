import os
import json
x = [0.0,90.0,180.0,270.0,-90.0,-180.0,-270.0]
y = [0.0, 90.0,180.0,270.0,-90.0,-180.0,-270.0]
z = [0.0, 90.0,180.0,270.0,-90.0,-180.0,-270.0]

k=str(1.0)

for i in x:
    for j in y:
        for k in z:

           dictionary={"x":0.0, "y":0.0, "z":0.0, "qx":i, "qy":j, "qz":k, "qw":0.0}
           json_object = json.dumps(dictionary, indent=4)
             
            # Writing to sample.json
           with open("transform1.json", "w") as outfile:
                outfile.write(json_object)

           plotname = str(i)[:-2]+'_'+str(j)[:-2]+'_'+str(k)[:-2]
           command = "evo_traj tum 1.txt --ref kitti09.txt  --plot_mode xyz  --save_plot plots/"+plotname+ " --transform_left transform1.json"
           print(command)
           # exit(0)
           os.system(command)


