from behavioral_model import Model
from training_mode import env_config
import torch
import time
import os
import shutil
import cv2
import lgsvl
import argparse
import numpy as np
from training_mode import MAX_STEER_DEGREES,MAX_ACCELERATION

print(os.path.dirname(os.path.realpath(__file__)))
def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model_dir',default=os.path.dirname(os.path.realpath(__file__)),type=dir_path,help='a directory where the model file is located')
    parser.add_argument('--runs',default=10,type=int)
    args= parser.parse_args()
    return args


def overtaking(model_path,driving_images_folder_path):
    if os.path.exists(driving_images_folder_path) and os.path.isdir(driving_images_folder_path):
        shutil.rmtree(driving_images_folder_path)
        os.makedirs(driving_images_folder_path)
    else:
        os.makedirs(driving_images_folder_path)
    model=Model()
    model.load_state_dict(torch.load(model_path))
    

    sim,ego,npc1,npc2=env_config(testing=True)

    duration=8
    step_time=0.10
    step_rate=int(1.0/step_time)
    steps=duration*step_rate
    
    model.eval()
    steering_commands=[]
    for i in range(steps):
        # get front camera data
        for sensor in ego.get_sensors():
            if sensor.name == "Main Camera":
                filename=driving_images_folder_path+'/img_'+'_'+str(i)+'.jpg'
                sensor.save(filename,compression=8) # save front camera image
                img=cv2.imread(filename)            # read image of size h=1080, w=1920
                img_cropped=img[img.shape[0]//2:]   # crop some upper part of image
                img=cv2.resize(img_cropped,(320,90)) # resize image to w=90, h=320
                gray_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert RGB to grayscale
                gray_img=np.array(gray_img)[:,:,np.newaxis] # (H,W,C)
                # rewrite image to specified folder
                cv2.imwrite(filename,img)
                
            
        state=ego.state
        c=lgsvl.VehicleControl()
        x=torch.tensor(gray_img).unsqueeze(0).permute(0,3,1,2) # infer using grayscale image, add a zeroth axis, set to (N,C,H,W)

        pred=model(x) #Inference

        steer_angle,acceleration=pred.select(1,0).item(),pred.select(1,1).item() # select axis 1, index 0 and 1, for steer_angle and acceleration respectively
        print(f'ste:{steer_angle}, acc:{acceleration}')
        c.steering=steer_angle/MAX_STEER_DEGREES # Normalize to [-1,+1]
        ego.apply_control(c,True)
        c.throttle=acceleration/MAX_ACCELERATION # Normalize to [-1,+1]
        ego.apply_control(c,True)
        steering_commands.append([steer_angle,acceleration])
        # t1=time.time()
        sim.run(time_limit=step_time)
        # t2=time.time()

    np.save('driving.npy',np.array(steering_commands))
    sim.close()

if __name__=='__main__':
    args=parse_args()
    model_path=args.model_dir+'/model.pt'
    path=os.path.dirname(os.path.realpath(__file__))
    driving_images_folder_path=path+'/driving'
    runs=args.runs
    for i in range(runs):
        overtaking(model_path,driving_images_folder_path)