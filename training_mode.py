import lgsvl
from environs import Env
import time
import cv2
import os
import csv
import copy 
import random
import math
import argparse
import shutil
import numpy as np

# Lincoln 2017 MKZ Apollo 5.0 vehicle parameters
WHEEL_BASE=2.845
MAX_STEER_DEGREES=39.4
MAX_ACCELERATION=6

def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")

def run_trials(log_file,training_images_folder,number_of_trials):
    if os.path.exists(training_images_folder) and os.path.isdir(training_images_folder):
        shutil.rmtree(training_images_folder)
        os.makedirs(training_images_folder)
    else:
        os.makedirs(training_images_folder)
    if os.path.isfile(log_file):
        os.remove(log_file)
    for trial_no in range(number_of_trials):
        generate_dataset(log_file,training_images_folder,trial_no)

def get_steering_angle(yaw_rate,velocity):
    steer_degrees=yaw_rate*WHEEL_BASE*180/(velocity*math.pi)
    steer_angle=min(steer_degrees,MAX_STEER_DEGREES)
    steer_angle=max(steer_degrees,-MAX_STEER_DEGREES)

    return steer_angle

def get_acceleration(prev_v,v,step_time):
    return (v-prev_v)/step_time


def env_config(testing=False):
    env=Env()
    # load environment
    sim = lgsvl.Simulator(os.environ.get('SIMULATOR_HOST', '127.0.0.1'), 8181)
    if sim.current_scene == 'Straight2LaneSame':
        sim.reset()
    else:
        sim.load('Straight2LaneSame')

    spawns = sim.get_spawn()

    ego_state=lgsvl.AgentState()
    ego_state.transform=spawns[0]


    ego_right=lgsvl.utils.transform_to_right(ego_state.transform)
    ego_state.transform.position=ego_state.transform.position-3.5*ego_right

    ego_forward=lgsvl.utils.transform_to_forward(spawns[0])
    # set agent initial velocity to 12 m/s
    ego_state.velocity=12*ego_forward
    # Lincoln Apollo 5.0 configuration 
    ego=sim.add_agent("47b529db-0593-4908-b3e7-4b24a32a0f70",lgsvl.AgentType.EGO,ego_state)

    npc1_state=copy.deepcopy(ego_state)
    # NPC, 12 meters ahead
    npc1_state.transform.position=ego_state.transform.position + (ego_forward * 12)
    # randomize npc color
    npc1_color=lgsvl.Vector(random.randint(0,10),random.randint(0,10),random.randint(0,10))
    npc1 = sim.add_agent("Sedan", lgsvl.AgentType.NPC,npc1_state,npc1_color)

    if testing==False:
        return sim,ego,npc1

    npc2_state=copy.deepcopy(ego_state)
    npc2_state.transform.position=ego_state.transform.position + (ego_forward * 32) + 3.5*ego_right
    npc2_color=lgsvl.Vector(random.randint(0,10),random.randint(0,10),random.randint(0,10))
    npc2 = sim.add_agent("Sedan", lgsvl.AgentType.NPC,npc2_state,npc2_color)
    
    
    return sim,ego,npc1,npc2


def generate_dataset(log_file,training_images_folder,trial_no):

    print(f"trial_no:{trial_no}")
    sim,ego,npc=env_config()

    duration=5
    step_time=0.10
    step_rate=int(1.0/step_time)
    steps=duration*step_rate


    with open(log_file,'a') as f1:
        writer=csv.writer(f1,delimiter=',',lineterminator='\n')
        state=ego.state
        prev_v=state.velocity.magnitude()
        for i in range(steps):
            sim.run(time_limit=step_time)
            state=ego.state

            for sensor in ego.get_sensors():
                if sensor.name == "Main Camera":
                    filename=training_images_folder+'/img_'+str(trial_no)+'_'+str(i)+'.jpg'
                    print(filename)
                    sensor.save(filename,compression=8) # save front camera image
                    img=cv2.imread(filename)            # read image of size h=1080, w=1920
                    img_cropped=img[img.shape[0]//2:]   # crop some upper part of image
                    img=cv2.resize(img_cropped,(320,90))# resize image to h=90, w=320
                    cv2.imwrite(filename,img)           # rewrite image to specified folder

            yaw_rate=state.angular_velocity.y
            v=state.velocity.magnitude()
            steer_angle=get_steering_angle(yaw_rate,v)
            acceleration=get_acceleration(prev_v,v,step_time)
            prev_v=v
            print(f'steer_angle: {steer_angle} acc: {acceleration}')
            row=[filename,steer_angle,acceleration]
            writer.writerow(row)

    sim.close()

#collision_callback

def get_data(log_file_path):
    samples=[]
    with open(log_file_path) as csvfile:
        reader=csv.reader(csvfile)
        for line in reader:
            samples.append(line)
            
    return samples


def preprocess(samples):
    s=np.array(samples)
    # filter out samples with very small steering angle values
    angles=s[:,1].astype('float')
    mask=abs(angles)>.75
    s=s[mask]
    return s


if __name__=="__main__":
    # parser
    print(__file__)
    parser=argparse.ArgumentParser(description='training mode, move the vehicle using the arrow keys')
    parser.add_argument('--num_trials',default=30,type=int,help='no. of trials')
    parser.add_argument('--saving_dir',default=os.path.dirname(os.path.realpath(__file__)),type=dir_path,help='specify a directory for the images and log file to be saved')
    args=parser.parse_args()
    log_file_path=args.saving_dir+'/log'+'.csv'
    training_images_folder_path=args.saving_dir+'/training_mode_images'
    run_trials(log_file_path,training_images_folder_path,args.num_trials)
    