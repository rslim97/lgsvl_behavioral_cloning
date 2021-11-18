
## Behavior Cloning for Vehicle Overtaking in LGSVL Simulator
---
Welcome. This repository contains the implementation of behavior cloning for autonomous overtaking in lgsvl simulator using PythonAPI. The vehicle used was `Lincoln2017MKZ` with `Apollo 5.0` sensor configuration. 
## Installation
start by cloning the behaviour branch in our group project repository
```sh
git clone -b behaviour https://github.com/HonWek/autonomous_overtaking.git
```
## Dependencies
External library required is the lgsvl PythonAPI. Please refer here https://www.svlsimulator.com/docs/python-api/python-api/#quickstart for a quick installation. The code here was developed using:
* Python 3.8.11
* pytorch=1.7.1=py3.8_cuda11.0.221_cudnn8.0.5_0
* numpy=1.19.5
* opencv-python=4.5.4.58
* scikit-learn=1.0.1
* lgsvl Python API 
* lgsvl simulator `release-2021-3`
<a/>

  Note: The Pytorch version used will depend on your cuda version. The `requirements.txt` is also available. It was generated using `conda list -e >  requirements.txt` hence it is not in the right format for `pip`.

## Environment Setup

Lgsvl simulator is provided the `svl` folder in the repository https://github.com/HonWek/autonomous_overtaking/tree/behaviour. 
The Lgsvl simulator can also be downloaded from [https://www.svlsimulator.com/](https://www.svlsimulator.com/). 
Please refer to https://www.svlsimulator.com/docs/installation-guide/installing-simulator/ during installation.
A good tutorial for the simulator installation [tutorial video](https://youtu.be/Ucr0aM334_k?t=1244).
To run the simulator, right-click `simulator.exe` go to properties, permissions tab, and check `Allow executing as program`.
<p align="center">
  <img src="readme_img/Screenshot%20from%202021-11-18%2001-38-54.png" width="25%"/>
</p>

## Simulation Setup

In the simulator WebUI navigate to 'Simulations'. Click 'Add New'. In the General tab, enter simulation name and select 'cluster'. Create cluster first if you have not already. Leave `Headless mode` unselected.
<p align="center">
  <img src="readme_img/Screenshot%20from%202021-11-18%2001-24-06.png" width="86%"/>
</p>

In the 'Test Case' tab, select runtime template as `API Only`. This will automatically disable `Interactive mode`.
<p align="center">
  <img src="readme_img/Screenshot%20from%202021-11-18 01-25-13.png" width="86%"/>
</p>

Click 'Next' in 'Autopilot' tab, and finally you can click `Publish` to publish your simulation. To run a simulation select a simulation and click 'run simulation'.
<p align="center">
  <img src="readme_img/Screenshot%20from%202021-11-18%2001-25-54.png" width="86%"/>
</p>


## Details About Files In This Directory
### `training_mode.py`
Usage of `training_mode.py` is to collect data i.e., front camera images, yaw rate (degrees/s), and vehicle velocity (m/s) of the ego vehicle (agent) while being navigated by a human operator to overtake an npc vehicle in the left lane. The default lgsvl map is `Straight2LaneSame`. The ego vehicle can be controlled by keyboard arrow keys.
```sh
python3 training_mode.py --num_trials <num_trials>
```
After data collection, saved camera images can be viewed under the folder
```sh
training_mode_images
```
If the directory already exists, it will be overwritten.
A log file containing absolute path to each image with steering angle and acceleration will be generated as well.
`log.csv`
```sh
[absolute path to image.jpg]  [steering angle (degrees)] [acceleration (m/s^2)]
...
```
If `log.csv` already exists, it will be overwritten. As a precaution `log.csv` will be replaced with `log_augmented.csv` when `behavioral_model.py` is called.
### `behavioral_model.py`
Contains a Pytorch deep neural network as a function approximator to predict steering angles and acceleration from front camera image. To train the agent with default settings with the hyperparameters `--epochs 80 --batch_size 60`
```sh
python3 behavioral_model.py
```
After training, a pytorch model `model.pt` will be generated in the working directory.
### `model.pt`
`model.pt` is a fully trained model of the network using `80 epochs` and learning rate `lr=7e-5`.
### `driving.py`
To evaluate the model
```sh
python3 driving.py --runs <runs>
```
`model_dir` by default would be the current working directory where `behavioral_model.py` is located as `model.pt` is generated there.
Specify number of test trials `--runs <runs>`.

### `data_augmentation.py`
`data_augmentation` contains the function `augment_data` which will be called in `behavioral_model.py`. It takes in the path to the directory where the original `log.csv` is located. The function will create a directory called `training_mode_images_flipped` containing training images flipped horizontally and `log_augmented.csv` containing absolute paths of the original saved images and the flipped images with their corresponding steering angle and acceleration data. During execution the original `log.csv` will be removed. At the end of execution, returns the path to `log_augmented.csv` .
If `log.csv` exists the function `augment_data` will be executed. Otherwise, the path to `log_augmented.csv` will be returned immediately.

### `training_mode_images`
A folder containing 4500 front camera images collected from 90 trials used to train `model.pt`. Each trial lasts for 5 seconds, meaning the time interval of which the images are collected are 0.1s.
### `training_mode_images_flipped`
A folder containing images in `training_mode_images` flipped horizontally, i.e., around the y-axis.
### `behavioral_demo1.mp4`
Please use git clone or download to view the full video
<p align="center">
  <img src="readme_img/demo1.gif" width="40%"/>
</p>

### `behavioral_demo2.mp4`
<p align="center">
  <img src="readme_img/demo2.gif" width="40%"/>
</p>







