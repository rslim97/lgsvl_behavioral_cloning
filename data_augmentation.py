from training_mode import get_data
import os
import cv2
import shutil
import numpy as np
import csv

def augment_data(log_file_dir):
	log_file_path=log_file_dir+'/log.csv'
	log_file_augmented_path=log_file_dir+'/log_augmented'+'.csv'
	new_training_images_folder=log_file_dir+'/training_mode_images_flipped'
	# this program checks if log_file exists in log_file dir, during execution log_file will be removed
	if os.path.exists(log_file_path):
		pass
	else:
		# if both log_file and log_file_augmented does not exist raise exception
		if not os.path.exists(log_file_augmented_path):
			raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), log_file_path)
		else:
			# if log_file_augmented exists,
			return log_file_augmented_path
	samples=get_data(log_file_path)
	s=np.array(samples)
	# copy rows in original log file to log_augmented.csv
	with open(log_file_path,'r') as f_input, open(log_file_augmented_path,'w') as f_temp:
	    csv_input=csv.reader(f_input)
	    writer=csv.writer(f_temp,delimiter=',',lineterminator='\n')
	    writer.writerows(csv_input)
	# remove original log file
	os.remove(log_file_path)
	# negate steering angle measurements
	s[:,1]=-s[:,1].astype('float')
	# make new folder for flipped images
	if os.path.exists(new_training_images_folder) and os.path.isdir(new_training_images_folder):
	    shutil.rmtree(new_training_images_folder)
	    os.makedirs(new_training_images_folder)
	else:
	    os.makedirs(new_training_images_folder)
	with open(log_file_augmented_path,'a') as f_output:
	    writer=csv.writer(f_output,delimiter=',',lineterminator='\n')
	    for i in range(s.shape[0]):
	        image_path=s[:,0][i]
	        new_image_name=image_path.split('/')[-1].rpartition('.')[0]+'_'+'flipped'+'.jpg'
	        new_image_path=image_path.rpartition('training_mode_images')[0]+'training_mode_images_flipped/'+new_image_name

	        negative_steering_angle=s[:,1][i]
	        acceleration=s[:,2][i]
	        row=[new_image_path,negative_steering_angle,acceleration]
	        writer.writerow(row)

	        image=cv2.imread(image_path)
	        image_flipped=cv2.flip(image,1)
	        cv2.imwrite(new_image_path,image_flipped)
	    
	return log_file_augmented_path

	#     for i in range(s.shape[0]):
	#         img_path=s[i,0]

	# try:
	#     with open(new_log_file,'a') as f_temp:

