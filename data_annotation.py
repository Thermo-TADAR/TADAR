import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import pickle  
from tqdm import tqdm
import os

from data_sensor_config import sensors_info_list
# Some basic setup:
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries

import numpy as np
import os, json, cv2, random
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2 import model_zoo



def dataloading(data_dict, sensor_topic):
    new_data_flag =  data_dict[sensor_topic + '_new_data_flag'].tolist()
    image_path = data_dict['image_path'].tolist()
    depth_path =  data_dict['depth_path'].tolist()
    mlx_matrix_path = data_dict[ sensor_topic + '_mt_path'].tolist()
    mlx_ambient_temperature = data_dict[ sensor_topic + '_at'].tolist()
    all_timestamps = data_dict[sensor_topic + '_timestamp']
    print("Original number of samples: ", len(all_timestamps) )
    # get valid data
    mlx_matrix = []
    mlx_at = []
    images = []
    depth_maps = []
    timestamps = []
    for index ,flag in enumerate(tqdm(new_data_flag)):
        if flag:
            matrix_path = mlx_matrix_path[index]
            mlx_matrix.append(np.load(matrix_path))
            mlx_at.append(mlx_ambient_temperature[index])
            images.append(cv2.imread(image_path[index]))
            depth_maps.append(np.load(depth_path[index]))
            timestamps.append(all_timestamps[index])
            
    print("Num of samples: " , len(mlx_matrix))
    print("Time duration (s): ", (timestamps[-1] - timestamps[0]).total_seconds())
    print("Sampling rate: ", len(mlx_matrix)/ ((timestamps[-1] - timestamps[0]).total_seconds()))
    return mlx_matrix, mlx_at, images, depth_maps, timestamps


def SubpageInterpolating(subpage):
    shape = subpage.shape
    mat = subpage.copy()
    for i in range(shape[0]):
        for j in range(shape[1]):
            if mat[i,j] > 0.0:
                continue
            num = 0
            try:
                top = mat[i-1,j]
                num = num+1
            except:
                top = 0.0
            
            try:
                down = mat[i+1,j]
                num = num+1
            except:
                down = 0.0
            
            try:
                left = mat[i,j-1]
                num = num+1
            except:
                left = 0.0
            
            try:
                right = mat[i,j+1]
                num = num+1
            except:
                right = 0.0
            mat[i,j] = (top + down + left + right)/num
    return mat


def Alignment(rgb_frame,depth_frame, resize_ratio, offset, rgb_size_reference):
    # the rgb and depth frames are pre-aligned 
    rgb_height = rgb_frame.shape[0]
    rgb_width = rgb_frame.shape[1]
    
    height = int(resize_ratio[0] * rgb_height)
    width = int(resize_ratio[1] * rgb_width)
    
    offset_height = int((offset[0] / rgb_size_reference[0]) * rgb_height)
    offset_width = int((offset[1] / rgb_size_reference[1]) * rgb_width)
    
    if height>rgb_height:  # mlx90640_110
        top = np.abs(offset_height)
        bottom = np.abs(height - (top+ rgb_height) )   
        left = np.abs(offset_width)
        right = np.abs(width - (left+ rgb_width) )
        
        re_rgb = cv2.copyMakeBorder(rgb_frame,top, bottom, left, right, cv2.BORDER_CONSTANT, value= (0,0,0))
        re_depth = cv2.copyMakeBorder(depth_frame,top, bottom, left, right, cv2.BORDER_CONSTANT, value= (255,255,255))
    else:
        re_rgb = rgb_frame[offset_height: offset_height+height , offset_width: offset_width+width, :]
        re_depth = depth_frame[offset_height: offset_height+height , offset_width: offset_width+width, :]
    return re_rgb, re_depth



if __name__=='__main__':
    data_paths = [
        # 'Trainset101/set1/data.pkl',
        # 'Trainset101/set2/data.pkl',
        # 'Trainset101/set3/data.pkl',
        # 'Trainset101/set4/data.pkl',
        # 'Trainset101/set5/data.pkl',
        # 'Trainset101/set6/data.pkl',
        # 'Trainset101/set7/data.pkl',
        # 'Trainset101/set8/data.pkl',
        # 'Trainset101/set9/data.pkl',
        
        # 'Data/HW101/set1/data.pkl',
        # 'Data/HW101/set2/data.pkl',
        # 'Data/HW101/set3/data.pkl',
        # 'Data/HW101/set4/data.pkl',
        # 'Data/HW101/set5/data.pkl',
        # 'Data/HW101/set6/data.pkl',
        # 'Data/HW101/set7/data.pkl',
        # 'Data/HW101/set8/data.pkl',
        # 'Data/HW101/set9/data.pkl',
        # 'Data/HW101/set10/data.pkl',
        # 'Data/HW101/set11/data.pkl',
        # 'Data/HW101/set12/data.pkl',
        # 'Data/HW101/set13/data.pkl',
        # 'Data/HW101/set14/data.pkl',
        # 'Data/HW101/set15/data.pkl',
        # 'Data/HW101/set16/data.pkl',
        # 'Data/HW101/set17/data.pkl',
        # 'Data/HW101/set18/data.pkl',
        # 'Data/HW101/set19/data.pkl',
        # 'Data/HW101/set20/data.pkl',
        # 'Data/HW101/set21/data.pkl',
        # 'Data/HW101/set22/data.pkl',
        # 'Data/HW101/set23/data.pkl',
        # 'Data/HW101/set24/data.pkl',
        # 'Data/HW101/set25/data.pkl',
        # 'Data/HW101/set26/data.pkl',
        # 'Data/HW101/set27/data.pkl',
        # 'Data/HW101/set28/data.pkl',
        # 'Data/HW101/set29/data.pkl',
        # 'Data/HW101/set30/data.pkl',
        # 'Data/HW101/set31/data.pkl',
        # 'Data/HW101/set32/data.pkl',
        # 'Data/HW101/set33/data.pkl',
        # 'Data/HW101/set34/data.pkl',
        # 'Data/HW101/set35/data.pkl',
        # 'Data/HW101/set36/data.pkl',
        # 'Data/HW101/set37/data.pkl',
        # 'Data/HW101/set38/data.pkl',
        # 'Data/HW101/set39/data.pkl',
        # 'Data/HW101/set40/data.pkl',
        # 'Data/HW101/set41/data.pkl',
        # 'Data/HW101/set42/data.pkl',
        # 'Data/HW101/set43/data.pkl',
        # 'Data/HW101/set44/data.pkl',
        # 'Data/HW101/set45/data.pkl',
        # 'Data/HW101/set46/data.pkl',
        # 'Data/HW101/set47/data.pkl',
        # 'Data/HW101/set48/data.pkl',
        # 'Data/HW101/set49/data.pkl',
        # 'Data/HW101/set50/data.pkl',
        # 'Data/HW101/set51/data.pkl',
        # 'Data/HW101/set52/data.pkl',
        # 'Data/HW101/set53/data.pkl',
        # 'Data/HW101/set54/data.pkl',
        # 'Data/HW101/set55/data.pkl',
        # 'Data/HW101/set56/data.pkl',
        # 'Data/HW101/set57/data.pkl',
        # 'Data/HW101/set58/data.pkl',
        # 'Data/HW101/set59/data.pkl',
        # 'Data/HW101/set60/data.pkl',
        # 'Data/HW101/set61/data.pkl',
        
        
        # 'Data/Bathroom_0/data.pkl',
        # 'Data/Bathroom_1/data.pkl',
        # 'Data/Bathroom_2/data.pkl',
        # 'Data/Bathroom_3/data.pkl',
        # 'Data/Bathroom_4/data.pkl',
        # 'Data/Bathroom_5/data.pkl',
        # 'Data/Bathroom_6/data.pkl',
        # 'Data/Bathroom_7/data.pkl',
        # 'Data/Bathroom_8/data.pkl',
        # 'Data/Bathroom1_0/data.pkl',
        # 'Data/Bathroom1_1/data.pkl',
        # 'Data/Bathroom1_2/data.pkl',
        # 'Data/Bathroom1_3/data.pkl',
        # 'Data/Bathroom1_4/data.pkl',
        # 'Data/Bathroom1_5/data.pkl',
        # 'Data/Bathroom1_6/data.pkl',
        # 'Data/Bathroom1_7/data.pkl',
        # 'Data/Bathroom1_8/data.pkl',
        # 'Data/Bathroom1_9/data.pkl',
        # 'Data/Bathroom1_10/data.pkl',
        # 'Data/Bedroom_0/data.pkl',
        # 'Data/Bedroom_1/data.pkl',
        # 'Data/Bedroom_2/data.pkl',
        # 'Data/Bedroom_3/data.pkl',
        # 'Data/Bedroom_4/data.pkl',
        # 'Data/Bedroom_5/data.pkl',
        # 'Data/Bedroom_6/data.pkl',
        # 'Data/Bedroom_7/data.pkl',
        # 'Data/Bedroom_8/data.pkl',
        # 'Data/Bedroom_9/data.pkl',
        # 'Data/Bedroom_10/data.pkl',
        # 'Data/Bedroom_11/data.pkl',
        # 'Data/Bedroom_12/data.pkl',
        # 'Data/Bedroom_13/data.pkl',
        # 'Data/Bedroom_14/data.pkl',
        # 'Data/Bedroom_15/data.pkl',
        # 'Data/Bedroom_16/data.pkl',
        # 'Data/Bedroom_17/data.pkl',
        # 'Data/Bedroom1_0/data.pkl',
        # 'Data/Bedroom1_1/data.pkl',
        # 'Data/Bedroom1_2/data.pkl',
        # 'Data/Bedroom1_3/data.pkl',
        # 'Data/Bedroom1_4/data.pkl',
        # 'Data/Bedroom1_5/data.pkl',
        # 'Data/Bedroom1_6/data.pkl',
        # 'Data/Bedroom1_7/data.pkl',
        # 'Data/Bedroom1_8/data.pkl',
        # 'Data/Bedroom1_9/data.pkl',
        # 'Data/Bedroom1_10/data.pkl',
        # 'Data/Bedroom1_11/data.pkl',
        # 'Data/Bedroom1_12/data.pkl',
        # 'Data/Bedroom1_13/data.pkl',
        # 'Data/Bedroom1_14/data.pkl',
        
        
        # 'Data/HW101_CloseCoat_0/data.pkl',
        # 'Data/HW101_CloseCoat_1/data.pkl',
        # 'Data/HW101_CloseCoat_2/data.pkl',
        # 'Data/HW101_CloseJacket_0/data.pkl',
        # 'Data/HW101_CloseJacket_1/data.pkl',
        # 'Data/HW101_CloseJacket_2/data.pkl',
        # 'Data/HW101_CloseShirt_0/data.pkl',
        # 'Data/HW101_CloseShirt_1/data.pkl',
        # 'Data/HW101_CloseShirt_2/data.pkl',
        # 'Data/HW101_CloseTshirt_0/data.pkl',
        # 'Data/HW101_CloseTshirt_1/data.pkl',
        # 'Data/HW101_CloseTshirt_2/data.pkl',
        
        # 'Data/HW101_AmbientObjects_AC_0/data.pkl',
        # 'Data/HW101_AmbientObjects_AC_1/data.pkl',
        # 'Data/HW101_AmbientObjects_AC_2/data.pkl',
        # 'Data/HW101_AmbientObjects_display_0/data.pkl',
        # 'Data/HW101_AmbientObjects_display_1/data.pkl',
        # 'Data/HW101_AmbientObjects_display_2/data.pkl',
        # 'Data/HW101_AmbientObjects_hotwaterpot_0/data.pkl',
        # 'Data/HW101_AmbientObjects_hotwaterpot_1/data.pkl',
        # 'Data/HW101_AmbientObjects_hotwaterpot_2/data.pkl',
        # 'Data/HW101_AmbientObjects_laptop_0/data.pkl',
        # 'Data/HW101_AmbientObjects_laptop_1/data.pkl',
        # 'Data/HW101_AmbientObjects_laptop_2/data.pkl',
        # 'Data/HW101_AmbientObjects_lights_0/data.pkl',
        # 'Data/HW101_AmbientObjects_lights_1/data.pkl',
        # 'Data/HW101_AmbientObjects_lights_2/data.pkl',
        # 'Data/HW101_AmbientObjects_router_0/data.pkl',
        # 'Data/HW101_AmbientObjects_router_1/data.pkl',
        # 'Data/HW101_AmbientObjects_router_2/data.pkl',
        
        # 'Data/HW101_AmbientObjects_lights_3/data.pkl',
        # 'Data/HW101_AmbientObjects_lights_4/data.pkl',
        # 'Data/HW101_AmbientObjects_lights_5/data.pkl',
        # 'Data/HW101_AmbientObjects_AC_3/data.pkl',
        # 'Data/HW101_AmbientObjects_AC_4/data.pkl',
        # 'Data/HW101_AmbientObjects_AC_5/data.pkl',
        # 'Data/HW101_AmbientObjects_hotwaterpot_3/data.pkl',
        # 'Data/HW101_AmbientObjects_hotwaterpot_4/data.pkl',
        # 'Data/HW101_AmbientObjects_hotwaterpot_5/data.pkl',
        # 'Data/HW101_AmbientObjects_no_0/data.pkl',
        # 'Data/HW101_AmbientObjects_no_1/data.pkl',
        # 'Data/HW101_AmbientObjects_no_2/data.pkl',
        
        # 'Data/HW101_Inangle_0D_0/data.pkl',
        # 'Data/HW101_Inangle_0D_1/data.pkl',
        # 'Data/HW101_Inangle_0D_2/data.pkl',
        # 'Data/HW101_Inangle_15D_0/data.pkl',
        # 'Data/HW101_Inangle_15D_1/data.pkl',
        # 'Data/HW101_Inangle_15D_2/data.pkl',
        # 'Data/HW101_Inangle_30D_0/data.pkl',
        # 'Data/HW101_Inangle_30D_1/data.pkl',
        # 'Data/HW101_Inangle_30D_2/data.pkl',
        # 'Data/HW101_Inangle_m15D_0/data.pkl',
        # 'Data/HW101_Inangle_m15D_1/data.pkl',
        # 'Data/HW101_Inangle_m15D_2/data.pkl',
        # 'Data/HW101_Inangle_m30D_0/data.pkl',
        # 'Data/HW101_Inangle_m30D_1/data.pkl',
        # 'Data/HW101_Inangle_m30D_2/data.pkl',
        
        # 'Data/HW101_Inangle_m15D_3/data.pkl',
        # 'Data/HW101_Inangle_m15D_4/data.pkl',
        # 'Data/HW101_Inangle_m15D_5/data.pkl',
        # 'Data/HW101_Inangle_m30D_3/data.pkl',
        # 'Data/HW101_Inangle_m30D_4/data.pkl',
        # 'Data/HW101_Inangle_m30D_5/data.pkl',
        
        # 'Data/HW101_LightCond0_0/data.pkl',
        # 'Data/HW101_LightCond0_1/data.pkl',
        # 'Data/HW101_LightCond0_2/data.pkl',
        # 'Data/HW101_LightCond1_0/data.pkl',
        # 'Data/HW101_LightCond1_1/data.pkl',
        # 'Data/HW101_LightCond1_2/data.pkl',
        # 'Data/HW101_LightCond2_0/data.pkl',
        # 'Data/HW101_LightCond2_1/data.pkl',
        # 'Data/HW101_LightCond2_2/data.pkl',
        # 'Data/HW101_LightCond3_0/data.pkl',
        # 'Data/HW101_LightCond3_1/data.pkl',
        # 'Data/HW101_LightCond3_2/data.pkl',
        
        # 'Data/HW101_Ori_0D_0/data.pkl',
        # 'Data/HW101_Ori_0D_1/data.pkl',
        # 'Data/HW101_Ori_0D_2/data.pkl',
        # 'Data/HW101_Ori_45D_0/data.pkl',
        # 'Data/HW101_Ori_45D_1/data.pkl',
        # 'Data/HW101_Ori_45D_2/data.pkl',
        # 'Data/HW101_Ori_90D_0/data.pkl',
        # 'Data/HW101_Ori_90D_1/data.pkl',
        # 'Data/HW101_Ori_90D_2/data.pkl',
        # 'Data/HW101_Ori_135D_0/data.pkl',
        # 'Data/HW101_Ori_135D_1/data.pkl',
        # 'Data/HW101_Ori_135D_2/data.pkl',
        # 'Data/HW101_Ori_180D_0/data.pkl',
        # 'Data/HW101_Ori_180D_1/data.pkl',
        # 'Data/HW101_Ori_180D_2/data.pkl', 
        
        # 'Data/HW101_Ori_0D_3/data.pkl',
        # 'Data/HW101_Ori_0D_4/data.pkl',
        # 'Data/HW101_Ori_0D_5/data.pkl',
        # 'Data/HW101_Ori_45D_3/data.pkl',
        # 'Data/HW101_Ori_45D_4/data.pkl',
        # 'Data/HW101_Ori_45D_5/data.pkl',
        # 'Data/HW101_Ori_90D_3/data.pkl',
        # 'Data/HW101_Ori_90D_4/data.pkl',
        # 'Data/HW101_Ori_90D_5/data.pkl',
        # 'Data/HW101_Ori_135D_3/data.pkl',
        # 'Data/HW101_Ori_135D_4/data.pkl',
        # 'Data/HW101_Ori_135D_5/data.pkl',
        # 'Data/HW101_Ori_180D_3/data.pkl',
        # 'Data/HW101_Ori_180D_4/data.pkl',
        # 'Data/HW101_Ori_180D_5/data.pkl',
        
        
        # 'Data/Corridor1_0/data.pkl',  
        # 'Data/Corridor1_1/data.pkl',
        # 'Data/Corridor1_2/data.pkl',
        # 'Data/Corridor1_3/data.pkl',
        # 'Data/Corridor1_4/data.pkl', 
        # 'Data/Corridor1_5/data.pkl', 
        
        # 'Data/Corridor2_0/data.pkl',  
        # 'Data/Corridor2_1/data.pkl',
        # 'Data/Corridor2_2/data.pkl',
        # 'Data/Corridor2_3/data.pkl',
        # 'Data/Corridor2_4/data.pkl', 
        # 'Data/Corridor2_5/data.pkl',
         
        # 'Data/Corridor3_0/data.pkl',  
        # 'Data/Corridor3_1/data.pkl',
        # 'Data/Corridor3_2/data.pkl',
        # 'Data/Corridor3_3/data.pkl',
        # 'Data/Corridor3_4/data.pkl', 
        # 'Data/Corridor3_5/data.pkl', 
        
        # 'Data/Hall_0/data.pkl', 
        # 'Data/Hall_1/data.pkl', 
        # 'Data/Hall_2/data.pkl', 
        # 'Data/Hall_3/data.pkl', 
        # 'Data/Hall_4/data.pkl', 
        # 'Data/Hall_5/data.pkl', 
        
        # 'Data/Meetingroom_0/data.pkl', 
        # 'Data/Meetingroom_1/data.pkl', 
        # 'Data/Meetingroom_2/data.pkl', 
        # 'Data/Meetingroom_3/data.pkl', 
        # 'Data/Meetingroom_4/data.pkl', 
        # 'Data/Meetingroom_5/data.pkl', 
        
        # 'Data/Outdoor1_0/data.pkl', 
        # 'Data/Outdoor1_1/data.pkl', 
        # 'Data/Outdoor1_2/data.pkl', 
        # 'Data/Outdoor1_3/data.pkl', 
        # 'Data/Outdoor1_4/data.pkl', 
        # 'Data/Outdoor1_5/data.pkl', 
        # 'Data/Outdoor1_6/data.pkl', 
        
        # 'Data/Outdoor2_0/data.pkl', 
        # 'Data/Outdoor2_1/data.pkl', 
        # 'Data/Outdoor2_2/data.pkl', 
        # 'Data/Outdoor2_3/data.pkl', 
        # 'Data/Outdoor2_4/data.pkl', 
        # 'Data/Outdoor2_5/data.pkl', 
        # 'Data/Outdoor2_6/data.pkl', 
        
        'Data/HW101_W_phone0/data.pkl', 
        'Data/HW101_W_phone1/data.pkl', 
        'Data/HW101_W_phone2/data.pkl', 
        'Data/HW101_W_phone3/data.pkl', 
        
    ]
    # sensor_indexes = [0,1,2,3,4,5]
    sensor_indexes = [1,4]
    experiment_names = [
        # 'train101set1',
        # 'train101set2',
        # 'train101set3',
        # 'train101set4',
        # 'train101set5',
        # 'train101set6',
        # 'train101set7',
        # 'train101set8',
        # 'train101set9',
        
        # 'HW101set1',
        # 'HW101set2',
        # 'HW101set3',
        # 'HW101set4',
        # 'HW101set5',
        # 'HW101set6',
        # 'HW101set7',
        # 'HW101set8',
        # 'HW101set9',
        # 'HW101set10',
        # 'HW101set11',
        # 'HW101set12',
        # 'HW101set13',
        # 'HW101set14',
        # 'HW101set15',
        # 'HW101set16',
        # 'HW101set17',
        # 'HW101set18',
        # 'HW101set19',
        # 'HW101set20',
        # 'HW101set21',
        # 'HW101set22',
        # 'HW101set23',
        # 'HW101set24',
        # 'HW101set25',
        # 'HW101set26',
        # 'HW101set27',
        # 'HW101set28',
        # 'HW101set29',
        # 'HW101set30',
        # 'HW101set31',
        # 'HW101set32',
        # 'HW101set33',
        # 'HW101set34',
        # 'HW101set35',
        # 'HW101set36',
        # 'HW101set37',
        # 'HW101set38',
        # 'HW101set39',
        # 'HW101set40',
        # 'HW101set41',
        # 'HW101set42',
        # 'HW101set43',
        # 'HW101set44',
        # 'HW101set45',
        # 'HW101set46',
        # 'HW101set47',
        # 'HW101set48',
        # 'HW101set49',
        # 'HW101set50',
        # 'HW101set51',
        # 'HW101set52',
        # 'HW101set53',
        # 'HW101set54',
        # 'HW101set55',
        # 'HW101set56',
        # 'HW101set57',
        # 'HW101set58',
        # 'HW101set59',
        # 'HW101set60',
        # 'HW101set61',
        
        # 'Bathroom_0',
        # 'Bathroom_1',
        # 'Bathroom_2',
        # 'Bathroom_3',
        # 'Bathroom_4',
        # 'Bathroom_5',
        # 'Bathroom_6',
        # 'Bathroom_7',
        # 'Bathroom_8',
        # 'Bathroom1_0',
        # 'Bathroom1_1',
        # 'Bathroom1_2',
        # 'Bathroom1_3',
        # 'Bathroom1_4',
        # 'Bathroom1_5',
        # 'Bathroom1_6',
        # 'Bathroom1_7',
        # 'Bathroom1_8',
        # 'Bathroom1_9',
        # 'Bathroom1_10',
        # 'Bedroom_0',
        # 'Bedroom_1',
        # 'Bedroom_2',
        # 'Bedroom_3',
        # 'Bedroom_4',
        # 'Bedroom_5',
        # 'Bedroom_6',
        # 'Bedroom_7',
        # 'Bedroom_8',
        # 'Bedroom_9',
        # 'Bedroom_10',
        # 'Bedroom_11',
        # 'Bedroom_12',
        # 'Bedroom_13',
        # 'Bedroom_14',
        # 'Bedroom_15',
        # 'Bedroom_16',
        # 'Bedroom_17',
        # 'Bedroom1_0',
        # 'Bedroom1_1',
        # 'Bedroom1_2',
        # 'Bedroom1_3',
        # 'Bedroom1_4',
        # 'Bedroom1_5',
        # 'Bedroom1_6',
        # 'Bedroom1_7',
        # 'Bedroom1_8',
        # 'Bedroom1_9',
        # 'Bedroom1_10',
        # 'Bedroom1_11',
        # 'Bedroom1_12',
        # 'Bedroom1_13',
        # 'Bedroom1_14',
        
        
        # 'HW101_ClothesCoat_0',
        # 'HW101_ClothesCoat_1',
        # 'HW101_ClothesCoat_2',
        # 'HW101_ClothesJacket_0',
        # 'HW101_ClothesJacket_1',
        # 'HW101_ClothesJacket_2',
        # 'HW101_ClothesShirt_0',
        # 'HW101_ClothesShirt_1',
        # 'HW101_ClothesShirt_2',
        # 'HW101_ClothesTshirt_0',
        # 'HW101_ClothesTshirt_1',
        # 'HW101_ClothesTshirt_2',
        
        # 'HW101_AmbientObjects_AC_0',
        # 'HW101_AmbientObjects_AC_1',
        # 'HW101_AmbientObjects_AC_2',
        # 'HW101_AmbientObjects_display_0',
        # 'HW101_AmbientObjects_display_1',
        # 'HW101_AmbientObjects_display_2',
        # 'HW101_AmbientObjects_hotwaterpot_0',
        # 'HW101_AmbientObjects_hotwaterpot_1',
        # 'HW101_AmbientObjects_hotwaterpot_2',
        # 'HW101_AmbientObjects_laptop_0',
        # 'HW101_AmbientObjects_laptop_1',
        # 'HW101_AmbientObjects_laptop_2',
        # 'HW101_AmbientObjects_lights_0',
        # 'HW101_AmbientObjects_lights_1',
        # 'HW101_AmbientObjects_lights_2',
        # 'HW101_AmbientObjects_router_0',
        # 'HW101_AmbientObjects_router_1',
        # 'HW101_AmbientObjects_router_2',
        
        # 'HW101_AmbientObjects_lights_3',
        # 'HW101_AmbientObjects_lights_4',
        # 'HW101_AmbientObjects_lights_5',
        # 'HW101_AmbientObjects_AC_3',
        # 'HW101_AmbientObjects_AC_4',
        # 'HW101_AmbientObjects_AC_5',
        # 'HW101_AmbientObjects_hotwaterpot_3',
        # 'HW101_AmbientObjects_hotwaterpot_4',
        # 'HW101_AmbientObjects_hotwaterpot_5',
        # 'HW101_AmbientObjects_no_0',
        # 'HW101_AmbientObjects_no_1',
        # 'HW101_AmbientObjects_no_2',
        
        # 'HW101_Inangle_0D_0',
        # 'HW101_Inangle_0D_1',
        # 'HW101_Inangle_0D_2',
        # 'HW101_Inangle_15D_0',
        # 'HW101_Inangle_15D_1',
        # 'HW101_Inangle_15D_2',
        # 'HW101_Inangle_30D_0',
        # 'HW101_Inangle_30D_1',
        # 'HW101_Inangle_30D_2',
        # 'HW101_Inangle_m15D_0',
        # 'HW101_Inangle_m15D_1',
        # 'HW101_Inangle_m15D_2',
        # 'HW101_Inangle_m30D_0',
        # 'HW101_Inangle_m30D_1',
        # 'HW101_Inangle_m30D_2',
        
        # 'HW101_Inangle_m15D_3',
        # 'HW101_Inangle_m15D_4',
        # 'HW101_Inangle_m15D_5',
        # 'HW101_Inangle_m30D_3',
        # 'HW101_Inangle_m30D_4',
        # 'HW101_Inangle_m30D_5',
        
        # 'HW101_LightCond0_0',
        # 'HW101_LightCond0_1',
        # 'HW101_LightCond0_2',
        # 'HW101_LightCond1_0',
        # 'HW101_LightCond1_1',
        # 'HW101_LightCond1_2',
        # 'HW101_LightCond2_0',
        # 'HW101_LightCond2_1',
        # 'HW101_LightCond2_2',
        # 'HW101_LightCond3_0',
        # 'HW101_LightCond3_1',
        # 'HW101_LightCond3_2',
        
        # 'HW101_Ori_0D_0',
        # 'HW101_Ori_0D_1',
        # 'HW101_Ori_0D_2',
        # 'HW101_Ori_45D_0',
        # 'HW101_Ori_45D_1',
        # 'HW101_Ori_45D_2',
        # 'HW101_Ori_90D_0',
        # 'HW101_Ori_90D_1',
        # 'HW101_Ori_90D_2',
        # 'HW101_Ori_135D_0',
        # 'HW101_Ori_135D_1',
        # 'HW101_Ori_135D_2',
        # 'HW101_Ori_180D_0',
        # 'HW101_Ori_180D_1',
        # 'HW101_Ori_180D_2',
        
        # 'HW101_Ori_0D_3',
        # 'HW101_Ori_0D_4',
        # 'HW101_Ori_0D_5',
        # 'HW101_Ori_45D_3',
        # 'HW101_Ori_45D_4',
        # 'HW101_Ori_45D_5',
        # 'HW101_Ori_90D_3',
        # 'HW101_Ori_90D_4',
        # 'HW101_Ori_90D_5',
        # 'HW101_Ori_135D_3',
        # 'HW101_Ori_135D_4',
        # 'HW101_Ori_135D_5',
        # 'HW101_Ori_180D_3',
        # 'HW101_Ori_180D_4',
        # 'HW101_Ori_180D_5',
        
        # 'Corridor1_0',  
        # 'Corridor1_1',
        # 'Corridor1_2',
        # 'Corridor1_3',
        # 'Corridor1_4', 
        # 'Corridor1_5', 
        
        # 'Corridor2_0',  
        # 'Corridor2_1',
        # 'Corridor2_2',
        # 'Corridor2_3',
        # 'Corridor2_4', 
        # 'Corridor2_5',
         
        # 'Corridor3_0',  
        # 'Corridor3_1',
        # 'Corridor3_2',
        # 'Corridor3_3',
        # 'Corridor3_4', 
        # 'Corridor3_5', 
        
        # 'Hall_0', 
        # 'Hall_1', 
        # 'Hall_2', 
        # 'Hall_3', 
        # 'Hall_4', 
        # 'Hall_5', 
        
        # 'Meetingroom_0', 
        # 'Meetingroom_1', 
        # 'Meetingroom_2', 
        # 'Meetingroom_3', 
        # 'Meetingroom_4', 
        # 'Meetingroom_5', 
        
        # 'Outdoor1_0', 
        # 'Outdoor1_1', 
        # 'Outdoor1_2', 
        # 'Outdoor1_3', 
        # 'Outdoor1_4', 
        # 'Outdoor1_5', 
        # 'Outdoor1_6', 
        
        # 'Outdoor2_0', 
        # 'Outdoor2_1', 
        # 'Outdoor2_2', 
        # 'Outdoor2_3', 
        # 'Outdoor2_4', 
        # 'Outdoor2_5', 
        # 'Outdoor2_6', 
        
        'HW101_W_phone0',
        'HW101_W_phone1',
        'HW101_W_phone2',
        'HW101_W_phone3',

    ]
    save_video_flag = False  # Save the video will take a lot of time, Carefully use this flag
    dataset_save_path = 'Dataset/'
    if not os.path.exists(dataset_save_path):
        os.makedirs(dataset_save_path)
    log_file = open(dataset_save_path + 'log.txt', 'a+')
    
    # parameters for preprocessing and alignment
    temperature_threshold = 300          # if there exist elements have temperature over 100 Celese degree, we discard this frame
    # expected width and height of the rgb, ira, and depth frame
    expect_size = (640, 480) 
    rgb_size_reference = (8.46, 11.28)
    alignment_coefficient = {
        'resize_ratio' : [(4.57/8.46, 5.66/11.28),(9.78/8.46, 12.4/11.28), (4.57/8.46, 5.66/11.28),(4.57/8.46, 5.66/11.28),(9.78/8.46, 12.4/11.28),(4.57/8.46, 5.66/11.28)],
        'offset': [(1.23, 2.94), (-1.3, -0.55), (1.03, 2.74) ,(1.72, 2.94), (-0.46, -0.55), (1.59, 2.74)],
        #  [(0.43, 2.94), (-0.70, -0.55), (0.23, 2.44) ,(1.32, 2.94), (-0.26, -0.55), (1.49, 2.44)],
    }

    # loading the pretrained model for labeling
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  
    # model link: https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md#coco-instance-segmentation-baselines-with-mask-r-cnn
    segmentation_predictor = DefaultPredictor(cfg)


    # Inference with a keypoint detection model
    cfg = get_cfg()   # get a fresh new config
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
    keypoints_predictor = DefaultPredictor(cfg)


    for dataset_index, data_path in enumerate(data_paths):
        experiment_name = experiment_names[dataset_index]
        # Loading the collected data file
        file = open(data_path, 'rb')
        data_dict = pickle.load(file)
        
        for sensor_index in sensor_indexes:
            print("Processing: ", data_path,  experiment_name, sensor_index)
            save_file_prefix = dataset_save_path + experiment_name + "_sensor_" + str(sensor_index)
            
            if save_video_flag:
                fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                fps = 20
                out = cv2.VideoWriter(save_file_prefix + "recording.mp4",fourcc,fps,(expect_size[0]*3, expect_size[1]))
                    
            # print(sensors_info_list)
            topic = sensors_info_list[sensor_index]['topic']
            sensors_type = sensors_info_list[sensor_index]['device_type']

            # loading the target sensor's data
            mlx_matrix, mlx_at, images, depth_maps, timestamps = dataloading(data_dict,topic)

            resize_ratio =  alignment_coefficient['resize_ratio'][sensor_index]
            offset = alignment_coefficient['offset'][sensor_index]

            mlx_temperature_maps = []
            mlx_ambient_temperatures = []
            timestamps_list = [] 
            color_images = [] # save the color images
            GT_bbox = []   # save the ground truth of the bounding box of person
            GT_depth_mask = []   # save the masked depth map of the users regions
            GT_keypoints = []   # save the ground truth of the skeleton landmarks


            invalid_fram_index = []
            for i in tqdm(range(len(mlx_matrix))):
                mat = mlx_matrix[i]
                if sensors_type == 1:
                    mat = np.flip(mat.reshape((12,16)))
                    # mat = SubpageInterpolating(np.flip(mat.reshape((12,16))))
                    # ira_expand = np.repeat(mat, expansion_coefficient * 2, 0)
                    # ira_expand = np.repeat(ira_expand, expansion_coefficient * 2, 1)
                else:
                    mat = np.flip(mat.reshape((24,32)), 0)
                    # mat = SubpageInterpolating(np.flip(mat.reshape((24,32)), 0))
                    # ira_expand = np.repeat(mat, expansion_coefficient, 0)
                    # ira_expand = np.repeat(ira_expand, expansion_coefficient, 1)
                
                # discarding the invalid frame
                if np.any(np.where(mat>temperature_threshold)):
                    invalid_fram_index.append(i)
                    continue
                
                # algining ira, depth map and the rgb image.    
                color_image = images[i]
                depth_map = depth_maps[i]
                at = mlx_at[i]
                re_rgb,re_depth = Alignment(color_image,depth_map, resize_ratio, offset, rgb_size_reference)
                algined_ira = cv2.resize(mat,expect_size)
                algined_rgb = cv2.resize(re_rgb,expect_size)
                algined_depth = cv2.resize(re_depth,expect_size)
                depth = algined_depth.astype(float)
                depth_scale = 0.0010000000474974513
                algined_depth = depth * depth_scale
                ts = timestamps[i]
                
                # mlx_temperature_maps.append(algined_ira)
                mlx_temperature_maps.append(mat)
                mlx_ambient_temperatures.append(at)
                timestamps_list.append(ts)
                color_images.append(color_image)
                
                segmentation_outputs = segmentation_predictor(algined_rgb)
                result = segmentation_outputs['instances'].to("cpu")
                temp_box = []
                
                GT_mask = np.zeros_like(algined_depth)
                for index,pred_cls in enumerate(result.pred_classes.detach().numpy().tolist()):
                    if pred_cls == 0:
                        box = result.pred_boxes.tensor.detach().numpy()[index]
                        temp_box.append(box)
                        
                        mask = result.pred_masks.detach().numpy()[index]
                        depth_mask = np.where(mask, algined_depth, 0)
                        GT_mask = GT_mask + depth_mask
                
                keypoints_outputs = keypoints_predictor(algined_rgb)
                result = keypoints_outputs['instances'].to("cpu")
                temp_kepoints = []
                for index,pred_cls in enumerate(result.pred_classes.detach().numpy().tolist()):
                    if pred_cls == 0:
                        keypoints = result.pred_keypoints.detach().numpy()[index]
                        temp_kepoints.append(keypoints)
                        
                GT_bbox.append(temp_box)
                GT_depth_mask.append(GT_mask)
                GT_keypoints.append(temp_kepoints)
                
                if save_video_flag:
                    ira_norm = (algined_ira - np.min(algined_ira))/ (np.max(algined_ira) - np.min(algined_ira)) * 255
                    temperature_colormap = cv2.applyColorMap((ira_norm).astype(np.uint8), cv2.COLORMAP_JET)
                    for box in temp_box:
                        cv2.rectangle(temperature_colormap, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
                    
                    v = Visualizer(algined_rgb[:,:,::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
                    skeleton_vis = v.draw_instance_predictions(keypoints_outputs["instances"].to("cpu"))
                    skeleton_img = skeleton_vis.get_image()[:, :, ::-1]
                    skeleton_img = cv2.resize(skeleton_img,expect_size)
                    
                    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(GT_mask*1000, alpha=0.03), cv2.COLORMAP_JET)
                    save_images = [temperature_colormap, skeleton_img, depth_colormap]
                    out.write(np.hstack(save_images))
                    
            if save_video_flag:
                out.release()
                        
            save_dict = {
                'ira_temperature_matrix': mlx_temperature_maps,
                'ira_ambient_temperature': mlx_ambient_temperatures,
                'timestamps': timestamps_list,
                'GT_bbox': GT_bbox,
                'GT_depth_mask': GT_depth_mask,
                'GT_keypoints': GT_keypoints,
                'RGB_images': color_images,
            }
            file_name = save_file_prefix + '.pickle'
            with open(file_name, 'wb') as handle:
                pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            log_file.write("Dataset: " + file_name + "\n")
            log_file.write("Num of samples: " + str(len(mlx_matrix)- len(invalid_fram_index)) + "\n")
            log_file.write("Time duration (s): " + str((timestamps_list[-1] - timestamps_list[0]).total_seconds()) + "\n")
            log_file.write("Sampling rate: " + str(len(mlx_matrix)/ ((timestamps[-1] - timestamps[0]).total_seconds())) + "\n" + "\n")
                    
            print(len(mlx_matrix))
            print(len(mlx_ambient_temperatures))
            print(len(invalid_fram_index))