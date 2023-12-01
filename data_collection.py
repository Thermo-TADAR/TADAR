
from tkinter import OFF
import paho.mqtt.client as mqtt
import time
import sys
import pickle
import numpy as np
import os
import argparse
import ast
import random
import cv2
import pyrealsense2 as rs
import os
import pandas as pd
import datetime
from functions2 import *
import math


from  data_sensor_config import sensors_info_list


def Alignment(rgb_frame, resize_ratio, offset, rgb_size_reference):
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
        # re_depth = cv2.copyMakeBorder(depth_frame,top, bottom, left, right, cv2.BORDER_CONSTANT, value= (255,255,255))
    else:
        re_rgb = rgb_frame[offset_height: offset_height+height , offset_width: offset_width+width, :]
        # re_depth = depth_frame[offset_height: offset_height+height , offset_width: offset_width+width, :]
    return re_rgb 


if __name__ == "__main__":
    """
    Usage:
        collecting data: python data_collection.py -sl 0 1 2 3 4 5 -sp Storage/Path/ -sf 1
        Checking sensor: python data_collection.py -sl 0 1 2 3 4 5 -sp Storage/Path/ -sf 0
    """
    parser = argparse.ArgumentParser(description='Ranging dataset construction')
    parser.add_argument('-sl','--sensor_list', nargs='+',default='0',  help='the selected sensors')
    parser.add_argument('-sp','--storage_path', type=str,default='New_ex/', required=True,  help='the storage path of this experiment')
    parser.add_argument('-sf','--storage_flag', type=int,  help='>0: save data or 0:not')
    args = parser.parse_args()
    sensor_list = [int(ele) for ele in args.sensor_list]
    storage_path = args.storage_path
    if args.storage_flag ==0:
        storage_flag = False
    else:
        storage_flag = True

    print("Storage Flag:", storage_flag)
    if storage_flag:
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)

    Visalization_flag = True   # false means no visualization video
    expansion_coefficient = 20    # this is determinated by the size of the realsense frame

    expect_size = (320*2, 240*2)  # expected width and height of the rgb, ira, and depth frame
    rgb_size_reference = (8.46, 11.28)
    alignment_coefficient = {
        'resize_ratio' : [(4.57/8.46, 5.66/11.28),(9.78/8.46, 12.4/11.28), (4.57/8.46, 5.66/11.28),(4.57/8.46, 5.66/11.28),(9.78/8.46, 12.4/11.28),(4.57/8.46, 5.66/11.28)],
        'offset': [(1.23, 2.94), (-1.3, -0.55), (1.03, 2.74) ,(1.72, 2.94), (-0.46, -0.55), (1.59, 2.74)],
        #  [(0.43, 2.94), (-0.70, -0.55), (0.23, 2.44) ,(1.32, 2.94), (-0.26, -0.55), (1.49, 2.44)],
    }

    # MLX config ----------------------------------------------------------
    username = 'mqtt'
    password = '1234'
    mqttBroker = '127.0.0.1'
    mqttPort = 18839

    sensor_data_storage_path = storage_path + "MLX/"

    selected_sensor_info = []
    for i in sensor_list:
        selected_sensor_info.append(sensors_info_list[i])

    # realsense config -------------------------------------------------------
    if storage_flag:
        Realsense_depth_storage_path = storage_path + "Realsense/"
        images_storage_path = storage_path + "images/"
        if not os.path.exists(sensor_data_storage_path):
            os.makedirs(sensor_data_storage_path)
        if not os.path.exists(Realsense_depth_storage_path):
            os.makedirs(Realsense_depth_storage_path)
        if not os.path.exists(images_storage_path):
            os.makedirs(images_storage_path)

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))
    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Running ######################################################################

    # recordings
    frame_ids = []
    MLX_sensor_data = []
    for s in selected_sensor_info:
        MLX_sensor_data.append(
            {
                "sensor_topic": s['topic'],
                'onboard_timestamp': [],
                "temperature_matrix_path": [],
                "ambient_temperature": [],
                'new_data_flag': [],
                'timestamp': [],
            }
        )
        frame_ids.append(0)

    Realsense_depth_data = []
    images_data = []
    # Start streamin -------------------------------------------------
    pipeline.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to)

    sensors = []
    sensor_types = []
    for sensor_info in selected_sensor_info:
        sensor = Sensor(mqtt_brocker=mqttBroker,mqtt_port=mqttPort,username=username,pw=password,topic=sensor_info['topic'],sensor_type=sensor_info['device_type'])
        sensors.append(sensor)
        sensor_types.append(sensor_info['device_type'])

    for sensor in sensors:
        sensor.connect_mqtt()
        sensor.subscribe_topic()
        sensor.run()

    empty = cv2.applyColorMap(np.ones((32,24)).astype(np.uint8), cv2.COLORMAP_JET)
    empty_colored = cv2.resize(empty, (32* expansion_coefficient, 24*expansion_coefficient))

    frame_index = 0
    if Visalization_flag:
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        fps = 20
        num_rows = math.ceil(len(sensor_list) / 3)+ 1  
        out = cv2.VideoWriter(storage_path + "recording.mp4",fourcc,fps,(expect_size[0]*3, expect_size[1]*num_rows))

    try:
        while True:
            MLX_Realsense_data = []
            data_s = []
            for index,sensor in enumerate(sensors):
                data = sensor.get_data()
                # save:
                if storage_flag:
                    temp_path = sensor_data_storage_path + sensor.client_name + "_" + str(frame_index) + ".npy"
                    np.save(temp_path, data['Detected_temperature'])
                    MLX_sensor_data[index]['temperature_matrix_path'].append(temp_path)
                    MLX_sensor_data[index]['ambient_temperature'].append(data["Ambient_temperature"])
                    MLX_sensor_data[index]['new_data_flag'].append(data["data_flag"])
                    MLX_sensor_data[index]['onboard_timestamp'].append(data["Onboard_timestamp"])
                    MLX_sensor_data[index]['timestamp'].append(datetime.datetime.now())
                sensor.set_data_flag()  # set the new data flag to Fasle since we already used the data.

                if data["data_flag"]:
                    frame_ids[index] = frame_ids[index] + 1

                sensor_type = sensor_types[index]
                if Visalization_flag:
                    if sensor_type == 1:
                        mat = data['Detected_temperature'].reshape((12,16))
                        mat = SubpageInterpolating(mat)
                        ira_expand = np.repeat(mat, expansion_coefficient * 2, 0)
                        ira_expand = np.repeat(ira_expand, expansion_coefficient * 2, 1)
                    else:
                        mat = data['Detected_temperature'].reshape((24,32))
                        mat = SubpageInterpolating(mat)
                        ira_expand = np.repeat(mat, expansion_coefficient, 0)
                        ira_expand = np.repeat(ira_expand, expansion_coefficient, 1)
                    ira_norm = (ira_expand - np.min(ira_expand))/ (np.max(ira_expand) - np.min(ira_expand)) * 255
                    temperature_colormap = cv2.applyColorMap((np.flip(ira_norm,0)).astype(np.uint8), cv2.COLORMAP_JET)
                    data_s.append(temperature_colormap)

            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            for i in range(len(data_s)):
                j = sensor_list[i]
                resize_ratio =  alignment_coefficient['resize_ratio'][j]
                offset = alignment_coefficient['offset'][j]
                re_rgb = Alignment(color_image, resize_ratio, offset, rgb_size_reference)
                ira_frame = data_s[i]
                algined_ira = cv2.resize(ira_frame,expect_size)
                algined_rgb = cv2.resize(re_rgb,expect_size)
                overlapping = cv2.addWeighted(algined_rgb, 1, algined_ira, 0.8, 0)
                data_s[i] = overlapping

            ################################################### save depth:
            if storage_flag:
                temp_path = Realsense_depth_storage_path + "depth_" + str(frame_index) + ".npy"
                Realsense_depth_data.append(temp_path)
                np.save(temp_path, depth_image)

                temp_path = images_storage_path + "color_" + str(frame_index) + ".png"
                cv2.imwrite(temp_path, color_image)
                images_data.append(temp_path)

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            frame_index = frame_index +1

            # timestamps.append(datetime.datetime.now())
            if Visalization_flag:
                num_sensor = len(data_s)
                rows = []
                for i in range(num_sensor//3):
                    temp = np.hstack(data_s[3*i:3*(i+1)])
                    rows.append(temp)
                if num_sensor%3 != 0:
                    temp_list = []
                    if num_sensor%3 ==1:
                        temp_list += data_s[3*(num_sensor//3):]
                        temp_list.append(cv2.resize(empty_colored,expect_size))
                        temp_list.append(cv2.resize(empty_colored,expect_size))
                        temp = np.hstack(temp_list)
                        # print("len",len(temp_list))
                    else:
                        temp_list += data_s[3*(num_sensor//3):]
                        temp_list.append(cv2.resize(empty_colored,expect_size))
                        temp = np.hstack(temp_list)
                    rows.append(temp)
                gt_row = np.hstack((cv2.resize(color_image,expect_size)  , cv2.resize(depth_colormap,expect_size) , cv2.resize(empty_colored,expect_size) ))
                # print("gt row:", gt_row.shape)
                rows.append(gt_row)
                all_images = np.vstack(rows)
                cv2.namedWindow('All', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('All', all_images)
                out.write(all_images)
                key = cv2.waitKey(1)
                if key == 27 or key == 113:
                    break
        if Visalization_flag:
            cv2.destroyAllWindows()
            out.release()

        if storage_flag:
            save_dic = {"image_path": images_data, "depth_path":Realsense_depth_data,}
            for ele in MLX_sensor_data:
                save_dic[ele['sensor_topic']+"_mt_path"] = ele["temperature_matrix_path"]
                save_dic[ele['sensor_topic']+"_at"] = ele["ambient_temperature"]
                save_dic[ele['sensor_topic']+"_onboard_ts"] = ele["onboard_timestamp"]
                save_dic[ele['sensor_topic']+"_new_data_flag"] = ele["new_data_flag"]
                save_dic[ele['sensor_topic']+"_timestamp"] = ele["timestamp"]
            df = pd.DataFrame(save_dic)
            storage_file_name = storage_path + 'data' + '.pkl'
            df.to_pickle(storage_file_name)
    except KeyboardInterrupt:
        if storage_flag:
            save_dic = {"image_path": images_data, "depth_path":Realsense_depth_data,}
            for ele in MLX_sensor_data:
                save_dic[ele['sensor_topic']+"_mt_path"] = ele["temperature_matrix_path"]
                save_dic[ele['sensor_topic']+"_at"] = ele["ambient_temperature"]
                save_dic[ele['sensor_topic']+"_onboard_ts"] = ele["onboard_timestamp"]
                save_dic[ele['sensor_topic']+"_new_data_flag"] = ele["new_data_flag"]
                save_dic[ele['sensor_topic']+"_timestamp"] = ele["timestamp"]
            df = pd.DataFrame(save_dic)
            storage_file_name = storage_path + 'data' + '.pkl'
            df.to_pickle(storage_file_name)
        sys.exit(0)

