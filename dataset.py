import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
import os
import pickle


class Dataset():
    def __init__(self, datapaths) -> None:
        self.ira_matrix = []
        self.ambient_temperature = []
        self.timestamps = []
        self.GT_bbox = []
        self.GT_depth = []
        self.GT_range = []
        self.GT_image = []
        self.GT_depth_mask = []
        for datapath in datapaths:
            file = open(datapath, 'rb')
            data = pickle.load(file)
            matrix = data['ira_temperature_matrix']
            at = data['ira_ambient_temperature']
            ts = data['timestamps']
            temp_bbox = data['GT_bbox'] # each element is the list of bounding box of all people for one ira frame with the format [x1,y1,x2,y2]
            depth_mask = data['GT_depth_mask'] # the depth mask for one ira frame
            
            bbox = []
            depth = []
            range_ = []
            for index, boxs in enumerate(temp_bbox): # the bounding box for one ira frame
                temp1 = []
                temp2 = []
                temp3 = []
                for box in boxs: # the bounding box for one object
                    temp1.append([int(box[0]), int(box[1]), int(box[2] - box[0]),  int(box[3] - box[1])]) # change format to [x1,y1,w,h]
                    c_x, c_y = (box[0] + box[2])/2, (box[1] + box[3])/2  # calculate the center of the bounding box
                    depth_label = depth_mask[index][int(c_y), int(c_x)] # get the depth of the center of the bounding box
                    range_label = self.dist_calculate((c_x, c_y), depth_label, FrameWidth = 640) # calculate the range based on the depth and the center of the bounding box
                    temp2.append(depth_label)
                    temp3.append(range_label)
                bbox.append(temp1)
                depth.append(temp2)
                range_.append(temp3)
            
            try:
                image = data['RGB_images']
            except:
                image = []
            file.close()
            
            self.ira_matrix += matrix
            self.ambient_temperature += at
            self.timestamps += ts
            self.GT_bbox += bbox
            self.GT_depth += depth
            self.GT_range += range_
            self.GT_image += image
            self.GT_depth_mask += depth_mask
    
    def GetSample(self, index):
        return self.ira_matrix[index], self.ambient_temperature[index], self.timestamps[index], self.GT_bbox[index], self.GT_depth[index], self.GT_range[index], self.GT_image[index], self.GT_depth_mask[index]
    
    def GetAllSamples(self):
        return self.ira_matrix, self.ambient_temperature, self.timestamps, self.GT_bbox, self.GT_depth, self.GT_range, self.GT_image, self.GT_depth_mask
    
    def len(self):
        return min(len(self.ira_matrix), len(self.ambient_temperature), len(self.timestamps), len(self.GT_bbox), len(self.GT_depth), len(self.GT_range), len(self.GT_image), len(self.GT_depth_mask))
    
    def dist_calculate(self, center_points, depth, FrameWidth):  # calculating the distance/range based on the center point of the bbox and the depth value
        H_FOV = 87  # the horizontal field of view of the depth camera
        half_H_FOV_rad = np.deg2rad(H_FOV/2)
        x_c,y_c = center_points
        horizontal_shift_pixels = abs(x_c - FrameWidth/2)
        shift_distance = np.tan(half_H_FOV_rad) * depth * (horizontal_shift_pixels / (FrameWidth/2))
        range_ = np.sqrt(depth**2 + shift_distance**2)
        return range_



# class Old_Dataset():
#     def __init__(self,dataset_folders, roi_pooling = None) -> None:
#         if roi_pooling is None:
#             print("Please provide the ROI Pooling.")
#             return 0
        
#         self.ROI = []
#         self.ROI_depth_label = []
#         self.ROI_range_label = []
        
#         for index,dataset_folder in enumerate(dataset_folders):
#             # print("File " + str(index) + " of " + str(len(dataset_folders)) + " Files.")
#             print(dataset_folder + "Stage1Result_ROI.pickle")
#             with open(dataset_folder + "Stage1Result_ROI.pickle", 'rb') as file:
#                 raw_ROI = pickle.load(file)
#                 for r in tqdm(raw_ROI):
#                     pooled_roi = roi_pooling.PoolingNumpy(r)
#                     self.ROI.append(pooled_roi)
#             self.ROI_depth_label.append(np.load(dataset_folder + "Stage1Result_ROI_depth_label.npy"))
#             self.ROI_range_label.append(np.load(dataset_folder + "Stage1Result_ROI_range_label.npy"))
#         self.data = np.stack(self.ROI)
#         self.range_label = np.concatenate(self.ROI_depth_label, axis = 0)
#         self.depth_label = np.concatenate(self.ROI_range_label, axis = 0)
        
    
#     def GetSortData(self):
#         num_samples = len(self.range_label)
#         flat_data = np.reshape(self.data, (num_samples, -1))
#         sort_flat_data = np.sort(flat_data, axis=-1)[:,::-1]
#         return sort_flat_data, self.range_label, self.depth_label
    
#     def GetPooledData(self):
#         return self.data, self.range_label, self.depth_label
    
#     def GetDataPandasType(self):
#         num_samples = len(self.range_label)
#         flat_data = np.reshape(self.data, (num_samples, -1))
#         sort_flat_data = np.sort(flat_data, axis=-1)[:,::-1]
#         feature_names = []
#         for i in range(sort_flat_data.shape[1]):
#             feature_names.append('top_' + str(i))

#         dataset = pd.DataFrame(sort_flat_data, columns = feature_names)
#         dataset['range_label'] = self.range_label
#         return dataset
        
    

if __name__ == "__main__":
    datapaths = [
        'Dataset/set2_sensor_1.pickle',
        'Dataset/set2_sensor_4.pickle',
    ]

    dataset = Dataset(datapaths)