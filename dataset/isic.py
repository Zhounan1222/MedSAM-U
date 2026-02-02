import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import random

from utils import random_box, random_click,random_click_3point
import json



def generate_random_box_with_iou(gt_box, img_shape, target_iou=0.5, max_attempts=10000000):

    def calculate_iou(box1, box2):
        """
        计算两个边界框之间的 IoU 值。
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)

        iou = inter_area / (box1_area + box2_area - inter_area)
        return iou

    h, w = img_shape
    x_min, y_min, x_max, y_max = gt_box
    gt_area = (x_max - x_min) * (y_max - y_min)

    attempts = 0
    while attempts < max_attempts:
         # 随机生成一个宽度和高度
        rand_w = random.randint(max(1, int(0.5 * (x_max - x_min))), min(w, int(1.5 * (x_max - x_min))))
        rand_h = random.randint(max(1, int(0.5 * (y_max - y_min))), min(h, int(1.5 * (y_max - y_min))))

        # 检查生成的宽度和高度是否在合理范围内
        if rand_w >= w or rand_h >= h:
            continue

        # 随机生成左上角的点，确保整个框在图像范围内
        rand_x_min = random.randint(0, w - rand_w)
        rand_y_min = random.randint(0, h - rand_h)
        rand_x_max = rand_x_min + rand_w
        rand_y_max = rand_y_min + rand_h
        

        rand_box = (rand_x_min, rand_y_min, rand_x_max, rand_y_max)

        # 计算这个随机框与 GT 的 IoU
        iou = calculate_iou(gt_box, rand_box)
        # 如果 IoU 满足条件，返回这个框
        if abs(iou - target_iou) < 0.05:  # 允许一定的误差
            return np.array([rand_x_min, rand_y_min, rand_x_max, rand_y_max])
            # return rand_box

        attempts += 1

    raise ValueError(f"在 {max_attempts} 次尝试后未能找到满足 IoU={target_iou} 的随机框")


class ISIC2016(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Training',prompt = 'click', plane = False):

        df = pd.read_csv(os.path.join(data_path, 'ISBI2016_ISIC_Part1_' + mode + '_GroundTruth.csv'), encoding='gbk')
        self.name_list = df.iloc[:,1].tolist()
        self.label_list = df.iloc[:,2].tolist()
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        # if self.mode == 'Training':
        #     point_label = random.randint(0, 1)
        #     inout = random.randint(0, 1)
        # else:
        #     inout = 1
        #     point_label = 1
        point_label = 1

        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, name)
        
        mask_name = self.label_list[index]
        msk_path = os.path.join(self.data_path, mask_name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')

        # if self.mode == 'Training':
        #     label = 0 if self.label_list[index] == 'benign' else 1
        # else:
        #     label = int(self.label_list[index])

        newsize = (self.img_size, self.img_size)
        mask = mask.resize(newsize)

        if self.prompt == 'click':
            point_label, pt = random_click(np.array(mask) / 255, point_label)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)


            if self.transform_msk:
                mask = self.transform_msk(mask).int()
                
            # if (inout == 0 and point_label == 1) or (inout == 1 and point_label == 0):
            #     mask = 1 - mask
        name = name.split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj':name}


        
        return {
            'image':img,
            'label': mask,
            'p_label':point_label,
            'pt':pt,
            'image_meta_dict':image_meta_dict,
        }



class ISIC2017(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Training',prompt = 'click', plane = False):

        df = pd.read_csv(os.path.join(data_path, 'ISBI2017_ISIC_Part1_' + mode + '_GroundTruth.csv'), encoding='gbk')
        self.name_list = df.iloc[:,0].tolist()
        self.label_list = df.iloc[:,1].tolist()
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        # if self.mode == 'Training':
        #     point_label = random.randint(0, 1)
        #     inout = random.randint(0, 1)
        # else:
        #     inout = 1
        #     point_label = 1
        point_label = 1

        """Get the images"""
        name = self.name_list[index]
        # img_path = os.path.join(self.data_path, name)
        img_path = name
        
        mask_name = self.label_list[index]
        # msk_path = os.path.join(self.data_path, mask_name)
        msk_path = mask_name

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')

        # if self.mode == 'Training':
        #     label = 0 if self.label_list[index] == 'benign' else 1
        # else:
        #     label = int(self.label_list[index])

        newsize = (self.img_size, self.img_size)
        mask = mask.resize(newsize)

        if self.prompt == 'click':
            point_label, pt = random_click(np.array(mask) / 255, point_label)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)


            if self.transform_msk:
                mask = self.transform_msk(mask).int()
                
            # if (inout == 0 and point_label == 1) or (inout == 1 and point_label == 0):
            #     mask = 1 - mask
        name = name.split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj':name}


        
        return {
            'image':img,
            'label': mask,
            'p_label':point_label,
            'pt':pt,
            'image_meta_dict':image_meta_dict,
        }



class ISIC2017_pointbox_noise(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Train',prompt = 'point&box', plane = False):

        df = pd.read_csv(os.path.join(data_path, 'ISBI2017_ISIC_Part1_' + mode + '_GroundTruth.csv'), encoding='gbk')
        self.name_list = df.iloc[:,0].tolist()
        self.label_list = df.iloc[:,1].tolist()
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.device = args.gpu_device

        self.transform = transform
        self.transform_msk = transform_msk

        # self.coords = json.load(open(coords_file, "r"))

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        # if self.mode == 'Training':
        #     point_label = random.randint(0, 1)
        #     inout = random.randint(0, 1)
        # else:
        #     inout = 1
        #     point_label = 1
        point_label = 1

        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, name)
        
        mask_name = self.label_list[index]
        msk_path = os.path.join(self.data_path, mask_name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')

        # 将图像转换为NumPy数组
        img_array = np.array(img, dtype=np.float32)
        
        # 生成高斯噪声
        mean = 0.5
        sigma = 0.05*255  # 标准差（你可以调整这个值来控制噪声水平）
        noise = np.random.normal(mean, sigma, img_array.shape)
        
        # 将高斯噪声添加到图像中
        noisy_img_array = img_array + noise
        
        # 确保值在 [0, 255] 范围内
        noisy_img_array = np.clip(noisy_img_array, 0, 255)
        
        # 转换回PIL图像
        noisy_img = Image.fromarray(noisy_img_array.astype(np.uint8))


        mask_array = np.where((np.array(mask)) > 245, 0, 1)
        mask_array = mask_array.astype(np.uint8)
        mask = Image.fromarray(mask_array * 255)

        # if self.mode == 'Training':
        #     label = 0 if self.label_list[index] == 'benign' else 1
        # else:
        #     label = int(self.label_list[index])

        newsize = (self.img_size, self.img_size)
        mask = mask.resize(newsize)

        if self.prompt == 'point&box':
            boxes_list = []
            point_coords_list, point_labels_list = [], []
            point_label, pt = random_click_3point(np.array(mask) / 255, point_label)
            

            col, row = np.nonzero(mask)
            prompt = np.array([row.min(), col.min(), row.max(), col.max()])

            for i in range(3):

                # only_box = generate_random_box_with_iou(prompt,newsize)
                only_box =prompt_aug(prompt,mask,True)

                boxes_list.append(only_box)
        #     point_coords_list.append(point_coords)
        #     point_labels_list.append(point_label)
        # boxes = torch.stack(boxes_list, dim=0)
        # point_coords = torch.stack(point_coords_list, dim=0)
        # point_labels = torch.stack(point_labels_list, dim=0)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            img_noise = self.transform(noisy_img)
            torch.set_rng_state(state)


            if self.transform_msk:
                mask = self.transform_msk(mask).int()
                
            # if (inout == 0 and point_label == 1) or (inout == 1 and point_label == 0):
            #     mask = 1 - mask
        name = name.split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj':name}
        return {
            'image_raw':img,
            'image_lq':img_noise,
            'image_noise':img_noise,
            'label': mask,
            'p_label':point_label,
            'pt':pt,
            'box':boxes_list,
            'image_meta_dict':image_meta_dict,
        }


class ISIC2017_pointbox(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Train',prompt = 'point&box', plane = False):

        df = pd.read_csv(os.path.join(data_path, 'ISBI2017_ISIC_Part1_' + mode + '_GroundTruth.csv'), encoding='gbk')
        self.name_list = df.iloc[:,0].tolist()
        self.label_list = df.iloc[:,1].tolist()
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.device = args.gpu_device

        self.transform = transform
        self.transform_msk = transform_msk

        # self.coords = json.load(open(coords_file, "r"))

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        # if self.mode == 'Training':
        #     point_label = random.randint(0, 1)
        #     inout = random.randint(0, 1)
        # else:
        #     inout = 1
        #     point_label = 1
        point_label = 1

        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, name)
        
        mask_name = self.label_list[index]
        msk_path = os.path.join(self.data_path, mask_name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')

      


        mask_array = np.where((np.array(mask)) > 245, 0, 1)
        mask_array = mask_array.astype(np.uint8)
        mask = Image.fromarray(mask_array * 255)

        newsize = (self.img_size, self.img_size)
        mask = mask.resize(newsize)

        if self.prompt == 'point&box':
            boxes_list = []
            point_coords_list, point_labels_list = [], []
            point_label, pt = random_click(np.array(mask) / 255, point_label)
            

            col, row = np.nonzero(mask)
            prompt = np.array([row.min(), col.min(), row.max(), col.max()])

            for i in range(1):

                # only_box = generate_random_box_with_iou(prompt,newsize)
                only_box =prompt_aug(prompt,mask,True)

        

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)


            if self.transform_msk:
                mask = self.transform_msk(mask).int()
                
            # if (inout == 0 and point_label == 1) or (inout == 1 and point_label == 0):
            #     mask = 1 - mask
        name = name.split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'label': mask,
            'p_label':point_label,
            'pt':pt,
            'box':boxes_list,
            'image_meta_dict':image_meta_dict,
        }

class ISIC2017_pointbox_uncertainty(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Train',prompt = 'point&box', plane = False):

        df = pd.read_csv(os.path.join(data_path, 'ISBI2017_ISIC_Part1_' + mode + '_GroundTruth.csv'), encoding='gbk')
        self.name_list = df.iloc[:,0].tolist()
        self.label_list = df.iloc[:,1].tolist()
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.device = args.gpu_device

        self.transform = transform
        self.transform_msk = transform_msk

        # self.coords = json.load(open(coords_file, "r"))

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        # if self.mode == 'Training':
        #     point_label = random.randint(0, 1)
        #     inout = random.randint(0, 1)
        # else:
        #     inout = 1
        #     point_label = 1
        point_label = 1

        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, name)
        
        mask_name = self.label_list[index]
        msk_path = os.path.join(self.data_path, mask_name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')

        newsize = (self.img_size, self.img_size)


        # gaussian_noise_sigma = 0.05
        # noise_add = np.random.normal(0, gaussian_noise_sigma * 255, newsize)
        # # input = input + noise_add
        # img = img+noise_add

        # if self.mode == 'Training':
        #     label = 0 if self.label_list[index] == 'benign' else 1
        # else:
        #     label = int(self.label_list[index])

        # newsize = (self.img_size, self.img_size)
        mask = mask.resize(newsize)

        if self.prompt == 'point&box':
            boxes_list = []
            point_coords_list, point_labels_list = [], []
            # point_label, pt = random_click_3point(np.array(mask) / 255, point_label)
            

            col, row = np.nonzero(mask)
            prompt = np.array([row.min(), col.min(), row.max(), col.max()])

            for i in range(3):

                # only_box = generate_random_box_with_iou(prompt,newsize)
                only_box =generate_random_box_with_iou(prompt,newsize)
                # only_box = prompt_aug(prompt,mask,True)
                boxes_list.append(only_box)
        #     point_coords_list.append(point_coords)
        #     point_labels_list.append(point_label)
        # boxes = torch.stack(boxes_list, dim=0)
        # point_coords = torch.stack(point_coords_list, dim=0)
        # point_labels = torch.stack(point_labels_list, dim=0)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)


            if self.transform_msk:
                mask = self.transform_msk(mask).int()
                
            # if (inout == 0 and point_label == 1) or (inout == 1 and point_label == 0):
            #     mask = 1 - mask
        name = name.split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'label': mask,
            'box':boxes_list,
            'image_meta_dict':image_meta_dict,
        }




def generate_random_box_with_iou(gt_box, img_shape, target_iou=0.75, max_attempts=10000000):
    """
    生成一个与给定 GT 边界框有指定 IoU 的随机框。

    :param gt_box:  GT 边界框的坐标，格式为 (x_min, y_min, x_max, y_max)。
    :param img_shape: 图像的形状 (高度, 宽度)。
    :param target_iou: 目标 IoU 值，默认值为 0.5。
    :param max_attempts: 尝试生成满足条件的边界框的最大次数，默认值为 100 次。
    :return: 满足条件的随机边界框的坐标，格式为 (x_min, y_min, x_max, y_max)。
    """

    def calculate_iou(box1, box2):
        """
        计算两个边界框之间的 IoU 值。

        :param box1: 第一个边界框，格式为 (x_min, y_min, x_max, y_max)。
        :param box2: 第二个边界框，格式为 (x_min, y_min, x_max, y_max)。
        :return: IoU 值。
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)

        iou = inter_area / (box1_area + box2_area - inter_area)
        return iou

    h, w = img_shape
    x_min, y_min, x_max, y_max = gt_box
    gt_area = (x_max - x_min) * (y_max - y_min)

    attempts = 0
    while attempts < max_attempts:
         # 随机生成一个宽度和高度
        rand_w = random.randint(max(1, int(0.5 * (x_max - x_min))), min(w, int(1.5 * (x_max - x_min))))
        rand_h = random.randint(max(1, int(0.5 * (y_max - y_min))), min(h, int(1.5 * (y_max - y_min))))

        # 检查生成的宽度和高度是否在合理范围内
        if rand_w >= w or rand_h >= h:
            continue

        # 随机生成左上角的点，确保整个框在图像范围内
        rand_x_min = random.randint(0, w - rand_w)
        rand_y_min = random.randint(0, h - rand_h)
        rand_x_max = rand_x_min + rand_w
        rand_y_max = rand_y_min + rand_h
        

        rand_box = (rand_x_min, rand_y_min, rand_x_max, rand_y_max)

        # 计算这个随机框与 GT 的 IoU
        iou = calculate_iou(gt_box, rand_box)
        # 如果 IoU 满足条件，返回这个框
        if abs(iou - target_iou) < 0.05:  # 允许一定的误差
            return np.array([rand_x_min, rand_y_min, rand_x_max, rand_y_max])
            # return rand_box

        attempts += 1

    raise ValueError(f"在 {max_attempts} 次尝试后未能找到满足 IoU={target_iou} 的随机框")

class ISIC2017_box(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Training',prompt = 'point&box', plane = False):

        df = pd.read_csv(os.path.join(data_path, 'ISBI2017_ISIC_Part1_' + mode + '_GroundTruth.csv'), encoding='gbk')
        self.name_list = df.iloc[:,0].tolist()
        self.label_list = df.iloc[:,1].tolist()
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.device = args.gpu_device

        self.transform = transform
        self.transform_msk = transform_msk

        # self.coords = json.load(open(coords_file, "r"))

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        # if self.mode == 'Training':
        #     point_label = random.randint(0, 1)
        #     inout = random.randint(0, 1)
        # else:
        #     inout = 1
        #     point_label = 1
        # point_label = 1

        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, name)
        
        mask_name = self.label_list[index]
        msk_path = os.path.join(self.data_path, mask_name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')

        # if self.mode == 'Training':
        #     label = 0 if self.label_list[index] == 'benign' else 1
        # else:
        #     label = int(self.label_list[index])

        newsize = (self.img_size, self.img_size)
        mask = mask.resize(newsize)

        # if self.prompt == 'box':
        boxes_list = []
        point_coords_list, point_labels_list = [], []
        # point_label, pt = random_click(np.array(mask) / 255, point_label)
        

        col, row = np.nonzero(mask)
        prompt = np.array([row.min(), col.min(), row.max(), col.max()])

        for i in range(3):

            only_box = generate_random_box_with_iou(prompt,newsize)

            boxes_list.append(only_box)
        #     point_coords_list.append(point_coords)
        #     point_labels_list.append(point_label)
        # boxes = torch.stack(boxes_list, dim=0)
        # point_coords = torch.stack(point_coords_list, dim=0)
        # point_labels = torch.stack(point_labels_list, dim=0)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)


            if self.transform_msk:
                mask = self.transform_msk(mask).int()
                
            # if (inout == 0 and point_label == 1) or (inout == 1 and point_label == 0):
            #     mask = 1 - mask
        name = name.split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'label': mask,
            'box':boxes_list,
            'image_meta_dict':image_meta_dict,
        }




class ISIC2017_05box(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Training',prompt = 'point&box', plane = False):

        df = pd.read_csv(os.path.join(data_path, 'ISBI2017_ISIC_Part1_' + mode + '_GroundTruth.csv'), encoding='gbk')
        self.name_list = df.iloc[:,0].tolist()
        self.label_list = df.iloc[:,1].tolist()
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.device = args.gpu_device

        self.transform = transform
        self.transform_msk = transform_msk

        # self.coords = json.load(open(coords_file, "r"))

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        # if self.mode == 'Training':
        #     point_label = random.randint(0, 1)
        #     inout = random.randint(0, 1)
        # else:
        #     inout = 1
        #     point_label = 1
        # point_label = 1

        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, name)
        
        mask_name = self.label_list[index]
        msk_path = os.path.join(self.data_path, mask_name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')

        # if self.mode == 'Training':
        #     label = 0 if self.label_list[index] == 'benign' else 1
        # else:
        #     label = int(self.label_list[index])

        newsize = (self.img_size, self.img_size)
        mask = mask.resize(newsize)

        # if self.prompt == 'box':
        boxes_list = []
        point_coords_list, point_labels_list = [], []
        # point_label, pt = random_click(np.array(mask) / 255, point_label)
        

        col, row = np.nonzero(mask)
        prompt = np.array([row.min(), col.min(), row.max(), col.max()])

        for i in range(3):

            only_box = generate_random_box_with_iou(prompt,newsize)

            boxes_list.append(only_box)
        #     point_coords_list.append(point_coords)
        #     point_labels_list.append(point_label)
        # boxes = torch.stack(boxes_list, dim=0)
        # point_coords = torch.stack(point_coords_list, dim=0)
        # point_labels = torch.stack(point_labels_list, dim=0)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)


            if self.transform_msk:
                mask = self.transform_msk(mask).int()
                
            # if (inout == 0 and point_label == 1) or (inout == 1 and point_label == 0):
            #     mask = 1 - mask
        name = name.split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'label': mask,
            'box':boxes_list,
            'image_meta_dict':image_meta_dict,
        }



def prompt_aug(prompt, target, aug=True):
    target_h, target_w = target.size
    scale_factor = 128
    if aug:
        left_x = prompt[0] - 20 + random.randint(-target_w // scale_factor, target_w // scale_factor)
        left_y = prompt[1] - 20 + random.randint(-target_h // scale_factor, target_h // scale_factor)
        right_x = prompt[2]+ 20+ random.randint(-target_w // scale_factor, target_w // scale_factor)
        right_y = prompt[3]+ 20 + random.randint(-target_h // scale_factor, target_h // scale_factor)
    else:
        left_x = prompt[0]
        left_y = prompt[1]
        right_x = prompt[2]
        right_y = prompt[3]
    if left_x < 0:
        left_x = 0
    elif left_x > target_w:
        left_x = target_w
    else:
        pass
    if left_y < 0:
        left_y = 0
    elif left_y > target_h:
        left_y = target_h
    else:
        pass
    if right_x < 0:
        right_x = 0
    elif right_x > target_w:
        right_x = target_w
    else:
        pass
    if right_y < 0:
        right_y = 0
    elif right_y > target_h:
        right_y = target_h
    else:
        pass

    return np.array([left_x, left_y, right_x, right_y])


class ISIC_point(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Train',prompt = 'point&box', plane = False):

        df = pd.read_csv(os.path.join(data_path, 'ISBI2017_ISIC_Part1_' + mode + '_GroundTruth.csv'), encoding='gbk')
        self.name_list = df.iloc[:,0].tolist()
        self.label_list = df.iloc[:,1].tolist()
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.device = args.gpu_device

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        # if self.mode == 'Training':
        #     point_label = random.randint(0, 1)
        #     inout = random.randint(0, 1)
        # else:
        #     inout = 1
        #     point_label = 1
        point_label = 1

        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, name)
        
        mask_name = self.label_list[index]
        msk_path = os.path.join(self.data_path, mask_name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')


        # resize raters images for generating initial point click
        newsize = (self.img_size, self.img_size)



        # mask_tensor =  Image.fromarray(mask.astype(np.uint8))  # 转换为uint8类型并生成PIL图像
        # resized_image = mask_tensor.resize((1024, 1024), resample=Image.Resampling.LANCZOS)
        # mask = np.array(resized_image)
        # mask_size = mask.resize(newsize)
        mask_numpy = np.array(mask.resize(newsize))

        # first click is the target agreement among most raters
        if self.prompt == 'point':
            point_label, pt = random_click(mask_numpy / 255, point_label)
        elif self.prompt == '3point':
            point_label, pt = random_click_3point(mask_numpy / 255, point_label)
            # point_label, pt_disc = random_click(np.array(np.mean(np.stack(multi_rater_disc_np), axis=0)) / 255, point_label)
        
            
        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            mask = torch.as_tensor((self.transform_msk(mask) >0.5).float(), dtype=torch.float32)
            # transform to mask size (out_size) for mask define
            # mask = F.interpolate(mask, size=(self.mask_size, self.mask_size), mode='bilinear', align_corners=False).mean(dim=0)

            
            torch.set_rng_state(state)
            

       

        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'label': mask,
            'p_label':point_label,
            'pt':pt,
            'image_meta_dict':image_meta_dict,
        }
    

class ISIC2016_pointbox(Dataset):
    def __init__(self, args, data_path ,coords_file, transform = None, transform_msk = None, mode = 'Training',prompt = 'point&box', plane = False):

        df = pd.read_csv(os.path.join(data_path, 'ISBI2016_ISIC_Part1_' + mode + '_GroundTruth.csv'), encoding='gbk')
        self.name_list = df.iloc[:,1].tolist()
        self.label_list = df.iloc[:,2].tolist()
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size

        self.transform = transform
        self.transform_msk = transform_msk

        self.coords = json.load(open(coords_file, "r"))

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        # if self.mode == 'Training':
        #     point_label = random.randint(0, 1)
        #     inout = random.randint(0, 1)
        # else:
        #     inout = 1
        #     point_label = 1
        point_label = 1

        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, name)
        
        mask_name = self.label_list[index]
        msk_path = os.path.join(self.data_path, mask_name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')

        # if self.mode == 'Training':
        #     label = 0 if self.label_list[index] == 'benign' else 1
        # else:
        #     label = int(self.label_list[index])

        newsize = (self.img_size, self.img_size)
        mask = mask.resize(newsize)

        if self.prompt == 'point&box':
            boxes_list = []
            point_coords_list, point_labels_list = [], []
            point_label, pt = random_click(np.array(mask) / 255, point_label)
            for i in range(self.coords):
                if self.coords[i]['image_path'] == self.image_paths[index]:
                    boxes = torch.tensor(self.coords[i]['boxes'], dtype=torch.int64)
                    point_coords = torch.tensor(self.coords[i]['point'], dtype=torch.int64)
                    point_label = torch.tensor(self.coords[i]['label'], dtype=torch.int64)  # 假设所有点的标签为1
                else:
                    continue
            boxes_list.append(boxes)
            point_coords_list.append(point_coords)
            point_labels_list.append(point_label)
        boxes = torch.stack(boxes_list, dim=0)
        point_coords = torch.stack(point_coords_list, dim=0)
        point_labels = torch.stack(point_labels_list, dim=0)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)


            if self.transform_msk:
                mask = self.transform_msk(mask).int()
                
            # if (inout == 0 and point_label == 1) or (inout == 1 and point_label == 0):
            #     mask = 1 - mask
        name = name.split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'label': mask,
            'p_label':point_labels,
            'pt':pt,
            'image_meta_dict':image_meta_dict,
        }