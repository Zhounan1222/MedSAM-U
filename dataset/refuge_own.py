import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from utils import random_box, random_click,random_click_3point
import cv2
from dataset.isic import prompt_aug,generate_random_box_with_iou

def read_mask_Refuge(mask_path):
    # gt = np.load(mask_path).astype("uint8")
    # #gt = cv2.resize(gt, dsize=(256, 256), fx=1, fy=1, interpolation=cv2.INTER_NEAREST)
    # return gt
    # gt = plt.imread(mask_path).astype('uint8')
    # 使用cv2.imread读取图像
    gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # gt[gt != 0] = 1
    # # set the data type
    # # mask = mask.astype(np.int32)
    # # expand one dim, from (h, w) to (1, h, w)
    # gt = np.asarray(gt,dtype="uint8")
    # # assert to test mask
    # assert gt.min() >= 0 and gt.max() <= 1, "mask error !"
   

    # 确保gt是uint8类型
    # gt = gt.astype('uint8')
    # refuge_dataset
    gt[gt <= 245] = 1
    gt[gt > 245] = 0

    gt = np.asarray(gt, dtype="uint8")
    return gt

class REFUGE_point(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Train',prompt = 'point&box', plane = False):

        df = pd.read_csv(os.path.join(data_path, 'REFUGE_' + mode + '_GroundTruth.csv'), encoding='gbk')
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


        mask_array = np.where((np.array(mask)) > 245, 0, 1)
        mask_array = mask_array.astype(np.uint8)
        mask = Image.fromarray(mask_array * 255)

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
            # point_label, pt_disc = random_click(np.array(np.mean(np.stack(multi_rater_disc_np), axis=0)) / 255, point_label)
        else:
            # you may want to get rid of click prompts
            pt = np.array([0, 0], dtype=np.int32)
            
        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            mask = torch.as_tensor((self.transform_msk(mask) >0.5).float(), dtype=torch.float32)
            # transform to mask size (out_size) for mask define
            # mask = F.interpolate(mask, size=(self.mask_size, self.mask_size), mode='bilinear', align_corners=False).mean(dim=0)

            
            torch.set_rng_state(state)
            

        if self.prompt == 'box':
            x_min_cup, x_max_cup, y_min_cup, y_max_cup = random_box(multi_rater_cup)
            box_cup = [x_min_cup, x_max_cup, y_min_cup, y_max_cup]
            x_min_disc, x_max_disc, y_min_disc, y_max_disc = random_box(multi_rater_disc)
            box_disc = [x_min_disc, x_max_disc, y_min_disc, y_max_disc]
        else:
            # you may want to get rid of box prompts
            # box_cup = [0, 0, 0, 0]
            # box_disc = [0, 0, 0, 0]
            box_cup = None
            # box_disc = [0, 0, 0, 0]

        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'label': mask,
            'p_label':point_label,
            'pt':pt,
            'image_meta_dict':image_meta_dict,
        }
    

class REFUGE_box(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Train',prompt = 'box', plane = False):

        df = pd.read_csv(os.path.join(data_path, 'REFUGE_' + mode + '_GroundTruth.csv'), encoding='gbk')
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
        # point_label = 1

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

        # resize raters images for generating initial point click
        newsize = (self.img_size, self.img_size)



        # mask_tensor =  Image.fromarray(mask.astype(np.uint8))  # 转换为uint8类型并生成PIL图像
        # resized_image = mask_tensor.resize((1024, 1024), resample=Image.Resampling.LANCZOS)
        # mask = np.array(resized_image)
        # mask_size = mask.resize(newsize)
        mask_numpy = np.array(mask.resize(newsize))
        box_list = []

        # first click is the target agreement among most raters
        if self.prompt == 'box':
            col, row = np.nonzero(mask)
            prompt = np.array([row.min(), col.min(), row.max(), col.max()])

            for i in range(1):

                only_box = generate_random_box_with_iou(prompt,newsize)
                box_list.append(only_box)
            

            
        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            mask = torch.as_tensor((self.transform_msk(mask) >0.5).float(), dtype=torch.float32)
            # transform to mask size (out_size) for mask define
            # mask = F.interpolate(mask, size=(self.mask_size, self.mask_size), mode='bilinear', align_corners=False).mean(dim=0)

            
            torch.set_rng_state(state)
            

        if self.prompt == 'box':
            x_min_cup, x_max_cup, y_min_cup, y_max_cup = random_box(multi_rater_cup)
            box_cup = [x_min_cup, x_max_cup, y_min_cup, y_max_cup]
            x_min_disc, x_max_disc, y_min_disc, y_max_disc = random_box(multi_rater_disc)
            box_disc = [x_min_disc, x_max_disc, y_min_disc, y_max_disc]
        else:
            # you may want to get rid of box prompts
            # box_cup = [0, 0, 0, 0]
            # box_disc = [0, 0, 0, 0]
            box_cup = None
            # box_disc = [0, 0, 0, 0]

        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'label': mask,
            'box':box_list,
            'image_meta_dict':image_meta_dict
        }
    
class REFUGE_point_box(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Train',prompt = 'point&box', plane = False):

        df = pd.read_csv(os.path.join(data_path, 'REFUGE_' + mode + '_GroundTruth.csv'), encoding='gbk')
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
        box_list =[]

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
        sigma = 0.1*255  # 标准差（你可以调整这个值来控制噪声水平）
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

        # resize raters images for generating initial point click
        newsize = (self.img_size, self.img_size)

        mask = mask.resize(newsize)



        # mask_tensor =  Image.fromarray(mask.astype(np.uint8))  # 转换为uint8类型并生成PIL图像
        # resized_image = mask_tensor.resize((1024, 1024), resample=Image.Resampling.LANCZOS)
        # mask = np.array(resized_image)
        # mask_size = mask.resize(newsize)
        mask_numpy = np.array(mask.resize(newsize))

        # first click is the target agreement among most raters
        if self.prompt == 'point&box':
            point_label, pt = random_click_3point(mask_numpy / 255, point_label)
            col, row = np.nonzero(mask)
            prompt = np.array([row.min(), col.min(), row.max(), col.max()])

            for i in range(3):

                # only_box = generate_random_box_with_iou(prompt,newsize)
                only_box = prompt_aug(prompt,mask,True)
                box_list.append(only_box)
            
        elif self.prompt == 'box':
            # point_label, pt = random_click_3point(mask_numpy / 255, point_label)
            # col, row = np.nonzero(mask)
            # prompt = np.array([row.min(), col.min(), row.max(), col.max()])

            for i in range(3):

                only_box =prompt_aug (prompt=prompt,target=mask,aug=True)
                box_list.append(only_box)
            

            
        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(noisy_img)
            mask = torch.as_tensor((self.transform_msk(mask) >0.5).float(), dtype=torch.float32)
            # transform to mask size (out_size) for mask define
            # mask = F.interpolate(mask, size=(self.mask_size, self.mask_size), mode='bilinear', align_corners=False).mean(dim=0)

            
            torch.set_rng_state(state)
            

        if self.prompt == 'box':
            x_min_cup, x_max_cup, y_min_cup, y_max_cup = random_box(multi_rater_cup)
            box_cup = [x_min_cup, x_max_cup, y_min_cup, y_max_cup]
            x_min_disc, x_max_disc, y_min_disc, y_max_disc = random_box(multi_rater_disc)
            box_disc = [x_min_disc, x_max_disc, y_min_disc, y_max_disc]
        else:
            # you may want to get rid of box prompts
            # box_cup = [0, 0, 0, 0]
            # box_disc = [0, 0, 0, 0]
            box_cup = None
            # box_disc = [0, 0, 0, 0]

        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'label': mask,
            'pt':pt,
            'p_label':point_label,
            'box':box_list,
            'image_meta_dict':image_meta_dict
        }
    
   
class REFUGE_point_box(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Train',prompt = 'point&box', plane = False):

        df = pd.read_csv(os.path.join(data_path, 'REFUGE_' + mode + '_GroundTruth.csv'), encoding='gbk')
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
        box_list =[]

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
        sigma = 0.1*255  # 标准差（你可以调整这个值来控制噪声水平）
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

        # resize raters images for generating initial point click
        newsize = (self.img_size, self.img_size)

        mask = mask.resize(newsize)



        # mask_tensor =  Image.fromarray(mask.astype(np.uint8))  # 转换为uint8类型并生成PIL图像
        # resized_image = mask_tensor.resize((1024, 1024), resample=Image.Resampling.LANCZOS)
        # mask = np.array(resized_image)
        # mask_size = mask.resize(newsize)
        mask_numpy = np.array(mask.resize(newsize))

        # first click is the target agreement among most raters
        if self.prompt == 'point&box':
            point_label, pt = random_click_3point(mask_numpy / 255, point_label)
            col, row = np.nonzero(mask)
            prompt = np.array([row.min(), col.min(), row.max(), col.max()])

            for i in range(3):

                # only_box = generate_random_box_with_iou(prompt,newsize)
                only_box = prompt_aug(prompt,mask,True)
                box_list.append(only_box)
            
        elif self.prompt == 'box':
            # point_label, pt = random_click_3point(mask_numpy / 255, point_label)
            # col, row = np.nonzero(mask)
            # prompt = np.array([row.min(), col.min(), row.max(), col.max()])

            for i in range(3):

                only_box =prompt_aug (prompt=prompt,target=mask,aug=True)
                box_list.append(only_box)
            

            
        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(noisy_img)
            mask = torch.as_tensor((self.transform_msk(mask) >0.5).float(), dtype=torch.float32)
            # transform to mask size (out_size) for mask define
            # mask = F.interpolate(mask, size=(self.mask_size, self.mask_size), mode='bilinear', align_corners=False).mean(dim=0)

            
            torch.set_rng_state(state)
            

        if self.prompt == 'box':
            x_min_cup, x_max_cup, y_min_cup, y_max_cup = random_box(multi_rater_cup)
            box_cup = [x_min_cup, x_max_cup, y_min_cup, y_max_cup]
            x_min_disc, x_max_disc, y_min_disc, y_max_disc = random_box(multi_rater_disc)
            box_disc = [x_min_disc, x_max_disc, y_min_disc, y_max_disc]
        else:
            # you may want to get rid of box prompts
            # box_cup = [0, 0, 0, 0]
            # box_disc = [0, 0, 0, 0]
            box_cup = None
            # box_disc = [0, 0, 0, 0]

        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'label': mask,
            'pt':pt,
            'p_label':point_label,
            'box':box_list,
            'image_meta_dict':image_meta_dict
        }
        
    
class REFUGE_point_box_noise(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Train',prompt = 'point&box', plane = False):

        df = pd.read_csv(os.path.join(data_path, 'REFUGE_' + mode + '_GroundTruth.csv'), encoding='gbk')
        self.name_list = df.iloc[:,0].tolist()
        self.label_list = df.iloc[:,1].tolist()
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.device = args.gpu_device
        self.de_path = args.de_data_path

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
        box_list =[]

        """Get the images"""
        name = self.name_list[index]
        de_name = name.split('/')[-1].replace('img.jpg','img_111.jpg')
        img_path = os.path.join(self.data_path, name)
        
        de_path = os.path.join(self.de_path,de_name)
        mask_name = self.label_list[index]
        msk_path = os.path.join(self.data_path, mask_name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')

        img_lq = Image.open(de_path).convert('RGB')

        # 将图像转换为NumPy数组
        img_array = np.array(img, dtype=np.float32)
        
        # 生成高斯噪声
        mean = 0.5
        sigma = 0.1*255  # 标准差（你可以调整这个值来控制噪声水平）
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

        # resize raters images for generating initial point click
        newsize = (self.img_size, self.img_size)

        mask = mask.resize(newsize)



        # mask_tensor =  Image.fromarray(mask.astype(np.uint8))  # 转换为uint8类型并生成PIL图像
        # resized_image = mask_tensor.resize((1024, 1024), resample=Image.Resampling.LANCZOS)
        # mask = np.array(resized_image)
        # mask_size = mask.resize(newsize)
        mask_numpy = np.array(mask.resize(newsize))

        # first click is the target agreement among most raters
        if self.prompt == 'point&box':
            point_label, pt = random_click_3point(mask_numpy / 255, point_label)
            col, row = np.nonzero(mask)
            prompt = np.array([row.min(), col.min(), row.max(), col.max()])

            for i in range(3):

                only_box = generate_random_box_with_iou(prompt,newsize)
                # only_box = prompt_aug(prompt,mask,True)
                box_list.append(only_box)
            
        elif self.prompt == 'box':
            # point_label, pt = random_click_3point(mask_numpy / 255, point_label)
            # col, row = np.nonzero(mask)
            # prompt = np.array([row.min(), col.min(), row.max(), col.max()])

            for i in range(3):

                only_box =prompt_aug (prompt=prompt,target=mask,aug=True)
                box_list.append(only_box)
            

            
        if self.transform:
            state = torch.get_rng_state()
            img_noise = self.transform(noisy_img)
            img_raw = self.transform(img)
            img_lq = self.transform(img_lq)
            mask = torch.as_tensor((self.transform_msk(mask) >0.5).float(), dtype=torch.float32)
            # transform to mask size (out_size) for mask define
            # mask = F.interpolate(mask, size=(self.mask_size, self.mask_size), mode='bilinear', align_corners=False).mean(dim=0)

            
            torch.set_rng_state(state)
            

        if self.prompt == 'box':
            x_min_cup, x_max_cup, y_min_cup, y_max_cup = random_box(multi_rater_cup)
            box_cup = [x_min_cup, x_max_cup, y_min_cup, y_max_cup]
            x_min_disc, x_max_disc, y_min_disc, y_max_disc = random_box(multi_rater_disc)
            box_disc = [x_min_disc, x_max_disc, y_min_disc, y_max_disc]
        else:
            # you may want to get rid of box prompts
            # box_cup = [0, 0, 0, 0]
            # box_disc = [0, 0, 0, 0]
            box_cup = None
            # box_disc = [0, 0, 0, 0]

        image_meta_dict = {'filename_or_obj':name}
        return {
            'image_noise':img_noise,
            'image_raw':img_raw,
            'image_lq':img_lq,
            'label': mask,
            'pt':pt,
            'p_label':point_label,
            'box':box_list,
            'image_meta_dict':image_meta_dict
        }
class REFUGE_point_box_uncertainty(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Train',prompt = 'point&box', plane = False):

        df = pd.read_csv(os.path.join(data_path, 'REFUGE_' + mode + '_GroundTruth.csv'), encoding='gbk')
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
        box_list =[]

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

        # resize raters images for generating initial point click
        newsize = (self.img_size, self.img_size)

        mask = mask.resize(newsize)



        # mask_tensor =  Image.fromarray(mask.astype(np.uint8))  # 转换为uint8类型并生成PIL图像
        # resized_image = mask_tensor.resize((1024, 1024), resample=Image.Resampling.LANCZOS)
        # mask = np.array(resized_image)
        # mask_size = mask.resize(newsize)
        mask_numpy = np.array(mask.resize(newsize))

        # first click is the target agreement among most raters
        if self.prompt == 'point&box':
            # point_label, pt = random_click_3point(mask_numpy / 255, point_label)
            col, row = np.nonzero(mask)
            prompt = np.array([row.min(), col.min(), row.max(), col.max()])

            for i in range(3):

                # only_box = generate_random_box_with_iou(prompt,newsize)
                only_box = prompt_aug(prompt,mask,True)
                box_list.append(only_box)
            
        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            mask = torch.as_tensor((self.transform_msk(mask) >0.5).float(), dtype=torch.float32)
            # transform to mask size (out_size) for mask define
            # mask = F.interpolate(mask, size=(self.mask_size, self.mask_size), mode='bilinear', align_corners=False).mean(dim=0)

            
            torch.set_rng_state(state)
            

        if self.prompt == 'box':
            x_min_cup, x_max_cup, y_min_cup, y_max_cup = random_box(multi_rater_cup)
            box_cup = [x_min_cup, x_max_cup, y_min_cup, y_max_cup]
            x_min_disc, x_max_disc, y_min_disc, y_max_disc = random_box(multi_rater_disc)
            box_disc = [x_min_disc, x_max_disc, y_min_disc, y_max_disc]
        else:
            # you may want to get rid of box prompts
            # box_cup = [0, 0, 0, 0]
            # box_disc = [0, 0, 0, 0]
            box_cup = None
            # box_disc = [0, 0, 0, 0]

        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'label': mask,
            # 'pt':pt,
            # 'p_label':point_label,
            'box':box_list,
            'image_meta_dict':image_meta_dict
        }