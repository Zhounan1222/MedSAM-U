
import argparse
import os
import shutil
import sys
import tempfile
import time
from collections import OrderedDict
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from einops import rearrange
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.transforms import AsDiscrete
from PIL import Image
from skimage import io
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from tensorboardX import SummaryWriter
#from dataset import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

# import cfg_sam_point_box_train
from cfg import cfg_sam_point_box_train
# import models.sam.utils.transforms as samtrans
import pytorch_ssim
#from models.discriminatorlayer import discriminator
from conf import settings
from utils import *
import pandas as pd
# from lucent.modelzoo.util import get_model_layers
# from lucent.optvis import render, param, transform, objectives
# from lucent.modelzoo import inceptionv1


args = cfg_sam_point_box_train.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)
pos_weight =torch.ones(([1])*2,device=GPUdevice)
criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
seed = torch.randint(1,11,(args.b,7))

torch.backends.cudnn.benchmark = True
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
scaler = torch.cuda.amp.GradScaler()
max_iterations = settings.EPOCH
post_label = AsDiscrete(to_onehot=14)
post_pred = AsDiscrete(argmax=True, to_onehot=14)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []




def train_sam(args, net: nn.Module, optimizer, train_loader,
          epoch, writer, schedulers=None, vis = 50):
    hard = 0
    epoch_loss = 0
    ind = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)

    # train mode
    net.train()
    optimizer.zero_grad()

  

    epoch_loss = 0
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))

    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        lossfunc = criterion_G

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for pack in train_loader:
            # torch.cuda.empty_cache()
            imgs = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            masks = pack['label'].to(dtype = torch.float32, device = GPUdevice)
            # for k,v in pack['image_meta_dict'].items():
            #     print(k)
            if 'pt' not in pack:
                imgs, pt, masks = generate_click_prompt(imgs, masks)
            elif 'box' in pack:
                pt = pack['pt']
                point_labels = pack['p_label']
                box = pack['box'].to(dtype = torch.int64, device = GPUdevice)
            else:
                pt = pack['pt']
                point_labels = pack['p_label']
            name = pack['image_meta_dict']['filename_or_obj']

            if args.thd:
                imgs, pt, masks = generate_click_prompt(imgs, masks)

                pt = rearrange(pt, 'b n d -> (b d) n')
                imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                masks = rearrange(masks, 'b c h w d -> (b d) c h w ')

                imgs = imgs.repeat(1,3,1,1)
                point_labels = torch.ones(imgs.size(0))

                imgs = torchvision.transforms.Resize((args.image_size,args.image_size))(imgs)
                masks = torchvision.transforms.Resize((args.out_size,args.out_size))(masks)
            showp = pt

            mask_type = torch.float32
            ind += 1
            b_size,c,w,h = imgs.size()
            longsize = w if w >=h else h

            if point_labels.clone().flatten()[0] != -1:
                    # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                point_coords = pt
                coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                if(len(point_labels.shape)==1): # only one point prompt
                    coords_torch, labels_torch, showp = coords_torch[None, :, :], labels_torch[None, :], showp[None, :, :]
                pt = (coords_torch, labels_torch)

            '''init'''
            if hard:
                true_mask_ave = (true_mask_ave > 0.5).float()
                #true_mask_ave = cons_tensor(true_mask_ave)
            # imgs = imgs.to(dtype = mask_type,device = GPUdevice)

            '''Train'''
            if args.mod == 'sam_adpt':
                for n, value in net.image_encoder.named_parameters(): 
                    if "Adapter" not in n:
                        value.requires_grad = False
                    else:
                        value.requires_grad = True
            elif args.mod == 'sam_lora' or args.mod == 'sam_adalora':
                from models.common import loralib as lora
                lora.mark_only_lora_as_trainable(net.image_encoder)
                if args.mod == 'sam_adalora':
                    # Initialize the RankAllocator 
                    rankallocator = lora.RankAllocator(
                        net.image_encoder, lora_r=4, target_rank=8,
                        init_warmup=500, final_warmup=1500, mask_interval=10, 
                        total_step=3000, beta1=0.85, beta2=0.85, 
                    )
            else:
                for n, value in net.image_encoder.named_parameters(): 
                    value.requires_grad = True
                    
            imge= net.image_encoder(imgs)
            with torch.no_grad():
                if args.net == 'sam' or args.net == 'mobile_sam' or args.net == 'medsam' and 'box' not in pack:
                    se, de = net.prompt_encoder(
                        points=pt,
                        boxes=None,
                        masks=None,
                    )
                elif args.net == 'sam' or args.net == 'mobile_sam' or args.net == 'medsam' and 'box' in pack:
                    se, de = net.prompt_encoder(
                        points=pt,
                        boxes=box,
                        masks=None,
                    )
                elif args.net == "efficient_sam":
                    coords_torch,labels_torch = transform_prompt(coords_torch,labels_torch,h,w)
                    se = net.prompt_encoder(
                        coords=coords_torch,
                        labels=labels_torch,
                    )
                    
            if args.net == 'sam' or  args.net == 'medsam':
                pred, _ = net.mask_decoder(
                    image_embeddings=imge,
                    image_pe=net.prompt_encoder.get_dense_pe(), 
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de, 
                    multimask_output=(args.multimask_output > 1),
                )
            elif args.net == 'mobile_sam':
                pred, _ = net.mask_decoder(
                    image_embeddings=imge,
                    image_pe=net.prompt_encoder.get_dense_pe(), 
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de, 
                    multimask_output=False,
                )
            elif args.net == "efficient_sam":
                se = se.view(
                    se.shape[0],
                    1,
                    se.shape[1],
                    se.shape[2],
                )
                pred, _ = net.mask_decoder(
                    image_embeddings=imge,
                    image_pe=net.prompt_encoder.get_dense_pe(), 
                    sparse_prompt_embeddings=se,
                    multimask_output=False,
                )
                
            # Resize to the ordered output size
            pred_final = F.interpolate(pred,size=(args.out_size,args.out_size))
            # pred_final_numpy = pred_final.cpu().numpy()



            # numpy_pred_bool = np.where(pred_final_numpy >= 0, 1, 0)

            loss = lossfunc(pred_final, masks)

            

            iou,dice = eval_seg(pred_final,masks,threshold)



            pbar.set_postfix(**{'loss (batch)': loss.item()})
            epoch_loss += loss.item()

            # nn.utils.clip_grad_value_(net.parameters(), 0.1)
            if args.mod == 'sam_adalora':
                (loss+lora.compute_orth_regu(net, regu_weight=0.1)).backward()
                optimizer.step()
                rankallocator.update_and_mask(net, ind)
            else:
                loss.backward()
                optimizer.step()
            
            optimizer.zero_grad()

            '''vis images'''
            if vis:
                if 'box' not in pack:
                    if ind % vis == 0:
                        namecat = 'Train'
                        for na in name[:2]:
                            namecat = namecat + na.split('/')[-1].split('.')[0] + '+'
                        vis_image(imgs,pred,masks, os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch)+'+dice_'+str(dice)+'+iou_'+str(iou) + '.jpg'), reverse=False, points=showp)
                elif 'box' in pack:
                    if ind % vis == 0:
                        namecat = 'Train'
                        for na in name[:2]:
                            namecat = namecat + na.split('/')[-1].split('.')[0] 
                        # epoch_path = os.path.join(args.path_helper['sample_path'],'train_epoch_'+str(epoch))
                        epoch_path = make_sub_folder(os.path.join(args.path_helper['sample_path'],'Train'),namecat)
                        vis_image_isic_png(imgs,pred_final,masks, epoch_path,dice,iou, reverse=False, points=showp,box=box)

            pbar.update()
            break

    return loss



def train_medsam_point_box(args, net: nn.Module, optimizer, train_loader,
          epoch, writer, schedulers=None, vis = 50):
    hard = 0
    epoch_loss = 0
    ind = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)

    # train mode
    net.train()
    optimizer.zero_grad()

  

    epoch_loss = 0
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))

    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        lossfunc = criterion_G

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for pack in train_loader:
            # torch.cuda.empty_cache()
            imgs = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            masks = pack['label'].to(dtype = torch.float32, device = GPUdevice)
            # for k,v in pack['image_meta_dict'].items():
            #     print(k)
            if 'pt' not in pack:
                imgs, pt, masks = generate_click_prompt(imgs, masks)
            elif 'box' in pack:
                pt = pack['pt']
                point_labels = pack['p_label']
                box = pack['box']
                pt = pack['pt']
                point_labels = pack['p_label']
            name = pack['image_meta_dict']['filename_or_obj']


            if len(pack["box"][0]) != 4 or len(pack['box'][1])!=4:
                continue

            if args.thd:
                imgs, pt, masks = generate_click_prompt(imgs, masks)

                pt = rearrange(pt, 'b n d -> (b d) n')
                imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                masks = rearrange(masks, 'b c h w d -> (b d) c h w ')

                imgs = imgs.repeat(1,3,1,1)
                point_labels = torch.ones(imgs.size(0))

                imgs = torchvision.transforms.Resize((args.image_size,args.image_size))(imgs)
                masks = torchvision.transforms.Resize((args.out_size,args.out_size))(masks)
            showp = pt

            mask_type = torch.float32
            ind += 1
            b_size,c,w,h = imgs.size()
            longsize = w if w >=h else h

            if point_labels.clone().flatten()[0] != -1:
                    # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                point_coords = pt
                coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                if(len(point_labels)==1): # only one point prompt
                    coords_torch, labels_torch, showp = coords_torch[None, :, :], labels_torch[None, :], showp[None, :, :]
                    pt = (coords_torch, labels_torch)
                else:
                    pt = (coords_torch.unsqueeze(1), labels_torch.unsqueeze(1))

            '''init'''
            if hard:
                true_mask_ave = (true_mask_ave > 0.5).float()
                #true_mask_ave = cons_tensor(true_mask_ave)
            imgs = imgs.to(dtype = mask_type,device = GPUdevice)

            '''Train'''
            if args.mod == 'sam_adpt':
                for n, value in net.image_encoder.named_parameters(): 
                    if "Adapter" not in n:
                        value.requires_grad = False
                    else:
                        value.requires_grad = True
            elif args.mod == 'sam_lora' or args.mod == 'sam_adalora':
                from models.common import loralib as lora
                lora.mark_only_lora_as_trainable(net.image_encoder)
                if args.mod == 'sam_adalora':
                    # Initialize the RankAllocator 
                    rankallocator = lora.RankAllocator(
                        net.image_encoder, lora_r=4, target_rank=8,
                        init_warmup=500, final_warmup=1500, mask_interval=10, 
                        total_step=3000, beta1=0.85, beta2=0.85, 
                    )
            else:
                for n, value in net.image_encoder.named_parameters(): 
                    value.requires_grad = True
                    
            imge= net.image_encoder(imgs)
            with torch.no_grad():
                # if args.net == 'sam' or args.net == 'mobile_sam' or args.net == 'medsam' and 'box' not in pack:
                #     se, de = net.prompt_encoder(
                #         points=pt,
                #         boxes=None,
                #         masks=None,
                #     )
                if args.net == 'sam' or args.net == 'mobile_sam' or args.net == 'medsam' and 'box' in pack:
                    se, de = net.prompt_encoder(
                        points=pt,
                        boxes=box.to(dtype = torch.int64, device = GPUdevice),
                        masks=None,
                    )
                elif args.net == "efficient_sam":
                    coords_torch,labels_torch = transform_prompt(coords_torch,labels_torch,h,w)
                    se = net.prompt_encoder(
                        coords=coords_torch,
                        labels=labels_torch,
                    )
                    
            if args.net == 'sam' or  args.net == 'medsam':
                pred, _ = net.mask_decoder(
                    image_embeddings=imge,
                    image_pe=net.prompt_encoder.get_dense_pe(), 
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de, 
                    multimask_output=(args.multimask_output > 1),
                )
            elif args.net == 'mobile_sam':
                pred, _ = net.mask_decoder(
                    image_embeddings=imge,
                    image_pe=net.prompt_encoder.get_dense_pe(), 
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de, 
                    multimask_output=False,
                )
            elif args.net == "efficient_sam":
                se = se.view(
                    se.shape[0],
                    1,
                    se.shape[1],
                    se.shape[2],
                )
                pred, _ = net.mask_decoder(
                    image_embeddings=imge,
                    image_pe=net.prompt_encoder.get_dense_pe(), 
                    sparse_prompt_embeddings=se,
                    multimask_output=False,
                )
                
            # Resize to the ordered output size
            pred_final = F.interpolate(pred,size=(args.out_size,args.out_size))
            # pred_final_numpy = pred_final.cpu().numpy()



            # numpy_pred_bool = np.where(pred_final_numpy >= 0, 1, 0)

            loss = lossfunc(pred_final, masks)

            

            temp = eval_seg(pred_final,masks,threshold)



            pbar.set_postfix(**{'loss (batch)': loss.item()})
            epoch_loss += loss.item()

            # nn.utils.clip_grad_value_(net.parameters(), 0.1)
            if args.mod == 'sam_adalora':
                (loss+lora.compute_orth_regu(net, regu_weight=0.1)).backward()
                optimizer.step()
                rankallocator.update_and_mask(net, ind)
            else:
                loss.backward()
                optimizer.step()
            
            optimizer.zero_grad()

            '''vis images'''
            # if vis:
            #     if 'box' not in pack:
            #         if ind % vis == 0:
            #             namecat = 'Train'
            #             for na in name[:2]:
            #                 namecat = namecat + na.split('/')[-1].split('.')[0] + '+'
            #             vis_image(imgs,pred,masks, os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch)+'+dice_'+str(dice)+'+iou_'+str(iou) + '.jpg'), reverse=False, points=showp)
            #     elif 'box' in pack:
            #         if ind % vis == 0:
            #             namecat = 'Train'
            #             for na in name[:2]:
            #                 namecat = namecat + na.split('/')[-1].split('.')[0] 
            #             # epoch_path = os.path.join(args.path_helper['sample_path'],'train_epoch_'+str(epoch))
            #             epoch_path = make_sub_folder(os.path.join(args.path_helper['sample_path'],'Train'),namecat)
            #             vis_image_isic_png(imgs,pred_final,masks, epoch_path,dice,iou, reverse=False, points=showp,box=box)

            pbar.update()
            # break

    return loss

def make_sub_folder(output_dir,item_path,sub_list=None):
    subfolder_path_list = []
     # 在样本文件夹下创建子文件夹
    sample_folder = os.path.join(output_dir, item_path)
    os.makedirs(sample_folder, exist_ok=True)
        
        # 在样本文件夹下创建子文件夹
    if sub_list is not None:
        for subfolder_name in sub_list:
            subfolder_path = os.path.join(sample_folder, subfolder_name)
            os.makedirs(subfolder_path, exist_ok=True)
            subfolder_path_list.append(subfolder_path)
        return subfolder_path_list
    else:
        return sample_folder


def validation_medsam_1box(args, val_loader, epoch, net: nn.Module, clean_dir=True):
     # eval mode
    net.eval()

    mask_type = torch.float32
    n_val = len(val_loader)  # the number of batch
    ave_res, mix_res = (0,0,0,0), (0,)*args.multimask_output*2
    rater_res = [(0,0,0,0) for _ in range(6)]
    tot = 0
    hard = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice

    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        lossfunc = criterion_G
    df = pd.DataFrame(columns=['File Name'])

    dice_ave =[]
    iou_ave = []

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
           

            imgsw = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            masksw = pack['label'].to(dtype = torch.float32, device = GPUdevice)
            # for k,v in pack['image_meta_dict'].items():
            #     print(k)
            
            box = pack['box'].to(dtype = torch.int64, device = GPUdevice)
            name = pack['image_meta_dict']['filename_or_obj']
            
            buoy = 0

            namecat = 'Test'
            for na in name[:2
            
            ]:
                img_name = na.split('/')[-1].split('.')[0]
                namecat = namecat + img_name
            epoch_path = make_sub_folder(os.path.join(args.path_helper['sample_path']),namecat)
            sub_path_list = make_sub_folder_2(epoch_path,['prompt','gt','mask'])
            
            if args.evl_chunk:
                evl_ch = int(args.evl_chunk)
            else:
                evl_ch = int(imgsw.size(-1))

            while (buoy + evl_ch) <= imgsw.size(-1):
                
                imgs = imgsw[...,buoy:buoy + evl_ch]
                masks = masksw[...,buoy:buoy + evl_ch]
                buoy += evl_ch

                if args.thd:
                    pt = rearrange(pt, 'b n d -> (b d) n')
                    imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                    masks = rearrange(masks, 'b c h w d -> (b d) c h w ')
                    imgs = imgs.repeat(1,3,1,1)
                    point_labels = torch.ones(imgs.size(0))

                    imgs = torchvision.transforms.Resize((args.image_size,args.image_size))(imgs)
                    masks = torchvision.transforms.Resize((args.out_size,args.out_size))(masks)
                
                

                mask_type = torch.float32
                ind += 1
                b_size,c,w,h = imgs.size()
                longsize = w if w >=h else h

               

                '''init'''
                if hard:
                    true_mask_ave = (true_mask_ave > 0.5).float()
                    #true_mask_ave = cons_tensor(true_mask_ave)
                imgs = imgs.to(dtype = mask_type,device = GPUdevice)
                imgs = imgs.to(dtype = mask_type,device = GPUdevice)
                imgs_1 = imgs.cpu().numpy().squeeze(0).transpose(1,2,0)
                imgs_1 = (imgs_1*255).astype(np.uint8) 
                imgs_2 = imgs_1.copy()
                imgs_2 = cv2.cvtColor(imgs_2, cv2.COLOR_RGB2BGR)

                gt = masks.cpu().numpy().squeeze(0).transpose(1,2,0)
                gt = (gt* 255).astype(np.uint8) 
                cv2.imwrite(os.path.join(sub_path_list[1], 'gt.png'), gt)
                
                '''test'''
                with torch.no_grad():
                    imge= net.image_encoder(imgs)
                    

                    if args.net == 'sam' or args.net == 'mobile_sam' or args.net =='medsam' and 'box' in pack:
                        # for i in range(len(box)):
                        x1,y1,x2,y2 = box[0]
                        cv2.rectangle(imgs_2, (x1.cpu().item(), y1.cpu().item()), (x2.cpu().item(), y2.cpu().item()), (0,0,255), 1)  # 2 表示线的粗细
                            # cv2.circle(imgs_2,(points[0][0][0].item(),points[0][0][1].item()),3,(0,255,0),-1)
                        cv2.imwrite(os.path.join(sub_path_list[0], 'prompt.png'), imgs_2)

                        se, de = net.prompt_encoder(
                            points=None,
                            boxes=box,
                            masks=None,
                        )
                

                    if args.net == 'sam' or args.net =='medsam':
                        pred, _ = net.mask_decoder(
                            image_embeddings=imge,
                            image_pe=net.prompt_encoder.get_dense_pe(), 
                            sparse_prompt_embeddings=se,
                            dense_prompt_embeddings=de, 
                            multimask_output=(args.multimask_output > 1),
                        )
                
                        # Resize to the ordered output size
                        pred = F.interpolate(pred,size=(args.out_size,args.out_size))
                        temp = eval_seg(pred, masks, threshold)

                        pred_img = np.where((pred.squeeze(0).cpu().numpy())>=0.5,True,False)
                        pred_img =(pred_img*255).astype(np.uint8).transpose(1,2,0)

                            
                        cv2.imwrite(os.path.join(sub_path_list[2], 'mask_{}.png'.format(temp[1])), pred_img)

                        tot += lossfunc(pred, masks)

                        # print(f'{name} : before : dice1 = {temp[1]},iou={temp[0]}')
                        df = df._append({'File Name': name,'DICE Score ': temp[1],'IOU Score': temp[0]}, ignore_index=True)
                        df.to_csv(os.path.join(args.path_helper['prefix'],'output_values.csv'), index=False)


                        
                        mix_res = tuple([sum(a) for a in zip(mix_res, temp)])

                        dice_ave.append(temp[1])
                        iou_ave.append(temp[0])
                
                        print(f'{name} : before : dice1 = {temp[1]},iou={temp[0]}')



                    
            pbar.update()
            # break
    dice_score = np.array(dice_ave)
    iou_score = np.array(iou_ave)

    print('----------Have finished testing --------------------')
    print('The Ave_Dice:{}'.format(dice_score.mean(axis=0)))
    print('The Ave_Iou:{}'.format(iou_score.mean(axis=0)))
    print('---------------------------------------------------')

    df = df._append({'File Name': 'avg','Dice_ave:': dice_score.mean(axis=0),'IOU_ave': iou_score.mean(axis=0)}, ignore_index=True)
    df.to_csv(os.path.join(args.path_helper['prefix'],'output_values.csv'), index=False)
    if args.evl_chunk:
        n_val = n_val * (imgsw.size(-1) // evl_ch)

    return tot/ n_val , tuple([a/n_val for a in mix_res])



def validation_medsam_3box(args, val_loader, epoch, net: nn.Module, clean_dir=True):
     # eval mode
    net.eval()

    mask_type = torch.float32
    n_val = len(val_loader)  # the number of batch
    ave_res, mix_res = (0,0,0,0), (0,)*args.multimask_output*2
    rater_res = [(0,0,0,0) for _ in range(6)]
    tot = 0
    hard = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice

    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        lossfunc = criterion_G

    df = pd.DataFrame(columns=['File Name'])

    dice_ave =[]
    iou_ave = []

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            pred_list = []
            uncertainty_list = []
    
            imgsw = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            masksw = pack['label'].to(dtype = torch.float32, device = GPUdevice)
            # for k,v in pack['image_meta_dict'].items():
            #     print(k)
           
            
            box = pack['box']
            name = pack['image_meta_dict']['filename_or_obj']

            namecat = 'Test'
            for na in name[:2
            
            ]:
                img_name = na.split('/')[-1].split('.')[0]
                namecat = namecat + img_name
            epoch_path = make_sub_folder(os.path.join(args.path_helper['sample_path']),namecat)
            sub_path_list = make_sub_folder_2(epoch_path,['prompt','gt','mask','uncertainty'])
            
            buoy = 0
            if args.evl_chunk:
                evl_ch = int(args.evl_chunk)
            else:
                evl_ch = int(imgsw.size(-1))

            while (buoy + evl_ch) <= imgsw.size(-1):
    

                imgs = imgsw[...,buoy:buoy + evl_ch]
                masks = masksw[...,buoy:buoy + evl_ch]
                buoy += evl_ch

                if args.thd:
                    pt = rearrange(pt, 'b n d -> (b d) n')
                    imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                    masks = rearrange(masks, 'b c h w d -> (b d) c h w ')
                    imgs = imgs.repeat(1,3,1,1)
                    point_labels = torch.ones(imgs.size(0))

                    imgs = torchvision.transforms.Resize((args.image_size,args.image_size))(imgs)
                    masks = torchvision.transforms.Resize((args.out_size,args.out_size))(masks)
                

                mask_type = torch.float32
                ind += 1
                b_size,c,w,h = imgs.size()
                longsize = w if w >=h else h

               
                '''init'''
                if hard:
                    true_mask_ave = (true_mask_ave > 0.5).float()
                    #true_mask_ave = cons_tensor(true_mask_ave)
                imgs = imgs.to(dtype = mask_type,device = GPUdevice)
                imgs_1 = imgs.cpu().numpy().squeeze(0).transpose(1,2,0)
                imgs_1 = (imgs_1*255).astype(np.uint8) 
                imgs_2 = imgs_1.copy()
                imgs_2 = cv2.cvtColor(imgs_2, cv2.COLOR_RGB2BGR)

                gt = masks.cpu().numpy().squeeze(0).transpose(1,2,0)
                gt = (gt* 255).astype(np.uint8) 
                cv2.imwrite(os.path.join(sub_path_list[1], 'gt.png'), gt)

                '''test'''
                with torch.no_grad():
                    imge= net.image_encoder(imgs)
                    

                    if args.net == 'sam' or args.net == 'mobile_sam' or args.net =='medsam' and 'box' in pack:
                        for i in range(len(box)):
                            x1,y1,x2,y2 = box[i][0]
                            cv2.rectangle(imgs_2, (x1.cpu().item(), y1.cpu().item()), (x2.cpu().item(), y2.cpu().item()), (0,0,255), 1)  # 2 表示线的粗细
                            # cv2.circle(imgs_2,(points[0][0][0].item(),points[0][0][1].item()),3,(0,255,0),-1)
                            cv2.imwrite(os.path.join(sub_path_list[0], 'prompt.png'), imgs_2)




                            se, de = net.prompt_encoder(
                                points=None,
                                boxes=box[i].to(dtype = torch.int64, device = GPUdevice),
                                masks=None,
                            )
                

                   
                            pred, _ = net.mask_decoder(
                                image_embeddings=imge,
                                image_pe=net.prompt_encoder.get_dense_pe(), 
                                sparse_prompt_embeddings=se,
                                dense_prompt_embeddings=de, 
                                multimask_output=(args.multimask_output > 1),
                            )
                    
                            # Resize to the ordered output size
                            pred = F.interpolate(pred,size=(args.out_size,args.out_size))
                            temp = eval_seg(pred, masks, threshold)

                            pred_img = np.where((pred.squeeze(0).cpu().numpy())>=0.5,True,False)
                            pred_img =(pred_img*255).astype(np.uint8).transpose(1,2,0)

                            
                            cv2.imwrite(os.path.join(sub_path_list[2], 'iter0_mask_{}_{}.png'.format(i+1,temp[1])), pred_img)

                            pred_list.append(pred)
                            uncertainty_list.append(pred)
                        
                        ave_pred = torch.mean(torch.cat(pred_list,dim=1),dim=1,keepdim=True)

                        ave_temp = eval_seg(ave_pred, masks, threshold)

                        dice_ave.append(ave_temp[1])
                        iou_ave.append(ave_temp[0])

                        ave_pred_img = np.where((ave_pred.squeeze(0).cpu().numpy())>=0.5,True,False)
                        ave_pred_img =(ave_pred_img*255).astype(np.uint8).transpose(1,2,0)
                        cv2.imwrite(os.path.join(sub_path_list[2], 'iter0_mask_ave_{}.png'.format(ave_temp[1])), ave_pred_img)
                        


                            
                        tot += lossfunc(ave_pred, masks)



                            
                        mix_res = tuple([sum(a) for a in zip(mix_res, ave_temp)])
                        y1 = torch.cat(uncertainty_list, dim=1)
                        y1 = torch.mean(torch.sigmoid(y1), dim=1)
                        uncertainty_1 = entropy(y1)  # 4numof tta
                        uncertainty_1 = uncertainty_1.squeeze(0)

                        # 绘制热力图
                        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
                        plt.imshow(uncertainty_1, cmap=plt.cm.jet,norm=norm)  # 使用热色图作为颜色映射
                        plt.axis('off')  # 关闭坐标轴
                        plt.savefig(os.path.join(sub_path_list[3], 'iter0_hot_uncertainty.png'))

                        uncertainty1_png = (uncertainty_1 * 255).astype(np.uint8) 
                        
                        cv2.imwrite(os.path.join(sub_path_list[3], 'iter0_uncertainty_{}_{}.png'.format(uncertainty_1.sum(),uncertainty_1.sum()/(ave_pred.sum()))), uncertainty1_png)
                        

                        print(f'{name} : before : dice1 = {ave_temp[1]},iou={ave_temp[0]}')
                        df = df._append({'File Name': name,'DICE Score ': ave_temp[1],'IOU Score': ave_temp[0]}, ignore_index=True)
                        df.to_csv(os.path.join(args.path_helper['prefix'],'output_values.csv'), index=False)


               
                    
            pbar.update()
            # break
    
    dice_score = np.array(dice_ave)
    iou_score = np.array(iou_ave)

    print('----------Have finished testing --------------------')
    print('The Ave_Dice:{}'.format(dice_score.mean(axis=0)))
    print('The Ave_Iou:{}'.format(iou_score.mean(axis=0)))
    print('---------------------------------------------------')

    df = df._append({'File Name': 'avg','Dice_ave:': dice_score.mean(axis=0),'IOU_ave': iou_score.mean(axis=0)}, ignore_index=True)
    df.to_csv(os.path.join(args.path_helper['prefix'],'output_values.csv'), index=False)

    if args.evl_chunk:
        n_val = n_val * (imgsw.size(-1) // evl_ch)

    return tot/ n_val , tuple([a/n_val for a in mix_res])




def validation_medsam_1point_3box(args, val_loader, epoch, net: nn.Module, clean_dir=True):
     # eval mode
    net.eval()

    mask_type = torch.float32
    n_val = len(val_loader)  # the number of batch
    ave_res, mix_res = (0,0,0,0), (0,)*args.multimask_output*2
    rater_res = [(0,0,0,0) for _ in range(6)]
    tot = 0
    hard = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice

    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        lossfunc = criterion_G

    df = pd.DataFrame(columns=['File Name','DICE_0 Score ','IOU_0 Score','uncertainty_0 Score','ratio1','DICE_1 Score ','IOU_1 Score','uncertainty_2 Score','ratio2'])

    dice0_ave =[]
    iou0_ave = []
    dice1_ave =[]
    iou1_ave = []

    uncertainty1_ave = []
    uncertainty2_ave = []
    count = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            pred_list = []
            uncertainty_list = []

            pred_list1 = []
            uncertainty_list1 = []
    
            imgsw = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            masksw = pack['label'].to(dtype = torch.float32, device = GPUdevice)
            # for k,v in pack['image_meta_dict'].items():
            #     print(k)
            
            box = pack['box']
            name = pack['image_meta_dict']['filename_or_obj']

            namecat = 'Test'
            for na in name[:2
            
            ]:
                img_name = na.split('/')[-1].split('.')[0]
                namecat = namecat + img_name
            epoch_path = make_sub_folder(os.path.join(args.path_helper['sample_path']),namecat)
            sub_path_list = make_sub_folder_2(epoch_path,['prompt','gt','mask','uncertainty','FN_mask','FP_mask'])
            
            buoy = 0
            if args.evl_chunk:
                evl_ch = int(args.evl_chunk)
            else:
                evl_ch = int(imgsw.size(-1))

            while (buoy + evl_ch) <= imgsw.size(-1):
               
                imgs = imgsw[...,buoy:buoy + evl_ch]
                masks = masksw[...,buoy:buoy + evl_ch]
                buoy += evl_ch

                if args.thd:
                    pt = rearrange(pt, 'b n d -> (b d) n')
                    imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                    masks = rearrange(masks, 'b c h w d -> (b d) c h w ')
                    imgs = imgs.repeat(1,3,1,1)
                    point_labels = torch.ones(imgs.size(0))

                    imgs = torchvision.transforms.Resize((args.image_size,args.image_size))(imgs)
                    masks = torchvision.transforms.Resize((args.out_size,args.out_size))(masks)
                
               

                mask_type = torch.float32
                ind += 1
                b_size,c,w,h = imgs.size()
                longsize = w if w >=h else h

                
                '''init'''
                if hard:
                    true_mask_ave = (true_mask_ave > 0.5).float()
                    #true_mask_ave = cons_tensor(true_mask_ave)
                imgs = imgs.to(dtype = mask_type,device = GPUdevice)
                imgs_256 = F.interpolate(imgs, size=(256, 256), mode='nearest')
                img_256_np = imgs_256.cpu().numpy().squeeze(0).transpose(1,2,0)
                imgs_np = imgs.cpu().numpy().squeeze(0).transpose(1,2,0)
                imgs_1 = (imgs_np*255).astype(np.uint8) 
                imgs_2 = imgs_1.copy()
                imgs_2 = cv2.cvtColor(imgs_2, cv2.COLOR_RGB2BGR)

                gt_np = masks.cpu().numpy().squeeze(0)
                gt = (gt_np* 255).astype(np.uint8).transpose(1,2,0)
                cv2.imwrite(os.path.join(sub_path_list[1], 'gt.png'), gt)

                '''test'''
                with torch.no_grad():
                    imge= net.image_encoder(imgs)
                    

                    if args.net == 'sam' or args.net == 'mobile_sam' or args.net =='medsam' and 'box' in pack:
                        for i in range(len(box)):
                            x1,y1,x2,y2 = box[i][0]
                            # y1_1,x1_1 =pt[0][0][0]
                            # cv2.circle(imgs_2,(int(x1_1.item()),int(y1_1.item())),5,(255,0,0),-1)
                            cv2.rectangle(imgs_2, (x1.cpu().item(), y1.cpu().item()), (x2.cpu().item(), y2.cpu().item()), (0,0,255), 1)  # 2 表示线的粗细
                            # cv2.circle(imgs_2,(points[0][0][0].item(),points[0][0][1].item()),3,(0,255,0),-1)
                            cv2.imwrite(os.path.join(sub_path_list[0], 'iter0_box.png'), imgs_2)




                            se, de = net.prompt_encoder(
                                points=None,
                                boxes=box[i].to(dtype = torch.int64, device = GPUdevice),
                                masks=None,
                            )
                

                   
                            pred, _ = net.mask_decoder(
                                image_embeddings=imge,
                                image_pe=net.prompt_encoder.get_dense_pe(), 
                                sparse_prompt_embeddings=se,
                                dense_prompt_embeddings=de, 
                                multimask_output=(args.multimask_output > 1),
                            )
                    
                            # Resize to the ordered output size
                            pred = F.interpolate(pred,size=(args.out_size,args.out_size))
                            temp = eval_seg(pred, masks, threshold)

                            pred_img = np.where((pred.squeeze(0).cpu().numpy())>=0.5,True,False)
                            pred_img =(pred_img*255).astype(np.uint8).transpose(1,2,0)

                            
                            cv2.imwrite(os.path.join(sub_path_list[2], 'iter0_mask_{}_{}.png'.format(i+1,temp[1])), pred_img)

                            pred_list.append(pred)
                            uncertainty_list.append(pred)
                        
                        ave_pred = torch.mean(torch.cat(pred_list,dim=1),dim=1,keepdim=True)

                        ave_temp = eval_seg(ave_pred, masks, threshold)

                        dice0_ave.append(ave_temp[1])
                        iou0_ave.append(ave_temp[0])

                        ave_pred_binary = np.where((ave_pred.squeeze(0).cpu().numpy())>=0.5,True,False)
                        ave_pred_img =(ave_pred_binary*255).astype(np.uint8).transpose(1,2,0)
                        cv2.imwrite(os.path.join(sub_path_list[2], 'iter0_mask_ave_{}.png'.format(ave_temp[1])), ave_pred_img)
                        


                            
                        tot += lossfunc(ave_pred, masks)



                            
                        mix_res = tuple([sum(a) for a in zip(mix_res, ave_temp)])
                        y1 = torch.cat(uncertainty_list, dim=1)
                        y1 = torch.mean(torch.sigmoid(y1), dim=1)
                        uncertainty_0 = entropy(y1)  # 4numof tta
                        uncertainty_0 = uncertainty_0.squeeze(0)

                        # 绘制热力图
                        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
                        plt.imshow(uncertainty_0, cmap=plt.cm.jet,norm=norm)  # 使用热色图作为颜色映射
                        plt.axis('off')  # 关闭坐标轴
                        plt.savefig(os.path.join(sub_path_list[3], 'iter0_hot_uncertainty.png'))

                        uncertainty0_png = (uncertainty_0 * 255).astype(np.uint8) 
                        
                        cv2.imwrite(os.path.join(sub_path_list[3], 'iter0_uncertainty_{}_{}.png'.format(uncertainty_0.sum(),uncertainty_0.sum()/(ave_pred.sum()))), uncertainty0_png)
                        
                        uncertainty1_ave.append(uncertainty_0.sum())
                        ratio1 = uncertainty_0.sum()/ave_pred_binary.sum()

                        print(f'{name} : before : dice1 = {ave_temp[1]},iou={ave_temp[0]}')
                        # df = df._append({'File Name': name,'DICE_0 Score ': ave_temp[1],'IOU_0 Score': ave_temp[0],'uncertainty_0 Score':uncertainty_0.sum(),'ratio1':ratio1}, ignore_index=True)
                        # df.to_csv(os.path.join(args.path_helper['prefix'],'output_values.csv'), index=False)
                    
    


            masked_uncertainty_map,FN_output_mask, FN_UH, FN_UH_png,FN_xUH,FN_xUH_png,FN_output_mask, FP_UH,FP_xUH_png,FN_condition_mask,FP_condition_mask  = uc_refine_correct_FN_add_FP_gt(ave_pred_binary.copy(),
                                                                       gt_np.copy(),uncertainty_0.copy(),
                                                                       img_256_np.copy(),sub_path_list)
            
            
            # 产生points和points_labels
            indices_FN = np.argwhere(masked_uncertainty_map.squeeze(0) > 0)
            indices_FP = np.argwhere(FP_xUH_png>0)
            random_index_FN = np.random.choice(len(indices_FN), int(len(indices_FN)),replace=False)

            selected_points_FN = [indices_FN[i] for i in random_index_FN] #以行，列的形式存储

            random_index_FP = np.random.choice(len(indices_FP), int(len(indices_FP)),replace=False)

            selected_points_FP = [indices_FP[i] for i in random_index_FP] #以行，列的形式存储


            uncertainty_values_FN = []  # 初始化最大不确定性值为负无穷
            uncertainty_values_FP = []
            # max_uncertainty_point = None  # 初始化最大不.确定性值对应的点为None
            points_list = []
            image2 = imgs_np.copy()


            for i in range(len(selected_points_FN)):
                uncertainty_value_FN = get_uncertainty_value(uncertainty_0.copy(),imgs_np.copy(), selected_points_FN[i],sub_path_list,i+1)
                uncertainty_values_FN.append((selected_points_FN[i], uncertainty_value_FN,'FN'))
            # 找到具有最大不确定性值的三个点
            # max_uncertainty_points_FN = sorted(uncertainty_values_FN, key=lambda x: x[1], reverse=True)[:1]

            # for i in range(len(selected_points_FP)):
            #     uncertainty_value_FP = get_uncertainty_value(uncertainty_1.copy(),image.copy(), selected_points_FP[i],subfolder_path_list,i+1)
            #     uncertainty_values_FP.append((selected_points_FP[i], uncertainty_value_FP,'FP'))
            # # # 找到具有最大不确定性值的三个点
            # # 合并两个列表
            # uncertainty_values_FN.extend(uncertainty_values_FP)

            # max_uncertainty_points_FN = sorted(uncertainty_values_FN, key=lambda x: x[1], reverse=True)[:1]


            # max_uncertainty_points_FP = sorted(uncertainty_values_FP, key=lambda x: x[1], reverse=True)[:1]
            scale_x = 1024//256
            for i in range(len(uncertainty_values_FN)):


                max_uncertainty_points_FN = sorted(uncertainty_values_FN, key=lambda x: x[1], reverse=True)[0]

                y1,x1 = list(max_uncertainty_points_FN)[0]
                break
            # x1 = x1.item() if isinstance(x1, np.generic) else int(x1)
            # y1 = y1.item() if isinstance(y1, np.generic) else int(y1)

                # if x_1largest <=  x1 <= x_2largest and y1 <= y1_largest <= y_2largest:
            points_list.append([y1*scale_x,x1*scale_x])  #点为行，列
            # cv2.circle(image2,(int(x1*scale_x),int(y1*scale_x)),3,(255,0,0),-1) #opencv中点为列，行
                #     # cv2.circle(image1,(x1,y1),3,(255,0,0),-1)
                #     break
                # else :
                #     continue
            # if points_list == []:
            #     y1,x1 = list(max_uncertainty_points_FN)[0]
            #     points_list.append([y1,x1])  #点为行，列
            #     cv2.circle(imgs_2,(x1,y1),3,(255,0,0),-1) #opencv中点为列，行
            
            input_point = points_list
            # cv2.circle(image2,(x2,y2),3,(0,0,255),-1) #opencv中点为列，行
            # cv2.circle(image1,(x2,y2),3,(0,0,255),-1)        

            cv2.imwrite(os.path.join(sub_path_list[2], 'max_points_uncertainty_{}.png'.format(max_uncertainty_points_FN[0][1])), image2)
            cv2.imwrite(os.path.join(sub_path_list[0], 'iter1_point.png'), imgs_2)

            # if label_0_1 == 'FN':
            input_label = torch.tensor([1])
            if input_label.clone().flatten()[0] != -1:
                # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                point_coords = input_point
                coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                labels_torch = torch.as_tensor(input_label, dtype=torch.int, device=GPUdevice)
                if(len(input_label.shape)==1): # only one point prompt
                    coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                pt = (coords_torch, labels_torch)          
        # else:
                # input_label = np.array([0])  
            pred_uncertainty_list2 = []

            for t in range(len(box)):    
                # aug_prompt1 = prompt_aug1(prompt=box1, target=gt, aug=True,scale_factor=256)
                with torch.no_grad():
                    imge= net.image_encoder(imgs)
                    x1,y1,x2,y2 = box[t][0]
                    y1_1,x1_1 = points_list[0]
                    cv2.circle(imgs_2,(x1_1,y1_1),5,(255,0,0),-1)
                    cv2.rectangle(imgs_2, (x1.cpu().item(), y1.cpu().item()), (x2.cpu().item(), y2.cpu().item()), (0,0,255), 1)  # 2 表示线的粗细
                    # cv2.circle(imgs_2,(points[0][0][0].item(),points[0][0][1].item()),3,(0,255,0),-1)
                    cv2.imwrite(os.path.join(sub_path_list[0], 'prompt.png'), imgs_2)




                    se, de = net.prompt_encoder(
                        points=pt,
                        boxes=box[t].to(dtype = torch.int64, device = GPUdevice),
                        masks=None,
                    )
        

            
                    pred, _ = net.mask_decoder(
                        image_embeddings=imge,
                        image_pe=net.prompt_encoder.get_dense_pe(), 
                        sparse_prompt_embeddings=se,
                        dense_prompt_embeddings=de, 
                        multimask_output=(args.multimask_output > 1),
                    )
            
                    # Resize to the ordered output size
                    pred = F.interpolate(pred,size=(args.out_size,args.out_size))
                    temp = eval_seg(pred, masks, threshold)

                    pred_img = np.where((pred.squeeze(0).cpu().numpy())>=0.5,True,False)
                    pred_img =(pred_img*255).astype(np.uint8).transpose(1,2,0)

                    
                    cv2.imwrite(os.path.join(sub_path_list[2], 'iter1_mask_{}_{}.png'.format(t+1,temp[1])), pred_img)

                    pred_list1.append(pred)
                    uncertainty_list1.append(pred)
                
            ave_pred1 = torch.mean(torch.cat(pred_list1,dim=1),dim=1,keepdim=True)

            ave_temp1 = eval_seg(ave_pred1, masks, threshold)

            dice1_ave.append(ave_temp1[1])
            iou1_ave.append(ave_temp1[0])

            ave_pred_binary1 = np.where((ave_pred1.squeeze(0).cpu().numpy())>=0.5,True,False)
            ave_pred_img1=(ave_pred_binary1*255).astype(np.uint8).transpose(1,2,0)
            cv2.imwrite(os.path.join(sub_path_list[2], 'iter1_mask_ave_{}.png'.format(ave_temp1[1])), ave_pred_img1)
            


                    
            tot += lossfunc(ave_pred1, masks)



                
            # mix_res1 = tuple([sum(a) for a in zip(mix_res, ave_temp1)])
            y2 = torch.cat(uncertainty_list1, dim=1)
            y2 = torch.mean(torch.sigmoid(y2), dim=1)
            uncertainty_2 = entropy(y2)  # 4numof tta
            uncertainty_2 = uncertainty_2.squeeze(0)

            # 绘制热力图
            norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
            plt.imshow(uncertainty_2, cmap=plt.cm.jet,norm=norm)  # 使用热色图作为颜色映射
            plt.axis('off')  # 关闭坐标轴
            plt.savefig(os.path.join(sub_path_list[3], 'iter1_hot_uncertainty.png'))

            uncertainty2_png = (uncertainty_2 * 255).astype(np.uint8) 

            uncertainty2_ave.append(uncertainty_2.sum())
            ratio2 = uncertainty_2.sum()/ave_pred_binary1.sum()
            
            cv2.imwrite(os.path.join(sub_path_list[3], 'iter1_uncertainty_{}_{}.png'.format(uncertainty_2.sum(),ratio2)), uncertainty2_png)
            

            print(f'{name} : after : dice1 = {ave_temp1[1]},iou1={ave_temp1[0]}')
            df = df._append({'File Name':name,'DICE_0 Score ': ave_temp[1],'IOU_0 Score': ave_temp[0],'uncertainty_0 Score':uncertainty_0.sum(),'ratio1':ratio1,'DICE_1 Score ': ave_temp1[1],'IOU_1 Score': ave_temp1[0],'uncertainty_2 Score':uncertainty_2.sum(),'ratio2':ratio2}, ignore_index=True)
            df.to_csv(os.path.join(args.path_helper['prefix'],'output_values.csv'), index=False)

            if ratio1 > ratio2:
                count +=1
    

        




               
                    
            pbar.update()
            # break
    
    dice0_score = np.array(dice0_ave)
    iou0_score = np.array(iou0_ave)

    uncertainty1_score = np.array(uncertainty1_ave)

    print('----------Have finished testing --------------------')
    print('The Ave_Dice:{}'.format(dice0_score.mean(axis=0)))
    print('The Ave_Iou:{}'.format(iou0_score.mean(axis=0)))
    print('The Ave_uncertainty:{}'.format(uncertainty1_score.mean(axis=0)))
    print('---------------------------------------------------')

    df = df._append({'File Name': name,'Dice0_ave': dice0_score.mean(axis=0),'IOU0_ave': iou0_score.mean(axis=0)}, ignore_index=True)
    df.to_csv(os.path.join(args.path_helper['prefix'],'output_values.csv'), index=False)

    dice1_score = np.array(dice1_ave)
    iou1_score = np.array(iou1_ave)
    uncertainty2_score = np.array(uncertainty2_ave)

    print('----------Have finished testing --------------------')
    print('The Ave_Dice:{}'.format(dice1_score.mean(axis=0)))
    print('The Ave_Iou:{}'.format(iou1_score.mean(axis=0)))
    print('The Ave_uncertainty:{}'.format(uncertainty2_score.mean(axis=0)))
    print('---------------------------------------------------')

    print('count: {}'.format(count))

    df = df._append({'File Name':name,'Dice1_ave': dice1_score.mean(axis=0),'IOU1_ave': iou1_score.mean(axis=0)}, ignore_index=True)
    df.to_csv(os.path.join(args.path_helper['prefix'],'output_values.csv'), index=False)

    if args.evl_chunk:
        n_val = n_val * (imgsw.size(-1) // evl_ch)

    return tot/ n_val , tuple([a/n_val for a in mix_res])


def test_medsamu_v2(args, val_loader, epoch, net: nn.Module, clean_dir=True):
     # eval mode
    net.eval()

    mask_type = torch.float32
    n_val = len(val_loader)  # the number of batch
    ave_res, mix_res = (0,0,0,0), (0,)*args.multimask_output*2
    rater_res = [(0,0,0,0) for _ in range(6)]
    tot = 0
    hard = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice

    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        lossfunc = criterion_G

    df = pd.DataFrame(columns=['File Name','DICE_0 Score ','IOU_0 Score','uncertainty_0 Score','ratio1','DICE_1 Score ','IOU_1 Score','uncertainty_2 Score','ratio2'])

    dice0_ave =[]
    iou0_ave = []
    dice1_ave =[]
    iou1_ave = []

    after_dice_ave =[]
    after_iou_ave = []

    uncertainty1_ave = []
    uncertainty2_ave = []
    after_uncertainty_ave = []
    uncertainty_down_count  = 0
    uncertainty_up_count = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            pred_list = []
            uncertainty_list = []

            pred_list1 = []
            uncertainty_list1 = []
    
            imgsw = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            masksw = pack['label'].to(dtype = torch.float32, device = GPUdevice)
            # for k,v in pack['image_meta_dict'].items():
            #     print(k)
            
            box = pack['box']
            name = pack['image_meta_dict']['filename_or_obj']

            namecat = 'Test'
            for na in name[:2
            
            ]:
                img_name = na.split('/')[-1].split('.')[0]
                namecat = namecat + img_name
            epoch_path = make_sub_folder(os.path.join(args.path_helper['sample_path']),namecat)
            sub_path_list = make_sub_folder_2(epoch_path,['prompt','gt','mask','uncertainty','FN_mask','FP_mask'])
            
            buoy = 0
            if args.evl_chunk:
                evl_ch = int(args.evl_chunk)
            else:
                evl_ch = int(imgsw.size(-1))

            while (buoy + evl_ch) <= imgsw.size(-1):
               
                imgs = imgsw[...,buoy:buoy + evl_ch]
                masks = masksw[...,buoy:buoy + evl_ch]
                buoy += evl_ch

                if args.thd:
                    pt = rearrange(pt, 'b n d -> (b d) n')
                    imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                    masks = rearrange(masks, 'b c h w d -> (b d) c h w ')
                    imgs = imgs.repeat(1,3,1,1)
                    point_labels = torch.ones(imgs.size(0))

                    imgs = torchvision.transforms.Resize((args.image_size,args.image_size))(imgs)
                    masks = torchvision.transforms.Resize((args.out_size,args.out_size))(masks)
                
               

                mask_type = torch.float32
                ind += 1
                b_size,c,w,h = imgs.size()
                longsize = w if w >=h else h

                
                '''init'''
                if hard:
                    true_mask_ave = (true_mask_ave > 0.5).float()
                    #true_mask_ave = cons_tensor(true_mask_ave)
                imgs = imgs.to(dtype = mask_type,device = GPUdevice)
                imgs_256 = F.interpolate(imgs, size=(256, 256), mode='nearest')
                img_256_np = imgs_256.cpu().numpy().squeeze(0).transpose(1,2,0)
                imgs_np = imgs.cpu().numpy().squeeze(0).transpose(1,2,0)
                imgs_1 = (imgs_np*255).astype(np.uint8) 
                imgs_2 = imgs_1.copy()
                imgs_2 = cv2.cvtColor(imgs_2, cv2.COLOR_RGB2BGR)

                gt_np = masks.cpu().numpy().squeeze(0)
                gt = (gt_np* 255).astype(np.uint8).transpose(1,2,0)
                cv2.imwrite(os.path.join(sub_path_list[1], 'gt.png'), gt)

                '''test'''
                with torch.no_grad():
                    imge= net.image_encoder(imgs)
                    

                    if args.net == 'sam' or args.net == 'mobile_sam' or args.net =='medsam' and 'box' in pack:
                        for i in range(len(box)):
                            x1,y1,x2,y2 = box[i][0]
                            # y1_1,x1_1 =pt[0][0][0]
                            # cv2.circle(imgs_2,(int(x1_1.item()),int(y1_1.item())),5,(255,0,0),-1)
                            cv2.rectangle(imgs_2, (x1.cpu().item(), y1.cpu().item()), (x2.cpu().item(), y2.cpu().item()), (0,0,255), 1)  # 2 表示线的粗细
                            # cv2.circle(imgs_2,(points[0][0][0].item(),points[0][0][1].item()),3,(0,255,0),-1)
                            cv2.imwrite(os.path.join(sub_path_list[0], 'iter0_box.png'), imgs_2)




                            se, de = net.prompt_encoder(
                                points=None,
                                boxes=box[i].to(dtype = torch.int64, device = GPUdevice),
                                masks=None,
                            )
                

                   
                            pred, _ = net.mask_decoder(
                                image_embeddings=imge,
                                image_pe=net.prompt_encoder.get_dense_pe(), 
                                sparse_prompt_embeddings=se,
                                dense_prompt_embeddings=de, 
                                multimask_output=(args.multimask_output > 1),
                            )
                    
                            # Resize to the ordered output size
                            pred = F.interpolate(pred,size=(args.out_size,args.out_size))
                            temp = eval_seg(pred, masks, threshold)

                            pred_img = np.where((pred.squeeze(0).cpu().numpy())>=0.5,True,False)
                            pred_img =(pred_img*255).astype(np.uint8).transpose(1,2,0)

                            
                            cv2.imwrite(os.path.join(sub_path_list[2], 'iter0_mask_{}_{}.png'.format(i+1,temp[1])), pred_img)

                            pred_list.append(pred)
                            uncertainty_list.append(pred)
                        
                        ave_pred = torch.mean(torch.cat(pred_list,dim=1),dim=1,keepdim=True)

                        ave_temp = eval_seg(ave_pred, masks, threshold)

                        dice0_ave.append(ave_temp[1])
                        iou0_ave.append(ave_temp[0])

                        ave_pred_binary = np.where((ave_pred.squeeze(0).cpu().numpy())>=0.5,True,False)
                        ave_pred_img =(ave_pred_binary*255).astype(np.uint8).transpose(1,2,0)
                        cv2.imwrite(os.path.join(sub_path_list[2], 'iter0_mask_ave_{}.png'.format(ave_temp[1])), ave_pred_img)
                        


                            
                        tot += lossfunc(ave_pred, masks)



                            
                        mix_res = tuple([sum(a) for a in zip(mix_res, ave_temp)])
                        y1 = torch.cat(uncertainty_list, dim=1)
                        y1 = torch.mean(torch.sigmoid(y1), dim=1)
                        uncertainty_0 = entropy(y1)  # 4numof tta
                        uncertainty_0 = uncertainty_0.squeeze(0)

                        # 绘制热力图
                        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
                        plt.imshow(uncertainty_0, cmap=plt.cm.jet,norm=norm)  # 使用热色图作为颜色映射
                        plt.axis('off')  # 关闭坐标轴
                        plt.savefig(os.path.join(sub_path_list[3], 'iter0_hot_uncertainty.png'))

                        uncertainty0_png = (uncertainty_0 * 255).astype(np.uint8) 
                        
                        cv2.imwrite(os.path.join(sub_path_list[3], 'iter0_uncertainty_{}_{}.png'.format(uncertainty_0.sum(),uncertainty_0.sum()/(ave_pred.sum()))), uncertainty0_png)
                        
                        uncertainty1_ave.append(uncertainty_0.sum())
                        ratio1 = uncertainty_0.sum()/ave_pred_binary.sum()

                        print(f'{name} : before : dice1 = {ave_temp[1]},iou={ave_temp[0]}')
                        # df = df._append({'File Name': name,'DICE_0 Score ': ave_temp[1],'IOU_0 Score': ave_temp[0],'uncertainty_0 Score':uncertainty_0.sum(),'ratio1':ratio1}, ignore_index=True)
                        # df.to_csv(os.path.join(args.path_helper['prefix'],'output_values.csv'), index=False)
                    
    


            masked_uncertainty_map,FN_output_mask, FN_UH, FN_UH_png,FN_xUH,FN_xUH_png,FN_output_mask, FP_UH,FP_xUH_png,FN_condition_mask,FP_condition_mask  = uc_refine_correct_FN_add_FP(ave_pred_binary.copy(),
                                                                     uncertainty_0.copy(),
                                                                       img_256_np.copy(),sub_path_list)
            
            
            # 产生points和points_labels
            indices_FN = np.argwhere(masked_uncertainty_map.squeeze(0) > 0)
            indices_FP = np.argwhere(FP_xUH_png>0)
            random_index_FN = np.random.choice(len(indices_FN), int(len(indices_FN)),replace=False)

            selected_points_FN = [indices_FN[i] for i in random_index_FN] #以行，列的形式存储

            random_index_FP = np.random.choice(len(indices_FP), int(len(indices_FP)),replace=False)

            selected_points_FP = [indices_FP[i] for i in random_index_FP] #以行，列的形式存储


            uncertainty_values_FN = []  # 初始化最大不确定性值为负无穷
            uncertainty_values_FP = []
            # max_uncertainty_point = None  # 初始化最大不.确定性值对应的点为None
            points_list = []
            image2 = imgs_np.copy()


            for i in range(len(selected_points_FN)):
                uncertainty_value_FN = get_uncertainty_value(uncertainty_0.copy(),imgs_np.copy(), selected_points_FN[i],sub_path_list,i+1)
                uncertainty_values_FN.append((selected_points_FN[i], uncertainty_value_FN,'FN'))
   
            scale_x = 1024//256
            for i in range(len(uncertainty_values_FN)):


                max_uncertainty_points_FN = sorted(uncertainty_values_FN, key=lambda x: x[1], reverse=True)[i]

                y1,x1 = list(max_uncertainty_points_FN)[0]
                x1 = x1*scale_x
                y1 = y1*scale_x
                

                # x1 = x1*scale_x
                # y1 = y1*scale_x

                break

                # if x_1largest <=  x1 <= x_2largest and y1 <= y1_largest <= y_2largest:
                # 检查并转换 x1 和 y1 为整数标量
            
            # 确保它们是整数
            x1 = int(x1)
            y1 = int(y1)

            points_list.append([y1,x1])  #点为行，列
            # cv2.circle(image2,(x1,y1),3,(255,0,0),-1) #opencv中点为列，行
          
            
            input_point = points_list
          

            cv2.imwrite(os.path.join(sub_path_list[2], 'max_points_uncertainty_{}.png'.format(max_uncertainty_points_FN[0][1])), image2)
            # cv2.imwrite(os.path.join(sub_path_list[0], 'iter1_point.png'), imgs_2)

            # if label_0_1 == 'FN':
            input_label = torch.tensor([1])
            if input_label.clone().flatten()[0] != -1:
                # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                point_coords = input_point
                coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                labels_torch = torch.as_tensor(input_label, dtype=torch.int, device=GPUdevice)
                if(len(input_label.shape)==1): # only one point prompt
                    coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                pt = (coords_torch, labels_torch)          
        # else:
                # input_label = np.array([0])  
            pred_uncertainty_list2 = []

            for t in range(len(box)):    
                # aug_prompt1 = prompt_aug1(prompt=box1, target=gt, aug=True,scale_factor=256)
                with torch.no_grad():
                    imge= net.image_encoder(imgs)
                    x1,y1,x2,y2 = box[t][0]
                    y1_1,x1_1 = points_list[0]
                    cv2.circle(imgs_2,(x1_1,y1_1),5,(255,0,0),-1)
                    cv2.rectangle(imgs_2, (x1.cpu().item(), y1.cpu().item()), (x2.cpu().item(), y2.cpu().item()), (0,0,255), 1)  # 2 表示线的粗细
                    # cv2.circle(imgs_2,(points[0][0][0].item(),points[0][0][1].item()),3,(0,255,0),-1)
                    cv2.imwrite(os.path.join(sub_path_list[0], 'prompt.png'), imgs_2)




                    se, de = net.prompt_encoder(
                        points=pt,
                        boxes=box[t].to(dtype = torch.int64, device = GPUdevice),
                        masks=None,
                    )
        

            
                    pred, _ = net.mask_decoder(
                        image_embeddings=imge,
                        image_pe=net.prompt_encoder.get_dense_pe(), 
                        sparse_prompt_embeddings=se,
                        dense_prompt_embeddings=de, 
                        multimask_output=(args.multimask_output > 1),
                    )
            
                    # Resize to the ordered output size
                    pred = F.interpolate(pred,size=(args.out_size,args.out_size))
                    temp = eval_seg(pred, masks, threshold)

                    pred_img = np.where((pred.squeeze(0).cpu().numpy())>=0.5,True,False)
                    pred_img =(pred_img*255).astype(np.uint8).transpose(1,2,0)

                    
                    cv2.imwrite(os.path.join(sub_path_list[2], 'iter1_mask_{}_{}.png'.format(t+1,temp[1])), pred_img)

                    pred_list1.append(pred)
                    uncertainty_list1.append(pred)
                
            ave_pred1 = torch.mean(torch.cat(pred_list1,dim=1),dim=1,keepdim=True)

            ave_temp1 = eval_seg(ave_pred1, masks, threshold)

            dice1_ave.append(ave_temp1[1])
            iou1_ave.append(ave_temp1[0])

            ave_pred_binary1 = np.where((ave_pred1.squeeze(0).cpu().numpy())>=0.5,True,False)
            ave_pred_img1=(ave_pred_binary1*255).astype(np.uint8).transpose(1,2,0)
            cv2.imwrite(os.path.join(sub_path_list[2], 'iter1_mask_ave_{}.png'.format(ave_temp1[1])), ave_pred_img1)
            


                    
            tot += lossfunc(ave_pred1, masks)



                
            # mix_res1 = tuple([sum(a) for a in zip(mix_res, ave_temp1)])
            y2 = torch.cat(uncertainty_list1, dim=1)
            y2 = torch.mean(torch.sigmoid(y2), dim=1)
            uncertainty_2 = entropy(y2)  # 4numof tta
            uncertainty_2 = uncertainty_2.squeeze(0)

            # 绘制热力图
            norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
            plt.imshow(uncertainty_2, cmap=plt.cm.jet,norm=norm)  # 使用热色图作为颜色映射
            plt.axis('off')  # 关闭坐标轴
            plt.savefig(os.path.join(sub_path_list[3], 'iter1_hot_uncertainty.png'))

            uncertainty2_png = (uncertainty_2 * 255).astype(np.uint8) 

            uncertainty2_ave.append(uncertainty_2.sum())
            ratio2 = uncertainty_2.sum()/ave_pred_binary1.sum()
            
            cv2.imwrite(os.path.join(sub_path_list[3], 'iter1_uncertainty_{}_{}.png'.format(uncertainty_2.sum(),ratio2)), uncertainty2_png)
            

            print(f'{name} : iter1 : dice1 = {ave_temp1[1]},iou1={ave_temp1[0]}')
            
            if ratio1 > ratio2:
                uncertainty_down_count +=1

                after_dice = ave_temp1[1]
                after_iou = ave_temp1[0]

                after_uncertainty = uncertainty_2.sum()
                ratio_finish = ratio2

                after_dice_ave.append(after_dice)
                after_iou_ave.append(after_iou)
                after_uncertainty_ave.append(after_uncertainty)
            else:
                after_dice = ave_temp[1]
                after_iou = ave_temp[0]
                after_uncertainty = uncertainty_0.sum()

                after_dice_ave.append(after_dice)
                after_iou_ave.append(after_iou)
                after_uncertainty_ave.append(after_uncertainty)

                uncertainty_up_count +=1

                ratio_finish = ratio1
                
            print(f'{name} : iter_finish : dice_finish = {after_dice},iou_finish={after_iou}')





            df = df._append({'File Name':name,'DICE_0 Score ': ave_temp[1],'IOU_0 Score': ave_temp[0],'uncertainty_0 Score':uncertainty_0.sum(),'ratio1':ratio1,'DICE_1 Score ': ave_temp1[1],'IOU_1 Score': ave_temp1[0],'uncertainty_2 Score':uncertainty_2.sum(),'ratio2':ratio2,'DICE_finish Score ': after_dice,'IOU_finish Score': after_iou,'uncertainty_finish Score':after_uncertainty.sum(),'ratio_finish':ratio_finish}, ignore_index=True)
            df.to_csv(os.path.join(args.path_helper['prefix'],'output_values.csv'), index=False)

    

        




               
                    
            pbar.update()
            # break
    
    dice0_score = np.array(dice0_ave)
    iou0_score = np.array(iou0_ave)

    uncertainty1_score = np.array(uncertainty1_ave)

    print('----------Have finished Iter0 testing --------------------')
    print('The Ave_Dice:{}'.format(dice0_score.mean(axis=0)))
    print('The Ave_Iou:{}'.format(iou0_score.mean(axis=0)))
    print('The Ave_uncertainty:{}'.format(uncertainty1_score.mean(axis=0)))
    print('---------------------------------------------------')

    df = df._append({'File Name': name,'Dice0_ave': dice0_score.mean(axis=0),'IOU0_ave': iou0_score.mean(axis=0)}, ignore_index=True)
    df.to_csv(os.path.join(args.path_helper['prefix'],'output_values.csv'), index=False)

    dice1_score = np.array(dice1_ave)
    iou1_score = np.array(iou1_ave)
    uncertainty2_score = np.array(uncertainty2_ave)

    print('----------Have finished Iter1 testing --------------------')
    print('The Ave_Dice:{}'.format(dice1_score.mean(axis=0)))
    print('The Ave_Iou:{}'.format(iou1_score.mean(axis=0)))
    print('The Ave_uncertainty:{}'.format(uncertainty2_score.mean(axis=0)))
    print('---------------------------------------------------')

    dice_finish_score = np.array(after_dice_ave)
    iou_finish_score = np.array(after_iou_ave)
    uncertainty_finish_score = np.array(after_uncertainty_ave)

    print('----------Have finished all testing --------------------')
    print('The Ave_Dice:{}'.format(dice_finish_score.mean(axis=0)))
    print('The Ave_Iou:{}'.format(iou_finish_score.mean(axis=0)))
    print('The Ave_uncertainty:{}'.format(uncertainty_finish_score.mean(axis=0)))
    print('---------------------------------------------------')

    print('uncertainty down count: {}'.format(uncertainty_down_count))
    print('uncertainty up count: {}'.format(uncertainty_up_count))

    df = df._append({'File Name':name,'Dice_finish_ave': dice_finish_score.mean(axis=0),'IOU_finish_ave': iou_finish_score.mean(axis=0)}, ignore_index=True)
    df.to_csv(os.path.join(args.path_helper['prefix'],'output_values.csv'), index=False)

    if args.evl_chunk:
        n_val = n_val * (imgsw.size(-1) // evl_ch)

    return tot/ n_val , tuple([a/n_val for a in mix_res])


def test_medsamu(args, val_loader, epoch, net: nn.Module, clean_dir=True):
     # eval mode
    net.eval()

    mask_type = torch.float32
    n_val = len(val_loader)  # the number of batch
    ave_res, mix_res = (0,0,0,0), (0,)*args.multimask_output*2
    rater_res = [(0,0,0,0) for _ in range(6)]
    tot = 0
    hard = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice

    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        lossfunc = criterion_G

    df = pd.DataFrame(columns=['File Name','DICE_0 Score ','IOU_0 Score','uncertainty_0 Score','ratio1','DICE_1 Score ','IOU_1 Score','uncertainty_2 Score','ratio2'])

    dice0_ave =[]
    iou0_ave = []
    dice1_ave =[]
    iou1_ave = []

    uncertainty1_ave = []
    uncertainty2_ave = []
    count = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            pred_list = []
            uncertainty_list = []

            pred_list1 = []
            uncertainty_list1 = []
    
            imgsw = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            masksw = pack['label'].to(dtype = torch.float32, device = GPUdevice)
            # for k,v in pack['image_meta_dict'].items():
            #     print(k)
            
            box = pack['box']
            name = pack['image_meta_dict']['filename_or_obj']

            namecat = 'Test'
            for na in name[:2
            
            ]:
                img_name = na.split('/')[-1].split('.')[0]
                namecat = namecat + img_name
            epoch_path = make_sub_folder(os.path.join(args.path_helper['sample_path']),namecat)
            sub_path_list = make_sub_folder_2(epoch_path,['prompt','gt','mask','uncertainty','FN_mask','FP_mask'])
            
            buoy = 0
            if args.evl_chunk:
                evl_ch = int(args.evl_chunk)
            else:
                evl_ch = int(imgsw.size(-1))

            while (buoy + evl_ch) <= imgsw.size(-1):
               
                imgs = imgsw[...,buoy:buoy + evl_ch]
                masks = masksw[...,buoy:buoy + evl_ch]
                buoy += evl_ch

                if args.thd:
                    pt = rearrange(pt, 'b n d -> (b d) n')
                    imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                    masks = rearrange(masks, 'b c h w d -> (b d) c h w ')
                    imgs = imgs.repeat(1,3,1,1)
                    point_labels = torch.ones(imgs.size(0))

                    imgs = torchvision.transforms.Resize((args.image_size,args.image_size))(imgs)
                    masks = torchvision.transforms.Resize((args.out_size,args.out_size))(masks)
                
               

                mask_type = torch.float32
                ind += 1
                b_size,c,w,h = imgs.size()
                longsize = w if w >=h else h

                
                '''init'''
                if hard:
                    true_mask_ave = (true_mask_ave > 0.5).float()
                    #true_mask_ave = cons_tensor(true_mask_ave)
                imgs = imgs.to(dtype = mask_type,device = GPUdevice)
                imgs_256 = F.interpolate(imgs, size=(256, 256), mode='nearest')
                img_256_np = imgs_256.cpu().numpy().squeeze(0).transpose(1,2,0)
                imgs_np = imgs.cpu().numpy().squeeze(0).transpose(1,2,0)
                imgs_1 = (imgs_np*255).astype(np.uint8) 
                imgs_2 = imgs_1.copy()
                imgs_2 = cv2.cvtColor(imgs_2, cv2.COLOR_RGB2BGR)

                gt_np = masks.cpu().numpy().squeeze(0)
                gt = (gt_np* 255).astype(np.uint8).transpose(1,2,0)
                cv2.imwrite(os.path.join(sub_path_list[1], 'gt.png'), gt)

                '''test'''
                with torch.no_grad():
                    imge= net.image_encoder(imgs)
                    

                    if args.net == 'sam' or args.net == 'mobile_sam' or args.net =='medsam' and 'box' in pack:
                        for i in range(len(box)):
                            x1,y1,x2,y2 = box[i][0]
                            # y1_1,x1_1 =pt[0][0][0]
                            # cv2.circle(imgs_2,(int(x1_1.item()),int(y1_1.item())),5,(255,0,0),-1)
                            cv2.rectangle(imgs_2, (x1.cpu().item(), y1.cpu().item()), (x2.cpu().item(), y2.cpu().item()), (0,0,255), 1)  # 2 表示线的粗细
                            # cv2.circle(imgs_2,(points[0][0][0].item(),points[0][0][1].item()),3,(0,255,0),-1)
                            cv2.imwrite(os.path.join(sub_path_list[0], 'iter0_box.png'), imgs_2)




                            se, de = net.prompt_encoder(
                                points=None,
                                boxes=box[i].to(dtype = torch.int64, device = GPUdevice),
                                masks=None,
                            )
                

                   
                            pred, _ = net.mask_decoder(
                                image_embeddings=imge,
                                image_pe=net.prompt_encoder.get_dense_pe(), 
                                sparse_prompt_embeddings=se,
                                dense_prompt_embeddings=de, 
                                multimask_output=(args.multimask_output > 1),
                            )
                    
                            # Resize to the ordered output size
                            pred = F.interpolate(pred,size=(args.out_size,args.out_size))
                            temp = eval_seg(pred, masks, threshold)

                            pred_img = np.where((pred.squeeze(0).cpu().numpy())>=0.5,True,False)
                            pred_img =(pred_img*255).astype(np.uint8).transpose(1,2,0)

                            
                            cv2.imwrite(os.path.join(sub_path_list[2], 'iter0_mask_{}_{}.png'.format(i+1,temp[1])), pred_img)

                            pred_list.append(pred)
                            uncertainty_list.append(pred)
                        
                        ave_pred = torch.mean(torch.cat(pred_list,dim=1),dim=1,keepdim=True)

                        ave_temp = eval_seg(ave_pred, masks, threshold)

                        dice0_ave.append(ave_temp[1])
                        iou0_ave.append(ave_temp[0])

                        ave_pred_binary = np.where((ave_pred.squeeze(0).cpu().numpy())>=0.5,True,False)
                        ave_pred_img =(ave_pred_binary*255).astype(np.uint8).transpose(1,2,0)
                        cv2.imwrite(os.path.join(sub_path_list[2], 'iter0_mask_ave_{}.png'.format(ave_temp[1])), ave_pred_img)
                        


                            
                        tot += lossfunc(ave_pred, masks)



                            
                        mix_res = tuple([sum(a) for a in zip(mix_res, ave_temp)])
                        y1 = torch.cat(uncertainty_list, dim=1)
                        y1 = torch.mean(torch.sigmoid(y1), dim=1)
                        uncertainty_0 = entropy(y1)  # 4numof tta
                        uncertainty_0 = uncertainty_0.squeeze(0)

                        # 绘制热力图
                        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
                        plt.imshow(uncertainty_0, cmap=plt.cm.jet,norm=norm)  # 使用热色图作为颜色映射
                        plt.axis('off')  # 关闭坐标轴
                        plt.savefig(os.path.join(sub_path_list[3], 'iter0_hot_uncertainty.png'))

                        uncertainty0_png = (uncertainty_0 * 255).astype(np.uint8) 
                        
                        cv2.imwrite(os.path.join(sub_path_list[3], 'iter0_uncertainty_{}_{}.png'.format(uncertainty_0.sum(),uncertainty_0.sum()/(ave_pred.sum()))), uncertainty0_png)
                        
                        uncertainty1_ave.append(uncertainty_0.sum())
                        ratio1 = uncertainty_0.sum()/ave_pred_binary.sum()

                        print(f'{name} : before : dice1 = {ave_temp[1]},iou={ave_temp[0]}')
                        # df = df._append({'File Name': name,'DICE_0 Score ': ave_temp[1],'IOU_0 Score': ave_temp[0],'uncertainty_0 Score':uncertainty_0.sum(),'ratio1':ratio1}, ignore_index=True)
                        # df.to_csv(os.path.join(args.path_helper['prefix'],'output_values.csv'), index=False)
                    
    


            masked_uncertainty_map,FN_output_mask, FN_UH, FN_UH_png,FN_xUH,FN_xUH_png,FN_output_mask, FP_UH,FP_xUH_png,FN_condition_mask,FP_condition_mask  = uc_refine_correct_FN_add_FP(ave_pred_binary.copy(),
                                                                        uncertainty_0.copy(),
                                                                       img_256_np.copy(),sub_path_list)
            
            
            # 产生points和points_labels
            indices_FN = np.argwhere(masked_uncertainty_map.squeeze(0) > 0)
            indices_FP = np.argwhere(FP_xUH_png>0)
            random_index_FN = np.random.choice(len(indices_FN), int(len(indices_FN)),replace=False)

            selected_points_FN = [indices_FN[i] for i in random_index_FN] #以行，列的形式存储

            random_index_FP = np.random.choice(len(indices_FP), int(len(indices_FP)),replace=False)

            selected_points_FP = [indices_FP[i] for i in random_index_FP] #以行，列的形式存储


            uncertainty_values_FN = []  # 初始化最大不确定性值为负无穷
            uncertainty_values_FP = []
            # max_uncertainty_point = None  # 初始化最大不.确定性值对应的点为None
            points_list = []
            image2 = imgs_np.copy()


            for i in range(len(selected_points_FN)):
                uncertainty_value_FN = get_uncertainty_value(uncertainty_0.copy(),imgs_np.copy(), selected_points_FN[i],sub_path_list,i+1)
                uncertainty_values_FN.append((selected_points_FN[i], uncertainty_value_FN,'FN'))
        
            scale_x = 1024//256
            for i in range(len(uncertainty_values_FN)):


                max_uncertainty_points_FN = sorted(uncertainty_values_FN, key=lambda x: x[1], reverse=True)[i]

                y1,x1 = list(max_uncertainty_points_FN)[0]
                break

                # if x_1largest <=  x1 <= x_2largest and y1 <= y1_largest <= y_2largest:
            points_list.append([y1*scale_x,x1*scale_x])  #点为行，列
            cv2.circle(image2,(x1,y1),3,(255,0,0),-1) #opencv中点为列，行
             
            
            input_point = points_list
           
            cv2.imwrite(os.path.join(sub_path_list[2], 'max_points_uncertainty_{}.png'.format(max_uncertainty_points_FN[0][1])), image2)
            cv2.imwrite(os.path.join(sub_path_list[0], 'iter1_point.png'), imgs_2)

            # if label_0_1 == 'FN':
            input_label = torch.tensor([1])
            if input_label.clone().flatten()[0] != -1:
                # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                point_coords = input_point
                coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                labels_torch = torch.as_tensor(input_label, dtype=torch.int, device=GPUdevice)
                if(len(input_label.shape)==1): # only one point prompt
                    coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                pt = (coords_torch, labels_torch)          
        # else:
                # input_label = np.array([0])  
            pred_uncertainty_list2 = []

                
            # aug_prompt1 = prompt_aug1(prompt=box1, target=gt, aug=True,scale_factor=256)
            with torch.no_grad():
                imge= net.image_encoder(imgs)
                # x1,y1,x2,y2 = box[t][0]
                y1_1,x1_1 = points_list[0]
                cv2.circle(imgs_2,(int(x1_1.item()),int(y1_1.item())),5,(255,0,0),-1)
          
                cv2.imwrite(os.path.join(sub_path_list[0], 'prompt.png'), imgs_2)
                for t in range(len(box)):
                    # x1,y1,x2,y2 = box[i][0]




                    se, de = net.prompt_encoder(
                        points=pt,
                        boxes=box[t].to(dtype = torch.int64, device = GPUdevice),
                        masks=ave_pred,
                    )
        

            
                    pred, _ = net.mask_decoder(
                        image_embeddings=imge,
                        image_pe=net.prompt_encoder.get_dense_pe(), 
                        sparse_prompt_embeddings=se,
                        dense_prompt_embeddings=de, 
                        multimask_output=(args.multimask_output > 1),
                    )
            
                    # Resize to the ordered output size
                    pred1 = F.interpolate(pred,size=(args.out_size,args.out_size))
                    temp1 = eval_seg(pred1, masks, threshold)


                    pred_img1= np.where((pred1.squeeze(0).cpu().numpy())>=0.5,True,False)
                    pred_img1 =(pred_img1*255).astype(np.uint8).transpose(1,2,0)

                    
                    cv2.imwrite(os.path.join(sub_path_list[2], 'iter1_mask_{}_{}.png'.format(t+1,temp1[1])), pred_img1)

                    pred_list1.append(pred1)
                    uncertainty_list1.append(pred1)
                
                ave_pred1 = torch.mean(torch.cat(pred_list1,dim=1),dim=1,keepdim=True)

                ave_temp1 = eval_seg(ave_pred1, masks, threshold)

                dice1_ave.append(ave_temp1[1])
                iou1_ave.append(ave_temp1[0])

                ave_pred_binary1 = np.where((ave_pred1.squeeze(0).cpu().numpy())>=0.5,True,False)
                ave_pred_img1 =(ave_pred_binary1*255).astype(np.uint8).transpose(1,2,0)
                cv2.imwrite(os.path.join(sub_path_list[2], 'iter1_mask_ave_{}.png'.format(ave_temp1[1])), ave_pred_img1)
                y2 = torch.cat(uncertainty_list1, dim=1)
                y2 = torch.mean(torch.sigmoid(y2), dim=1)
                uncertainty_1 = entropy(y2)  # 4numof tta
                uncertainty_1 = uncertainty_1.squeeze(0)

                # 绘制热力图
                norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
                plt.imshow(uncertainty_1, cmap=plt.cm.jet,norm=norm)  # 使用热色图作为颜色映射
                plt.axis('off')  # 关闭坐标轴
                plt.savefig(os.path.join(sub_path_list[3], 'iter0_hot_uncertainty.png'))

                uncertainty1_png = (uncertainty_1 * 255).astype(np.uint8) 
                
                cv2.imwrite(os.path.join(sub_path_list[3], 'iter1_uncertainty_{}_{}.png'.format(uncertainty_1.sum(),uncertainty_1.sum()/(ave_pred1.sum()))), uncertainty1_png)
                
                uncertainty2_ave.append(uncertainty_1.sum())
                ratio2 = uncertainty_1.sum()/ave_pred_binary1.sum()

            
                    
                tot += lossfunc(pred1, masks)



             
                print(f'{name} : after : dice1 = {ave_temp1[1]},iou1={ave_temp1[0]}')


                if ratio1 > ratio2:
                    count +=1
            df = df._append({'File Name':name,'DICE_0 Score ': ave_temp[1],'IOU_0 Score': ave_temp[0],'uncertainty_0 Score':uncertainty_0.sum(),'ratio1':ratio1,'DICE_1 Score ': ave_temp1[1],'IOU_1 Score': ave_temp1[0],'uncertainty_1 Score':uncertainty_1.sum()}, ignore_index=True)
            df.to_csv(os.path.join(args.path_helper['prefix'],'output_values.csv'), index=False)

           

        




               
                    
            pbar.update()
            # break
    
    dice0_score = np.array(dice0_ave)
    iou0_score = np.array(iou0_ave)

    uncertainty1_score = np.array(uncertainty1_ave)

    print('----------Have finished testing --------------------')
    print('The Ave_Dice:{}'.format(dice0_score.mean(axis=0)))
    print('The Ave_Iou:{}'.format(iou0_score.mean(axis=0)))
    print('The Ave_uncertainty:{}'.format(uncertainty1_score.mean(axis=0)))
    print('---------------------------------------------------')

    df = df._append({'File Name': name,'Dice0_ave': dice0_score.mean(axis=0),'IOU0_ave': iou0_score.mean(axis=0)}, ignore_index=True)
    df.to_csv(os.path.join(args.path_helper['prefix'],'output_values.csv'), index=False)

    dice1_score = np.array(dice1_ave)
    iou1_score = np.array(iou1_ave)
    uncertainty2_score = np.array(uncertainty2_ave)

    print('----------Have finished testing --------------------')
    print('The Ave_Dice:{}'.format(dice1_score.mean(axis=0)))
    print('The Ave_Iou:{}'.format(iou1_score.mean(axis=0)))
    print('The Ave_uncertainty:{}'.format(uncertainty2_score.mean(axis=0)))
    print('---------------------------------------------------')

    print('uncertainty down count: {}'.format(count))

    df = df._append({'File Name':name,'Dice1_ave': dice1_score.mean(axis=0),'IOU1_ave': iou1_score.mean(axis=0)}, ignore_index=True)
    df.to_csv(os.path.join(args.path_helper['prefix'],'output_values.csv'), index=False)

    if args.evl_chunk:
        n_val = n_val * (imgsw.size(-1) // evl_ch)

    return tot/ n_val , tuple([a/n_val for a in mix_res])


def validation_medsam_point_box(args, val_loader, epoch, net: nn.Module, clean_dir=True):
     # eval mode
    net.eval()

    mask_type = torch.float32
    n_val = len(val_loader)  # the number of batch
    ave_res, mix_res = (0,0,0,0), (0,)*args.multimask_output*2
    tot = 0
    hard = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))

    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        lossfunc = criterion_G
    l = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            
            imgsw = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            masksw = pack['label'].to(dtype = torch.float32, device = GPUdevice)
         
            if 'pt' not in pack:
                imgs, pt, masks = generate_click_prompt(imgs, masks)
            elif 'box' in pack:
                ptw = pack['pt']
                point_labels = pack['p_label']
                box = pack['box'].to(dtype = torch.int64, device = GPUdevice)
            name = pack['image_meta_dict']['filename_or_obj']
            
            if len(pack["box"][0]) != 4 or len(pack['box'][1])!=4:
                continue
            l +=1
            buoy = 0
            if args.evl_chunk:
                evl_ch = int(args.evl_chunk)
            else:
                evl_ch = int(imgsw.size(-1))

            while (buoy + evl_ch) <= imgsw.size(-1):
                if args.thd:
                    pt = ptw[:,:,buoy: buoy + evl_ch]
                else:
                    pt = ptw

                imgs = imgsw[...,buoy:buoy + evl_ch]
                masks = masksw[...,buoy:buoy + evl_ch]
                buoy += evl_ch

                if args.thd:
                    pt = rearrange(pt, 'b n d -> (b d) n')
                    imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                    masks = rearrange(masks, 'b c h w d -> (b d) c h w ')
                    imgs = imgs.repeat(1,3,1,1)
                    point_labels = torch.ones(imgs.size(0))

                    imgs = torchvision.transforms.Resize((args.image_size,args.image_size))(imgs)
                    masks = torchvision.transforms.Resize((args.out_size,args.out_size))(masks)
                
                showp = pt

                mask_type = torch.float32
                ind += 1
                b_size,c,w,h = imgs.size()
                longsize = w if w >=h else h

                if point_labels.clone().flatten()[0] != -1:
                    # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                    point_coords = pt
                    coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                    labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                    if(len(point_labels)==1): # only one point prompt
                        coords_torch, labels_torch, showp = coords_torch[None, :, :], labels_torch[None, :], showp[None, :, :]
                        pt = (coords_torch, labels_torch)
                    else:
                        pt = (coords_torch.unsqueeze(1), labels_torch.unsqueeze(1))

                '''init'''
                if hard:
                    true_mask_ave = (true_mask_ave > 0.5).float()
                    #true_mask_ave = cons_tensor(true_mask_ave)
                imgs = imgs.to(dtype = mask_type,device = GPUdevice)
                
                '''test'''
                with torch.no_grad():
                    imge= net.image_encoder(imgs)
                    

                    if args.net == 'sam' or args.net == 'mobile_sam' or args.net =='medsam' and 'box' in pack:
                        # for i in range(len(box)):
                        se, de = net.prompt_encoder(
                            points=pt,
                            boxes=box,
                            masks=None,
                        )
                

                    if args.net == 'sam' or args.net =='medsam':
                        pred, _ = net.mask_decoder(
                            image_embeddings=imge,
                            image_pe=net.prompt_encoder.get_dense_pe(), 
                            sparse_prompt_embeddings=se,
                            dense_prompt_embeddings=de, 
                            multimask_output=(args.multimask_output > 1),
                        )
                
                        # Resize to the ordered output size
                        pred = F.interpolate(pred,size=(args.out_size,args.out_size))
                        tot += lossfunc(pred, masks)


                        temp = eval_seg(pred, masks, threshold)
                        mix_res = tuple([sum(a) for a in zip(mix_res, temp)])


                    '''vis images'''
                    if ind % 50 == 0:
                        namecat = 'Test'
                        for na in name[:2
                        
                        ]:
                            img_name = na.split('/')[-1].split('.')[0]
                            namecat = namecat + img_name
                        epoch_path = make_sub_folder(os.path.join(args.path_helper['sample_path'],'Test','epoch_{}'.format(epoch)),namecat)
                        vis_image_isic_png(imgs,pred,masks, epoch_path,temp[1],temp[0], reverse=False, points=showp,box=box)
                    

                    
            pbar.update()
            # break
    if args.evl_chunk:
        n_val = l*2

    return tot/ n_val , tuple([a/n_val for a in mix_res])



def validation_sam(args, val_loader, epoch, net: nn.Module, clean_dir=True):
     # eval mode
    net.eval()

    mask_type = torch.float32
    n_val = len(val_loader)  # the number of batch
    ave_res, mix_res = (0,0,0,0), (0,)*args.multimask_output*2
    rater_res = [(0,0,0,0) for _ in range(6)]
    tot = 0
    hard = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice

    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        lossfunc = criterion_G

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            imgsw = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            masksw = pack['label'].to(dtype = torch.float32, device = GPUdevice)
            # for k,v in pack['image_meta_dict'].items():
            #     print(k)
            if 'pt' not in pack:
                imgs, pt, masks = generate_click_prompt(imgs, masks)
            elif 'box' in pack:
                ptw = pack['pt']
                point_labels = pack['p_label']
                box = pack['box'].to(dtype = torch.int64, device = GPUdevice)
            name = pack['image_meta_dict']['filename_or_obj']
            
            buoy = 0
            if args.evl_chunk:
                evl_ch = int(args.evl_chunk)
            else:
                evl_ch = int(imgsw.size(-1))

            while (buoy + evl_ch) <= imgsw.size(-1):
                if args.thd:
                    pt = ptw[:,:,buoy: buoy + evl_ch]
                else:
                    pt = ptw

                imgs = imgsw[...,buoy:buoy + evl_ch]
                masks = masksw[...,buoy:buoy + evl_ch]
                buoy += evl_ch

                if args.thd:
                    pt = rearrange(pt, 'b n d -> (b d) n')
                    imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                    masks = rearrange(masks, 'b c h w d -> (b d) c h w ')
                    imgs = imgs.repeat(1,3,1,1)
                    point_labels = torch.ones(imgs.size(0))

                    imgs = torchvision.transforms.Resize((args.image_size,args.image_size))(imgs)
                    masks = torchvision.transforms.Resize((args.out_size,args.out_size))(masks)
                
                showp = pt

                mask_type = torch.float32
                ind += 1
                b_size,c,w,h = imgs.size()
                longsize = w if w >=h else h

                if point_labels.clone().flatten()[0] != -1:
                    # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                    point_coords = pt
                    coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                    labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                    if(len(point_labels.shape)==1): # only one point prompt
                        coords_torch, labels_torch, showp = coords_torch[None, :, :], labels_torch[None, :], showp[None, :, :]
                    pt = (coords_torch, labels_torch)

                '''init'''
                if hard:
                    true_mask_ave = (true_mask_ave > 0.5).float()
                    #true_mask_ave = cons_tensor(true_mask_ave)
                imgs = imgs.to(dtype = mask_type,device = GPUdevice)
                
                '''test'''
                with torch.no_grad():
                    imge= net.image_encoder(imgs)
                    if args.net == 'sam' or args.net == 'mobile_sam' or args.net =='medsam' and 'box' not in pack:
                        se, de = net.prompt_encoder(
                            points=pt,
                            boxes=None,
                            masks=None,
                        )

                    elif args.net == 'sam' or args.net == 'mobile_sam' or args.net =='medsam' and 'box' in pack:
                        se, de = net.prompt_encoder(
                            points=pt,
                            boxes=box,
                            masks=None,
                        )
                    elif args.net == "efficient_sam":
                        coords_torch,labels_torch = transform_prompt(coords_torch,labels_torch,h,w)
                        se = net.prompt_encoder(
                            coords=coords_torch,
                            labels=labels_torch,
                        )

                    if args.net == 'sam' or args.net =='medsam':
                        pred, _ = net.mask_decoder(
                            image_embeddings=imge,
                            image_pe=net.prompt_encoder.get_dense_pe(), 
                            sparse_prompt_embeddings=se,
                            dense_prompt_embeddings=de, 
                            multimask_output=(args.multimask_output > 1),
                        )
                    elif args.net == 'mobile_sam':
                        pred, _ = net.mask_decoder(
                            image_embeddings=imge,
                            image_pe=net.prompt_encoder.get_dense_pe(), 
                            sparse_prompt_embeddings=se,
                            dense_prompt_embeddings=de, 
                            multimask_output=False,
                        )
                    elif args.net == "efficient_sam":
                        se = se.view(
                            se.shape[0],
                            1,
                            se.shape[1],
                            se.shape[2],
                        )
                        pred, _ = net.mask_decoder(
                            image_embeddings=imge,
                            image_pe=net.prompt_encoder.get_dense_pe(), 
                            sparse_prompt_embeddings=se,
                            multimask_output=False,
                        )

                    # Resize to the ordered output size
                    pred = F.interpolate(pred,size=(args.out_size,args.out_size))
                    tot += lossfunc(pred, masks)


                    temp = eval_seg(pred, masks, threshold)
                    mix_res = tuple([sum(a) for a in zip(mix_res, temp)])


                    '''vis images'''
                    if ind % args.vis == 0:
                        namecat = 'Test'
                        for na in name[:2
                        
                        ]:
                            img_name = na.split('/')[-1].split('.')[0]
                            namecat = namecat + img_name
                        epoch_path = make_sub_folder(os.path.join(args.path_helper['sample_path'],'Test'),namecat)
                        vis_image_isic_png(imgs,pred,masks, epoch_path,temp[1],temp[0], reverse=False, points=showp,box=box)
                    

                    
            pbar.update()

    if args.evl_chunk:
        n_val = n_val * (imgsw.size(-1) // evl_ch)

    return tot/ n_val , tuple([a/n_val for a in mix_res])

def transform_prompt(coord,label,h,w):
    coord = coord.transpose(0,1)
    label = label.transpose(0,1)

    coord = coord.unsqueeze(1)
    label = label.unsqueeze(1)

    batch_size, max_num_queries, num_pts, _ = coord.shape
    num_pts = coord.shape[2]
    rescaled_batched_points = get_rescaled_pts(coord, h, w)

    decoder_max_num_input_points = 6
    if num_pts > decoder_max_num_input_points:
        rescaled_batched_points = rescaled_batched_points[
            :, :, : decoder_max_num_input_points, :
        ]
        label = label[
            :, :, : decoder_max_num_input_points
        ]
    elif num_pts < decoder_max_num_input_points:
        rescaled_batched_points = F.pad(
            rescaled_batched_points,
            (0, 0, 0, decoder_max_num_input_points - num_pts),
            value=-1.0,
        )
        label = F.pad(
            label,
            (0, decoder_max_num_input_points - num_pts),
            value=-1.0,
        )
    
    rescaled_batched_points = rescaled_batched_points.reshape(
        batch_size * max_num_queries, decoder_max_num_input_points, 2
    )
    label = label.reshape(
        batch_size * max_num_queries, decoder_max_num_input_points
    )

    return rescaled_batched_points,label


def get_rescaled_pts(batched_points: torch.Tensor, input_h: int, input_w: int):
        return torch.stack(
            [
                torch.where(
                    batched_points[..., 0] >= 0,
                    batched_points[..., 0] * 1024 / input_w,
                    -1.0,
                ),
                torch.where(
                    batched_points[..., 1] >= 0,
                    batched_points[..., 1] * 1024 / input_h,
                    -1.0,
                ),
            ],
            dim=-1,
        )