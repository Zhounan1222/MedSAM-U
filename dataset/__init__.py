import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from albumentations import GaussNoise

from utils import *

from .atlas import Atlas
from .brat import Brat
from .ddti_2 import DDTI_2
from .isic import ISIC2016,ISIC2017,ISIC2017_pointbox,ISIC2017_box
from .kits import KITS
from .lidc import LIDC
from .lnq import LNQ
from .pendal import Pendal
from .refuge_own import REFUGE_box,REFUGE_point_box
from .segrap import SegRap
from .stare import STARE
from .toothfairy import ToothFairy
from .wbc import WBC

from .all_in_one import isic_refuge_ddti_all_in_once,all_in_onece_5_datasets


def get_dataloader(args):
    transform_train = transforms.Compose([
        transforms.Resize((args.image_size,args.image_size)),
        transforms.ToTensor(),
    ])

    transform_train_seg = transforms.Compose([
        transforms.Resize((args.out_size,args.out_size)),
        transforms.ToTensor(),
    ])

    transform_valid = transforms.Compose([
        transforms.Resize((args.image_size,args.image_size)),
        transforms.ToTensor(),
    ])

    transform_valid_seg = transforms.Compose([
        transforms.Resize((args.out_size,args.out_size)),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])



    transform_test_seg = transforms.Compose([
        transforms.Resize((args.out_size,args.out_size)),
        transforms.ToTensor(),
    ])

 
    
    if args.dataset == 'isic_2016':
        '''isic data'''
        isic_train_dataset = ISIC2016(args, args.data_path, transform = transform_train, transform_msk= transform_train_seg, mode = 'Train')
        isic_test_dataset = ISIC2016(args, args.data_path, transform = transform_test, transform_msk= transform_test_seg, mode = 'Test')

        nice_train_loader = DataLoader(isic_train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(isic_test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)
        
        '''end'''
    elif args.dataset == 'isic_2017_point':
        '''isic data'''
        isic_train_dataset = ISIC2017(args, args.data_path, transform = transform_train, transform_msk= transform_train_seg, mode = 'Train')
        isic_valid_dataset = ISIC2017(args, args.data_path, transform = transform_valid, transform_msk= transform_valid_seg, mode = 'Valid')
        # isic_test_dataset = ISIC2017(args, args.data_path, transform = transform_test, transform_msk= transform_test_seg, mode = 'Test')

        nice_train_loader = DataLoader(isic_train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
        nice_valid_loader = DataLoader(isic_valid_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)
        # nice_test_loader = DataLoader(isic_test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)

    elif args.dataset == 'isic_2017_point&box':
        '''isic data'''
        isic_train_dataset = ISIC2017_pointbox( args,args.data_path, transform = transform_train, transform_msk = transform_train_seg, mode = 'Train',prompt = 'point&box', plane = False)
        isic_valid_dataset = ISIC2017_pointbox( args,args.data_path, transform = transform_valid, transform_msk = transform_valid_seg, mode = 'Valid',prompt = 'point&box', plane = False)
        isic_test_dataset = ISIC2017_pointbox(args, args.data_path, transform = transform_test, transform_msk= transform_test_seg, mode = 'Test',prompt = 'point&box', plane = False)
        
        nice_train_loader = DataLoader(isic_train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
        nice_valid_loader = DataLoader(isic_valid_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(isic_test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)
    
    elif args.dataset == 'isic_2017_box':
        '''isic data'''
        isic_train_dataset = ISIC2017_box( args,args.data_path, transform = transform_train, transform_msk = transform_train_seg, mode = 'Train',prompt = 'point&box', plane = False)
        isic_valid_dataset = ISIC2017_box( args,args.data_path, transform = transform_valid, transform_msk = transform_valid_seg, mode = 'Valid',prompt = 'point&box', plane = False)
        isic_test_dataset = ISIC2017_box(args, args.data_path, transform = transform_test, transform_msk= transform_test_seg, mode = 'Test',prompt = 'point&box', plane = False)

        nice_train_loader = DataLoader(isic_train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
        nice_valid_loader = DataLoader(isic_valid_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(isic_test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)


    elif args.dataset == 'refuge_box':
        '''REFUGE data'''
        refuge_train_dataset = REFUGE_box(args,os.path.join(args.data_path), transform = transform_train, transform_msk= transform_train_seg, mode = 'Train',prompt='box')
        refuge_valid_dataset = REFUGE_box(args,os.path.join(args.data_path), transform = transform_valid, transform_msk= transform_valid_seg, mode = 'Valid',prompt='box')
        refuge_test_dataset = REFUGE_box(args, os.path.join(args.data_path), transform = transform_test, transform_msk= transform_test_seg, mode = 'Test',prompt='box')

        nice_train_loader = DataLoader(refuge_train_dataset, batch_size=args.b, shuffle=True, num_workers=0, pin_memory=True)
        nice_valid_loader = DataLoader(refuge_valid_dataset, batch_size=args.b, shuffle=True, num_workers=0, pin_memory=True)
        nice_test_loader = DataLoader(refuge_test_dataset, batch_size=args.b, shuffle=False, num_workers=0, pin_memory=True)
    elif args.dataset == 'refuge_point&box':
        '''REFUGE data'''
        refuge_train_dataset = REFUGE_point_box(args,os.path.join(args.data_path), transform = transform_train, transform_msk= transform_train_seg, mode = 'Train',prompt='point&box')
        refuge_valid_dataset = REFUGE_point_box(args,os.path.join(args.data_path), transform = transform_valid, transform_msk= transform_valid_seg, mode = 'Valid',prompt='point&box')
        refuge_test_dataset = REFUGE_point_box(args, os.path.join(args.data_path), transform = transform_test, transform_msk= transform_test_seg, mode = 'Test',prompt='point&box')

        nice_train_loader = DataLoader(refuge_train_dataset, batch_size=args.b, shuffle=True, num_workers=0, pin_memory=True)
        nice_valid_loader = DataLoader(refuge_valid_dataset, batch_size=args.b, shuffle=True, num_workers=0, pin_memory=True)
        nice_test_loader = DataLoader(refuge_test_dataset, batch_size=args.b, shuffle=False, num_workers=0, pin_memory=True)
    
    
    elif args.dataset == 'atlas':
        '''atlas data'''
        dataset = Atlas(args, data_path = args.data_path,transform = transform_train, transform_msk= transform_train_seg)

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.3 * dataset_size))
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices[split:])
        test_sampler = SubsetRandomSampler(indices[:split])

        nice_train_loader = DataLoader(dataset, batch_size=args.b, sampler=train_sampler, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(dataset, batch_size=args.b, sampler=test_sampler, num_workers=8, pin_memory=True)
        '''end'''

    elif args.dataset == 'pendal':
        '''pendal data'''
        dataset = Pendal(args, data_path = args.data_path,transform = transform_train, transform_msk= transform_train_seg)

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.3 * dataset_size))
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices[split:])
        test_sampler = SubsetRandomSampler(indices[:split])

        nice_train_loader = DataLoader(dataset, batch_size=args.b, sampler=train_sampler, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(dataset, batch_size=args.b, sampler=test_sampler, num_workers=8, pin_memory=True)
        '''end'''

    elif args.dataset == 'lnq':
        '''lnq data'''
        dataset = LNQ(args, data_path = args.data_path,transform = transform_train, transform_msk= transform_train_seg)

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.3 * dataset_size))
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices[split:])
        test_sampler = SubsetRandomSampler(indices[:split])

        nice_train_loader = DataLoader(dataset, batch_size=args.b, sampler=train_sampler, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(dataset, batch_size=args.b, sampler=test_sampler, num_workers=8, pin_memory=True)
        '''end'''

    else:
        print("the dataset is not supported now!!!")
        
    return nice_train_loader, nice_valid_loader