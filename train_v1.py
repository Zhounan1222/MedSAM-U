import os
import time

import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from tensorboardX import SummaryWriter
from tqdm import tqdm

from cfg import cfg
import function.function_medsam_point_box as function_medsam
from conf import settings
#from models.discriminatorlayer import discriminator
from dataset import *
from utils import *
# from model_pa_sam.model.mask_decoder_pa import MaskDecoderPA
from utils import *
def main():

    args = cfg.parse_args()

    GPUdevice = torch.device('cuda:{}'.format(args.gpu_device))

    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)
    if args.pretrain:
        weights = torch.load(args.pretrain)
        net.load_state_dict(weights,strict=False)

    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    '''load pretrained model'''
    if args.weights != 0:
        print(f'=> resuming from {args.weights}')
        assert os.path.exists(args.weights)
        checkpoint_file = os.path.join(args.weights)
        assert os.path.exists(checkpoint_file)
        loc = 'cuda:{}'.format(args.gpu_device)
        checkpoint = torch.load(checkpoint_file, map_location=loc)
        start_epoch = checkpoint['epoch']

        net.load_state_dict(checkpoint['state_dict'],strict=False)

        args.path_helper = checkpoint['path_helper']
        logger = create_logger(args.path_helper['log_path'])
        print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')

    args.path_helper = set_log_dir('./', args.exp_name)
    logger = create_logger(args.path_helper['log_path'])
    logger.info(args)

    nice_train_loader, nice_valid_loader = get_dataloader(args)

    '''checkpoint path and tensorboard'''
    # iter_per_epoch = len(Glaucoma_training_loader)
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW))
    # input_tensor = torch.Tensor(args.b, 3, 256, 256).cuda(device = GPUdevice)
    # writer.add_graph(net, Variable(input_tensor, requires_grad=True))

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    '''begain training'''
    best_acc = 0.0
    best_tol = 1e4
    best_dice = 0.0



    for epoch in range(settings.EPOCH):
        # if epoch and epoch <2:
        if epoch and epoch < 5:
            function_medsam.validation_medsam_point_box(args, nice_valid_loader, epoch, net, writer)
            logger.info(f'have valid ')
        net.train()
        time_start = time.time()
        loss = function_medsam.train_medsam_point_box(args, net,optimizer, nice_train_loader, epoch, writer, vis = args.vis)
        

        writer.add_scalar('Loss/train', loss.item(), epoch)

    

        logger.info(f'Train loss: {loss} || @ epoch {epoch}.')
        time_end = time.time()
        print('time_for_training ', time_end - time_start)

        net.eval()
        if epoch and epoch % args.val_freq == 0 or epoch == settings.EPOCH-1:
            
            iou = function_medsam.validation_medsam_point_box(args, nice_valid_loader, epoch, net, writer)
            logger.info(f'have valid ')
            if args.distributed != 'none':
                sd = net.module.state_dict()
            else:
                sd = net.state_dict()

            if iou > best_dice:
                is_best = True

                best_dice = iou

                save_checkpoint({
                'epoch': epoch + 1,
                'model': args.net,
                'state_dict': sd,
                'optimizer': optimizer.state_dict(),
                'best_tol': best_dice,
                'path_helper': args.path_helper,
            }, is_best, args.path_helper['ckpt_path'], filename="best_dice_checkpoint.pth")
            else:
                if epoch % 5 ==0:
                    is_best = False
                    file_name = '{}_{}_dice_ckpt.pth'.format(epoch,iou)
                    save_checkpoint({
                    'epoch': epoch + 1,
                    'model': args.net,
                    'state_dict': sd,
                    'optimizer': optimizer.state_dict(),
                    'best_tol': iou,
                    'path_helper': args.path_helper,
                }, is_best, args.path_helper['ckpt_path'], filename=file_name)

    writer.close()


if __name__ == '__main__':
    main()
