import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from utils import ramps, losses, test_patch
from dataloaders.dataset import *
from networks.net_factory import net_factory
import torch.multiprocessing as mp
import time
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str,  default='CoactSeg', help='name')
parser.add_argument('--root_path', type=str, default='./', help='Name of Dataset')
parser.add_argument('--exp', type=str,  default='reg', help='exp_name')
parser.add_argument('--model', type=str,  default='vnet', help='model_name')
parser.add_argument('--max_iteration', type=int,  default=2000, help='maximum iteration to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size of labeled data per gpu')
parser.add_argument('--base_lr', type=float,  default=0.001, help='learning rate')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--patch_size', type=int,  default=80, help='the size of patch')
parser.add_argument('--loss_func', type=str, default="dice", help="which loss function to use: options are specified in the README file")
parser.add_argument('--act', type=str, default='relu', help="activation function")
args = parser.parse_args()

snapshot_path = args.root_path + "model/{}_{}/{}".format(args.name, args.exp, args.model)
loss_logger_path = args.root_path + "code/utils/Experiments/"

num_classes = 2
patch_size = (args.patch_size, args.patch_size, args.patch_size)
args.root_path = args.root_path+"data/"
train_data_path = args.root_path 

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
max_iterations = args.max_iteration
base_lr = args.base_lr
batch_size = args.batch_size
secondary_batch_size = 1

loss_function = args.loss_func
activation = args.act

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)


if __name__ == "__main__":

    # start method
    mp.set_start_method("spawn")

    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('./code/', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    transforms = transforms.Compose([
                        RandomRotFlip(),
                        RandomRot(),
                        WeightCrop(patch_size),
                        ToTensor(),
                        ])
    weights = torch.load("./pretrained_pth/CoactSeg_reg/vnet/vnet_best_model_copy.pth", map_location="cuda", weights_only=True)
    model = net_factory(net_type=args.model, in_chns=3, class_num=num_classes, mode="train", act=activation)
    model_dict = model.state_dict()
    pretrained_dict = {
        k:v for k, v in weights.items()
        if k in model_dict and model_dict[k].shape == v.shape
    }
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.cuda()

    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.decoder.parameters():
        param.requires_grad = True

    # with augmentation
    db_train = MS(base_dir=train_data_path,
                    split='train',
                    transform = transforms,
                    cda=False
                )
    
    labeled_idxs = list(range(32))
    # this is the original sampler, we wrote a modified version as we didn't possess the single-time point samples used by the original
    # paper i.e CoactSeg
    # unlabeled_idxs = list(range(28, 32))
    # batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, secondary_batch_size=secondary_batch_size)
    batch_sampler = BatchSampler(labeled_idxs, batch_size)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=12)
    
    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    #optimizer = optim.AdamW(model.parameters(), lr=base_lr)

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))
    iter_num = 0
    best_dice = 0
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr
    iterator = tqdm(range(max_epoch), ncols=70)

    best_dice_to_log = 0
    final_dice = 0
    epoch_of_best_dice = 0
    for epoch_num in iterator:

        steps = 0
        start = 0
        end = 0
        total_step_time = 0
  

        for i_batch, sampled_batch in enumerate(trainloader):
            
            volume_batch_1, volume_batch_2, label_batch = sampled_batch['image_1'], sampled_batch['image_2'], sampled_batch['label']

            #longitudinal difference map
            volume_batch_sub = volume_batch_2 - volume_batch_1 
            volume_batch = torch.cat([volume_batch_1, volume_batch_2, volume_batch_sub], dim=1)

            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            


            with torch.amp.autocast(device_type='cuda'):    

                start = time.time()
                model.train()
                outputs_1, outputs_2, outputs_3 = model(volume_batch)
                


                y1 = F.softmax(outputs_1, dim=1)
                y2 = F.softmax(outputs_2, dim=1)
                y3 = F.softmax(outputs_3, dim=1)

                print(f"\ny3 contains NaN: {torch.isnan(y3).any()}, Inf: {torch.isinf(y3).any()}")
                print(f"\ny3 min: {y3.min()}, y3 max: {y3.max()}, y3 mean: {y3.mean()}")

                # for MSSEG'2
                # only containing the new lesion labels
                label_new_lesions = label_batch[:batch_size,...]
                # logging.info("label_new_lesions:{}".format(label_new_lesions))


                #logging.info("y1[:batch_size,1,...]: {}".format(y1[:batch_size,1,...]))
                selected_new_lesions_y1 = torch.masked_select(y1[:batch_size,1,...], label_new_lesions==1)
                selected_new_lesions_y2 = torch.masked_select(y2[:batch_size,1,...], label_new_lesions==1)
                # logging.info("\nselected_new_lesion_y1: {}".format(selected_new_lesions_y1))
                # logging.info("\nselected_new_lesion_y2:{}".format(selected_new_lesions_y2))

                selected_new_lesions_gt = torch.masked_select(label_new_lesions, label_new_lesions==1)


                
                loss_reg_pseudo = losses.mse_loss(selected_new_lesions_y1, 1-selected_new_lesions_gt) + losses.mse_loss(selected_new_lesions_y2, selected_new_lesions_gt)
            
                
                iter_num = iter_num + 1
                

                match loss_function:
                    case "crossEntropy":
                        loss =  F.cross_entropy(outputs_3[:batch_size,...], label_new_lesions)
                    case "dice":
                        loss = losses.Binary_dice_loss(y3[:batch_size,1,...], label_new_lesions == 1)
                    case "focal":
                        loss = losses.focal_loss(y3[:batch_size,1,...], label_new_lesions)
                    case "focalDice":
                        loss = losses.Binary_dice_loss(y3[:batch_size,1,...], label_new_lesions == 1) + losses.focal_loss(y3[:batch_size,1,...], label_new_lesions)
                    case "symmetricFocal":
                        loss = losses.FocalLoss().forward(y3[:batch_size, 1, ...], label_new_lesions==1, type="symmetric")
                    case "asymmetricFocal":
                        loss = losses.FocalLoss().forward(y3[:batch_size, 1, ...], label_new_lesions==1, type="asymmetric")
                    case "symmetricFocTv":
                        loss = losses.FocalTversky().forward(y3[:batch_size, 1, ...], label_new_lesions == 1, type="symmetric")
                    case "asymmetricFocTv":
                        loss = losses.FocalTversky().forward(y3[:batch_size, 1, ...], label_new_lesions == 1, type="asymmetric")
                    case "symmetricUnifiedFoc":
                        loss = losses.UnifiedFocal().forward(y3[:batch_size, 1, ...], label_new_lesions == 1, type="symmetric")
                    case "asymmetricUnifiedFoc":
                        loss =  losses.UnifiedFocal().forward(y3[:batch_size, 1, ...], label_new_lesions == 1, type="asymmetric")
                    case "tversky":
                        loss = losses.Tversky(type="tversky").forward(y3[:batch_size, 1, ...], label_new_lesions)
                    case "focalTversky":
                        loss = losses.Tversky(type="focalTversky").forward(y3[:batch_size, 1, ...], label_new_lesions)
                    case "dicePP":
                        loss = losses.DicePP().forward(y3[:batch_size,1,...], label_new_lesions)
                    case "logcosh":
                        loss = losses.LogCoshLoss().forward(y3[:batch_size,1,...], label_new_lesions)
                    case "hdloss":
                        loss = losses.HausdorffDTLoss().forward(y3[:batch_size,1,...], label_new_lesions)
                    case "comboloss":
                        loss = losses.Combo_loss(y3[:batch_size,1,...], label_new_lesions) 
                    case "hyTver":
                        loss = losses.hyTver(y3[:batch_size,1,...], label_new_lesions)
                    case "weightedce":
                        loss = losses.WeightedCrossEntropyLoss(weight=torch.tensor([0.1, 0.9]).cuda()).forward(outputs_3[:batch_size,...], label_new_lesions)

                if iter_num < 100:
                    loss = loss
                    # loss = loss_seg_dice_public + loss_focal_public
                else:
                    loss = loss + loss_reg_pseudo

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # logging.info('\niteration %d : loss : %03f, loss_seg_dice_public: %03f, loss_seg_public: %03f, loss_reg_pseudo: %03f' % (iter_num, loss, loss_seg_dice_public, loss_seg_public, loss_reg_pseudo))
                logging.info('\niteration %d : loss : %03f' % (iter_num, loss))
                if iter_num % 5 == 0 or iter_num == 1:
                    fileName = loss_logger_path + "{}_{}.csv".format(loss_function, activation)
                    with open(fileName, 'a') as f:
                            write = csv.writer(f)
                            write.writerow([iter_num, loss.item()])

                # measure time taken 
                end = time.time()
                total_step_time += end-start
                steps += 1

                avg_step_time = total_step_time / steps
                if iter_num == 1 or iter_num % 5 == 0:
                    with open(loss_logger_path + "{}_stepTime.txt".format(loss_function), 'a') as f:
                        f.writelines('iter:{}: {}\n'.format(iter_num, avg_step_time))


                if iter_num >= 0:#200 and iter_num % 200 == 0:
                    sample_index = np.random.randint(0, batch_size)
                    img_double = ramps.get_imgs(y1, y2, y3, volume_batch, label_batch, sample_index)
                    writer.add_images('Epoch_%d_Iter_%d_Double'% (epoch_num, iter_num), img_double)
                    # sample_index = np.random.randint(batch_size, 2*batch_size)
                    # img_single = ramps.get_imgs(y1, y2, y3, volume_batch, label_batch, sample_index)
                    # writer.add_images('Epoch_%d_Iter_%d_Single'% (epoch_num, iter_num), img_single)

                
                if iter_num >= 0:#5000 and iter_num % 200 == 0:
                    model.eval()
                    dice_sample = test_patch.var_all_case(model, num_classes=num_classes, patch_size=patch_size, stride_xy=20, stride_z=20, logger_path=loss_logger_path, loss_name=loss_function, iter = iter_num, act=activation)
                    final_dice = dice_sample
                    if dice_sample > best_dice:
                        best_dice = dice_sample
                        best_dice_to_log = best_dice
                        epoch_of_best_dice = iter_num
                        save_mode_path = os.path.join(snapshot_path,  'iter_{}_{}_{}.pth'.format(iter_num,loss_function, best_dice))
                        save_best_path = os.path.join(snapshot_path,'{}_{}_{}_best_model.pth'.format(args.model, loss_function, activation))
                        torch.save(model.state_dict(), save_mode_path)
                        torch.save(model.state_dict(), save_best_path)
                        logging.info("save best model to {}".format(save_mode_path))
                    writer.add_scalar('Var_dice/Dice', dice_sample, iter_num)
                    writer.add_scalar('Var_dice/Best_dice', best_dice, iter_num)

                    model.train()



                if iter_num >= max_iterations:
                    save_mode_path = os.path.join(snapshot_path, str(loss_function)+'_iter_' + str(iter_num) + '.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    logging.info("save model to {}".format(save_mode_path))
                    logging.info("epoch of best dice: {}, best dice: {}".format(epoch_of_best_dice, best_dice_to_log))
                    with open(loss_logger_path +"_"+activation+"_"+ "performance.txt", 'a') as f:
                        f.writelines('Iteration of best dice for {}:{}: Best Dice: {}: final_dice: {}\n'.format(epoch_of_best_dice, loss_function, best_dice_to_log, final_dice))

                    break


        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()