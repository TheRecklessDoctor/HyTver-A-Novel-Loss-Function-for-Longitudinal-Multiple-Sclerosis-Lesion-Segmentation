import os
import argparse
import torch
from networks.net_factory import net_factory
from utils.test_patch import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str,  default='CoactSeg', help='name')
parser.add_argument('--root_path', type=str, default='./', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='reg', help='exp_name')
parser.add_argument('--model', type=str,  default='vnet', help='model_name')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--detail', type=int,  default=1, help='print metrics for every samples?')
parser.add_argument('--nms', type=int, default=0, help='apply NMS post-procssing?')
parser.add_argument('--loss_func', type=str, default="dice", help='name of loss function currently testing')
parser.add_argument('--patch_size', type=int, default=80, help="patch size for images")
parser.add_argument('--act', type=str, default="relu", help="activation function")
parser.add_argument('--iter_num', type=int, default=250, help="number of iterations trained")


FLAGS = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = FLAGS.root_path + "model/{}_{}/{}".format(FLAGS.name, FLAGS.exp, FLAGS.model)
test_save_path = FLAGS.root_path + "model/{}_{}/{}_predictions/".format(FLAGS.name, FLAGS.exp, FLAGS.model)
loss_logger_path = FLAGS.root_path + 'code/utils/Experiments/'
patch_size = FLAGS.patch_size

num_classes = 2

patch_size = (patch_size, patch_size, patch_size)
FLAGS.root_path = FLAGS.root_path + 'data/'
with open(FLAGS.root_path + '/val.list', 'r') as f:
    image_list = f.readlines()
image_list = [item.replace('\n','') for item in image_list]
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)
print(test_save_path)

def test_calculate_metric():
    
    net = net_factory(net_type=FLAGS.model, in_chns=3, class_num=num_classes, mode="test")
    save_mode_path = os.path.join(snapshot_path, '{}_iter_{}.pth'.format(FLAGS.loss_func, FLAGS.iter_num))
    # save_mode_path = os.path.join(snapshot_path, 'generalCombo_iter_250.pth')
    # save_mode_path = "/home/dayan/Documents/PhD/MSLesion/CoactSeg/pretrained_pth/CoactSeg_reg/vnet/vnet_best_model.pth"
    net.load_state_dict(torch.load(save_mode_path, weights_only=True), strict=False)
    print("init weight from {}".format(save_mode_path))
    net.eval()
    

    avg_metric = test_all_case((80,80,80), loss_logger_path, FLAGS.loss_func, FLAGS.model, 1, net, image_list, num_classes=num_classes,
                    stride_xy=20, stride_z=20,
                    save_result=True, test_save_path=test_save_path,
                    metric_detail=FLAGS.detail, nms=FLAGS.nms, act=FLAGS.act)
    return avg_metric


if __name__ == '__main__':
    metric = test_calculate_metric()
    with open(loss_logger_path + "performance.txt", 'a') as f:
        f.writelines('{}_{}: {}\n'.format(FLAGS.loss_func,FLAGS.act, metric))
    print(metric)
