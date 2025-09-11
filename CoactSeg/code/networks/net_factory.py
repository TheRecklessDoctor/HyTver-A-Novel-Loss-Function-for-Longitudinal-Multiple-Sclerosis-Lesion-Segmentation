from networks.VNet import VNet
from segmentation_models_pytorch import Unet

def net_factory(net_type="vnet", in_chns=3, class_num=2, mode = "train", weights = None, norm:str="batchnorm", act:str="relu"):
    if net_type == "vnet" and mode == "train":
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True, act=act, has_residual=False)
    elif net_type == "vnet" and mode == "test":
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False, act=act).cuda()
    return net
