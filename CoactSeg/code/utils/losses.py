import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt as edt


def identify_axis(shape):
    """
    Enables the loss function to be used flexibly for 2D or 3D images
    """
    # Three dimensional
    if len(shape) == 5:
        return [2, 3, 4]

    # Two dimensional
    elif len(shape) == 4:
        return [2, 3]

    # Exception - Unknown
    else:
        raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')

def Binary_dice_loss(predictive, target, ep=1e-8):
    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    loss = 1 - intersection / union
    return loss

def kl_loss(inputs, targets, ep=1e-8):
    kl_loss=nn.KLDivLoss(reduction='mean')
    consist_loss = kl_loss(torch.log(inputs+ep), targets)
    return consist_loss

def soft_ce_loss(inputs, target, ep=1e-8):
    logprobs = torch.log(inputs+ep)
    return  torch.mean(-(target[:,0,...]*logprobs[:,0,...]+target[:,1,...]*logprobs[:,1,...]))

def mse_loss(input1, input2):
    return torch.mean((input1 - input2 + 1e-8)**2)

class DiceLoss(nn.Module):
    "This is dice loss for multi-class segmentation"
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-10
        intersection = torch.sum(score * target)
        union = torch.sum(score * score) + torch.sum(target * target) + smooth
        loss = 1 - intersection / union
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        print(inputs.size())
        print(target.size())
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def focal_loss(predictive, target, ep=1e-8, alpha=0.25, gamma=2., prob_conv="sigmoid"):
    target = target.float()
    ce_loss = F.binary_cross_entropy_with_logits(predictive, target, reduction="none")

    if prob_conv == "sigmoid":
        pred = torch.sigmoid(predictive)
    elif prob_conv == "softmax":
        pred = torch.softmax(predictive)

    prob = pred*target + (1-pred)*(1-target)

    alpha_t = alpha*target + (1-alpha)*(1-target)

    focal_loss = alpha_t*torch.pow(1-prob, gamma)*ce_loss

    return focal_loss.mean()


# Focal - Dice Loss
class FocalDiceLoss:
    def __init__(self, n_classes, bins):
        self.n_classes = n_classes
        self.bins = bins
        self.m = 0

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def loss_calculation(self, predictive, target, ep=1e-8):

        # print("Target: {}".format(predictive.shape))

        # tp = (predictive == target) * (target_one_hot == 1)
        # fn = (predictive != target) * (target_one_hot == 1)
        # fp = (predictive != target) * (target_one_hot == 0)
        # tn = (predictive == target) * (target_one_hot == 0)

        mask = (predictive*target)
        # print("mask: {}".format(mask.shape))
        tp = predictive[(predictive*target) == 1]
        tn = predictive[(1-predictive)*(torch.logical_not(target)) == 1]
        fn = predictive[(1-predictive)*target == 1]
        fp = predictive[predictive*(torch.logical_not(target)) == 1]

        sorted_tn = torch.sort(tn, descending=True, dim=0)

        bin_size = len(tn) // self.bins
        print(len(tn))
        print(bin_size)
        self.m = len(tn)
        sampled_indices = []
        for k in range(self.bins):
            bin_start = k * bin_size
            bin_end = bin_start + bin_size
            bin_indices = sorted_tn[bin_start:bin_end]

            sample_num = self.m // self.bins
            sampled_indices.extend(bin_indices[:sample_num])


        a = torch.cat((tp, fn),0)
        b = torch.cat((fp, torch.tensor(sampled_indices).to('cuda')), dim=0)

        numerator = 2 * torch.sum(predictive * target) + ep
        # print("numerator: {}".format(numerator))
        denominator = torch.sum(torch.square(torch.cat((a, b), dim=0))) + torch.sum(torch.square(target)) + ep
        # print("denominator: {}".format(denominator))

        print("loss: {}".format(1-(numerator/denominator)))
        return 1 - (numerator/denominator)


class FocalLoss(nn.Module):
    """
    Parameters
    ----------
    delta: float, optional
        controls weight given to false positives and false negatives, by default: 0.7
    gamma: float, optional
        Focal Tversky loss' focal parameter, controls degree of down-weighting of easy examples, by default: 2.0
    epsilon: float, optional
        clip values to prevent division by zero error
    """

    def __init__(self, delta:float=0.7, gamma:float=2.0, epsilon:float=1e-07):
        super(FocalLoss, self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon
    
    def forward(self, predictive, target, type="symmetric"):
        """
        Parameter
        -----------
        type: string, required
            what type of Focal variant to apply, by default: symmetric
        """
        if type == "symmetric":
            return self.symmetric_focal(predictive, target)
        elif type == "asymmetric":
            return self.asymmetric_focal(predictive, target)
    
    def symmetric_focal(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        cross_entropy = ~y_true * torch.log(y_pred)

        # Calculate losses separately for each class
        back_ce = torch.pow(1 - y_pred[:, 0, :, :], self.gamma) * cross_entropy[:, 0, :, :]
        back_ce = (1 - self.delta) * back_ce

        fore_ce = torch.pow(1 - y_pred[:, 1, :, :], self.gamma) * cross_entropy[:, 1, :, :]
        fore_ce = self.delta * fore_ce

        loss = torch.mean(torch.sum(torch.stack([back_ce, fore_ce], axis=-1), axis=-1))

        return loss
    
    def asymmetric_focal(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        cross_entropy = ~y_true * torch.log(y_pred)

        # Calculate losses separately for each class, only suppressing background class
        back_ce = torch.pow(1 - y_pred[:, :, :, 0], self.gamma) * cross_entropy[:, :, :, 0]
        back_ce = (1 - self.delta) * back_ce

        fore_ce = cross_entropy[:, :, :, 0]
        fore_ce = self.delta * fore_ce

        loss = torch.mean(torch.sum(torch.stack([back_ce, fore_ce], axis=-1), axis=-1))

        return loss


class FocalTversky(nn.Module):
    """
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
    smooth : float, optional
        smooithing constant to prevent division by 0 errors, by default 0.000001
    epsilon : float, optional
        clip values to prevent division by zero error
    """
    def __init__(self, delta:float=0.7, gamma:float=0.75, smooth:float=0.000001, epsilon:float=1e-07):
        super(FocalTversky, self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.smooth = smooth
        self.epsilon = epsilon
    
    def forward(self, predictive, target, type:str="symmetric"):
        """
        Parameters
        ----------
        type: string, required
            what type of focal tversky variant to apply
        """
        if type == "symmetric":
            return self.symmetric_focal_tversky(predictive, target)
        elif type == "asymmetric":
            return self.asymmetric_focal_tversky(predictive, target)

    def symmetric_focal_tversky(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        axis = identify_axis(y_true.size())

        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
        tp = torch.sum(y_true * y_pred, axis=axis)
        fn = torch.sum(y_true * (1 - y_pred), axis=axis)
        fp = torch.sum((~y_true) * y_pred, axis=axis)
        dice_class = (tp + self.epsilon) / (tp + self.delta * fn + (1 - self.delta) * fp + self.epsilon)

        # Calculate losses separately for each class, enhancing both classes
        back_dice = (1 - dice_class[:, 0]) * torch.pow(1 - dice_class[:, 0], -self.gamma)
        fore_dice = (1 - dice_class[:, 1]) * torch.pow(1 - dice_class[:, 1], -self.gamma)

        # Average class scores
        loss = torch.mean(torch.stack([back_dice, fore_dice], axis=-1))
        return loss
    
    def asymmetric_focal_tversky(self, y_pred, y_true):
        # Clip values to prevent division by zero error
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        axis = identify_axis(y_true.size())

        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
        tp = torch.sum(y_true * y_pred, axis=axis)
        fn = torch.sum(y_true * (1-y_pred), axis=axis)
        fp = torch.sum((~y_true) * y_pred, axis=axis)
        dice_class = (tp + self.epsilon)/(tp + self.delta*fn + (1-self.delta)*fp + self.epsilon)

        # Calculate losses separately for each class, only enhancing foreground class
        back_dice = (1-dice_class[:,0])
        fore_dice = (1-dice_class[:,1]) * torch.pow(1-dice_class[:,1], -self.gamma)

        # Average class scores
        loss = torch.mean(torch.stack([back_dice,fore_dice], axis=-1))
        # print("\nfocal_tversky:{}".format(loss))
        return loss


class UnifiedFocal(nn.Module):
    """
    Parameters
    ----------
    weight: float, optional
        represents lambda parameter and controls weight given to symmetric Focal Tversky loss and symmetric Focal loss, by default: 0.5
    delta: float, optional
        controls weight given to each class by default: 0.6
    gamma: float, optional
        focal parameter, controls the degree of background suppression and foreground enhancement, by default: 0.5
    """

    def __init__(self, weight=0.5, delta=0.6, gamma=0.5):
        super(UnifiedFocal, self).__init__()
        self.weight = weight
        self.delta = delta
        self.gamma = gamma
    
    def forward(self, predictive, target, type:str="symmetric"):
        """
        Parameters
        ----------
        type: string, required
            what type of Unified focal variant to apply
        """
        if type == "symmetric":
            return self.symmetric(predictive, target)
        elif type == "asymmetric":
            return self.asymmetric(predictive, target)

    def symmetric(self, y_pred, y_true):
        symmetric_ftl = FocalTversky(delta=self.delta, gamma=self.gamma).forward(y_pred, y_true, type="symmetric")
        symmetric_fl = FocalLoss(delta=self.delta, gamma=self.gamma).forward(y_pred, y_true, type="symmetric")
        if self.weight is not None:
            return (self.weight * symmetric_ftl) + ((1-self.weight)*symmetric_fl)
        else:
            return symmetric_fl + symmetric_ftl
    
    def asymmetric(self, y_pred, y_true):
        asymmetric_ftl = FocalTversky(delta=self.delta, gamma=self.gamma).forward(y_pred, y_true, type="asymmetric")
        asymmetric_fl = FocalLoss(delta=self.delta, gamma=self.gamma).forward(y_pred, y_true, type="asymmetric")
        if self.weight is not None:
            print("asymUniFoc:{}".format((self.weight * asymmetric_ftl) + ((1-self.weight)*asymmetric_fl)))
            return (self.weight * asymmetric_ftl) + ((1-self.weight)*asymmetric_fl)
        else:
            return asymmetric_fl + asymmetric_ftl



# referred from https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions/blob/master/loss_functions.py
class Tversky(nn.Module):
    def __init__(self, type:str = "tversky"):
        super(Tversky, self).__init__()
        self.type = type
    
    def tverskyIndex(self, predictive, target, alpha=0.7):
        smooth = 1
        pred_flat = torch.flatten(predictive)
        tar_flat = torch.flatten(target)

        true_positive = torch.sum(predictive * target)
        false_positive = torch.sum((1-target) *  predictive)
        false_negative = torch.sum((1-predictive)*target)

        return true_positive/(true_positive + (alpha * false_positive) + ((1-alpha)*false_negative) + smooth)
    
    def tverskyLoss(self, predictive, target, alpha=0.7):
        return 1 - self.tverskyIndex(predictive, target, alpha)
    
    def focalTverskyLoss(self, predictive, target, alpha=0.7, gamma=0.75):
        return torch.pow(self.tverskyLoss(predictive, target, alpha), gamma)
    
    def forward(self, predictive, target, alpha=0.7, gamma=0.75):
        if self.type == "tversky":
            return self.tverskyLoss(predictive, target, alpha)
        elif self.type == "focalTversky":
            return self.focalTverskyLoss(predictive, target, alpha, gamma)

class DicePP(nn.Module):
    def __init__(self, gamma=2):
        super(DicePP, self).__init__()
        self.gamma = gamma
    
    def forward(self, predictive, target):

        epsilon = 1e-07
        
        true_positives = torch.sum(predictive * target)
        false_positives = torch.sum(torch.pow((predictive*(1-target)), self.gamma))
        false_negatives = torch.sum(torch.pow(((1-target)*predictive), self.gamma))

        numerator = 2 * true_positives + epsilon
        denominator = (2 * true_positives) + false_negatives + false_positives

        return 1 - (numerator/denominator)


# tversky index
def tversky_index(y_true, y_pred, beta:int = 0.7):
        smooth = 1
        y_true_pos = torch.flatten(y_true)
        y_pred_pos = torch.flatten(y_pred)
        true_pos = torch.sum(y_true_pos * y_pred_pos)
        false_neg = torch.sum(y_true_pos * (1 - y_pred_pos))
        false_pos = torch.sum((1 - y_true_pos) * y_pred_pos)
        return (true_pos + smooth) / (true_pos + ((1-beta) * false_neg) + ((
                beta) * false_pos + smooth))

import numpy as np
from scipy.ndimage import label

def spatial_false_positives(gt, pred):
    """
    Computes the spatially weighted false positives (FP_S) between ground truth and prediction.
    
    Args:
        gt: Ground truth binary mask (ndarray, 0 and 1).
        pred: Predicted binary mask (ndarray, 0 and 1).

    Returns:
        FP_S: Spatial false positive score (float)
    """
    assert gt.shape == pred.shape, "Shape mismatch"
    gt = gt.cpu().detach().numpy()
    pred = pred.cpu().detach().numpy()

    # 1. Label connected components in ground truth
    labeled_gt, num_objects = label(gt)
    
    # 2. Label connected components in prediction
    labeled_pred, _ = label(pred)

    fp_s = 0.0
    for i in range(1, num_objects + 1):
        obj_mask = labeled_gt == i
        obj_size = np.sum(obj_mask)

        if obj_size == 0:
            continue  # avoid division by zero

        # False Positives: predicted as object but not in ground truth
        fp_mask = np.logical_and(pred == 1, gt == 0)

        # Count false positives near/within the object (can use dilated mask if spatial margin matters)
        fp_in_region = np.sum(fp_mask[obj_mask])

        # Accumulate weighted FP
        fp_s += fp_in_region / obj_size

    return fp_s

def generalisedJc(y_true, y_pred):
    smooth = 1
    y_true_pos = torch.flatten(y_true)
    y_pred_pos = torch.flatten(y_pred)

    true_pos = torch.sum(y_true_pos * y_pred_pos)
    false_negative = torch.sum(y_true_pos * (1 - y_pred_pos))
    false_positve_s = spatial_false_positives(y_true, y_pred)

    return abs(true_pos)/(abs(true_pos) + abs(false_negative) + abs(false_positve_s))

# tversky loss
def tversky_loss(y_true, y_pred):
        return 1 - tversky_index(y_true, y_pred)

# focal tversky
def focal_tversky(y_true, y_pred):
        pt_1 = tversky_index(y_true, y_pred)
        gamma = 0.75
        return torch.pow((1 - pt_1), gamma)


class LogCoshLoss(nn.Module):
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, y_pred, y_true):
        diff = y_true - y_pred
        return torch.mean(torch.log(torch.cosh(diff + 1e-12)))


class HausdorffDTLoss(nn.Module):
    """Binary Hausdorff loss based on distance transform"""

    def __init__(self, alpha=2.0, **kwargs):
        super(HausdorffDTLoss, self).__init__()
        self.alpha = alpha

    @torch.no_grad()
    def distance_field(self, img: np.ndarray) -> np.ndarray:
        field = np.zeros_like(img)

        for batch in range(len(img)):
            fg_mask = img[batch] > 0.5

            if fg_mask.any():
                bg_mask = ~fg_mask

                fg_dist = edt(fg_mask)
                bg_dist = edt(bg_mask)

                field[batch] = fg_dist + bg_dist

        return field

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, debug=False
    ) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, 1, x, y, z) or (b, 1, x, y)
        target: (b, 1, x, y, z) or (b, 1, x, y)
        """
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert (
            pred.dim() == target.dim()
        ), "Prediction and target need to be of same dimension"

        # pred = torch.sigmoid(pred)

        pred_dt = torch.from_numpy(self.distance_field(pred.cpu().detach().numpy())).float()
        target_dt = torch.from_numpy(self.distance_field(target.cpu().detach().numpy())).float()

        pred_error = (pred - target) ** 2
        distance = pred_dt ** self.alpha + target_dt ** self.alpha

        dt_field = pred_error * distance.cuda()
        loss = dt_field.mean()

        if debug:
            return (
                loss.numpy(),
                (
                    dt_field.numpy()[0, 0],
                    pred_error.numpy()[0, 0],
                    distance.numpy()[0, 0],
                    pred_dt.numpy()[0, 0],
                    target_dt.numpy()[0, 0],
                ),
            )

        else:
            return loss


def Combo_loss(y_true, y_pred):
    # borrowed from : https://github.com/asgsaeid/ComboLoss/blob/master/combo_loss.py
    e = 1e-12
    smooth = 1
    ce_w = 0.5
    ce_d_w = 0.5
    clip_val = 0.001

    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    d = (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)
    # y_pred_f = torch.clip(y_pred_f, e, 1.0 - e)
    y_pred_f = torch.clip(y_pred_f, clip_val, 1.0 - clip_val)
    y_true_f = torch.clip(y_true_f, clip_val, 1.0 - clip_val)
    out = - (ce_w * y_true_f * torch.log(y_pred_f)) + ((1 - ce_w) * (1 - y_true_f) * torch.log(1 - y_pred_f))
    weighted_ce = torch.mean(out, axis=-1)
    combo = (ce_d_w * weighted_ce) - ((1 - ce_d_w) * d)
    print(weighted_ce)
    print(d)
    return combo


def hyTver(y_true, y_pred):
    "slighly modified version of combo loss"

    e = 1e-12
    smooth = 1
    alpha = 0.7
    beta = 0.3
    gamma = 0.5
    clip_val = 0.001
    tv = tversky_index(y_true, y_pred, beta=beta)

    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    # y_pred_f = torch.clip(y_pred_f, e, 1.0 - e)
    y_pred_f = torch.clip(y_pred_f, clip_val, 1.0 - clip_val)
    y_true_f = torch.clip(y_true_f, clip_val, 1.0 - clip_val)


    out = - (alpha * y_true_f * torch.log(y_pred_f)) + ((1 - alpha) * (1 - y_true_f) * torch.log(1 - y_pred_f))
    weighted_ce = torch.mean(out, axis=-1)
    combo = (gamma * weighted_ce) - ((1 - gamma) * tv)

    return combo




class CrossentropyND(torch.nn.CrossEntropyLoss):
    """
    Network has to have NO NONLINEARITY!
    """
    def forward(self, inp, target):
        target = target.long()
        num_classes = inp.size()[1]

        i0 = 1
        i1 = 2

        while i1 < len(inp.shape): # this is ugly but torch only allows to transpose two axes at once
            inp = inp.transpose(i0, i1)
            i0 += 1
            i1 += 1

        inp = inp.contiguous()
        inp = inp.view(-1, num_classes)

        target = target.reshape(-1,)

        return super(CrossentropyND, self).forward(inp, target)
    

class WeightedCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    """
    Network has to have NO NONLINEARITY!
    """
    def __init__(self, weight=None):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weight = weight

    def forward(self, inp, target):
        target = target.long()
        num_classes = inp.size()[1]

        i0 = 1
        i1 = 2

        while i1 < len(inp.shape): # this is ugly but torch only allows to transpose two axes at once
            inp = inp.transpose(i0, i1)
            i0 += 1
            i1 += 1

        inp = inp.contiguous()
        inp = inp.view(-1, num_classes)

        target = target.view(-1,)
        wce_loss = torch.nn.CrossEntropyLoss(weight=self.weight)

        return wce_loss(inp, target)
    



