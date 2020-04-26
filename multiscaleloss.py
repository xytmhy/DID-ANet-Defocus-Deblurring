import torch
import torch.nn.functional as F


def realEPE(output, target):
    return EPE(output, target, mean=True)

def EPE(output, target, mean=True):

    EPE_map1 = torch.norm(target-output, 2, 1)
    batch_size = EPE_map1.size(0)
    if mean:
        return EPE_map1.mean()
    else:
        return EPE_map1.sum()/batch_size

def mapEPE(target_deblurring, target_defocusmap, output_defocusmap, output_deblurring, mean=True):

    EPE_map0 = torch.norm(target_deblurring - output_deblurring, 2, 1)
    EPE_map1 = target_defocusmap/target_defocusmap.mean() * torch.norm(target_deblurring-output_deblurring, 2, 1)
    EPE_map2 = torch.norm(target_defocusmap - output_defocusmap, 2, 1)

    batch_size = EPE_map1.size(0)
    if mean:
        return 1 * EPE_map0.mean() + 9 * EPE_map1.mean() + 2 * EPE_map2.mean()
    else:
        return (1 * EPE_map0.sum() + 9 * EPE_map1.sum() + 2 * EPE_map2.sum())/batch_size

def x1EPE(target_deblurring, target_defocusmap, output_defocusmap, output_deblurring, deblur_defocusmap, mean=True):

    # loss between deburring result and deblrring result
    EPE_map0 = torch.norm(target_deblurring - output_deblurring, 2, 1)
    # weighted loss between deburring result and deblrring result
    EPE_map1 = target_defocusmap/target_defocusmap.mean() * torch.norm(target_deblurring-output_deblurring, 2, 1)
    # loss between defocus estimation and defocus ground truth map
    EPE_map2 = torch.norm(target_defocusmap - output_defocusmap, 2, 1)
    # the defocus estimation of the deblurring result
    EPE_map3 = torch.norm(deblur_defocusmap, 2, 1)

    batch_size = EPE_map1.size(0)

    if mean:
        return 1 * EPE_map0.mean() + 10 * EPE_map1.mean() + 0 * EPE_map2.mean() + 1 * EPE_map3.mean()
    else:
        return (1 * EPE_map0.sum() + 10 * EPE_map1.sum() + 0 * EPE_map2.sum() + 1 * EPE_map3.sum())/batch_size

def x2EPE(target_deblurring, target_defocusmap, output_defocusmap, output_deblurring, deblur_defocusmap, gt_defocusmap, mean=True):

    # loss between deburring result and deblrring result
    EPE_map0 = torch.norm(target_deblurring - output_deblurring, 2, 1)
    # weighted loss between deburring result and deblrring result
    EPE_map1 = target_defocusmap/target_defocusmap.mean() * torch.norm(target_deblurring-output_deblurring, 2, 1)
    # loss between defocus estimation and defocus ground truth map
    EPE_map2 = torch.norm(target_defocusmap - output_defocusmap, 2, 1)
    # the defocus estimation of the deblurring result
    EPE_map3 = torch.norm(deblur_defocusmap - gt_defocusmap, 2, 1)

    batch_size = EPE_map1.size(0)

    if mean:
        return 1 * EPE_map0.mean() + 10 * EPE_map1.mean() + 0 * EPE_map2.mean() + 5 * EPE_map3.mean()
    else:
        return (1 * EPE_map0.sum() + 10 * EPE_map1.sum() + 0 * EPE_map2.sum() + 5 * EPE_map3.sum())/batch_size
