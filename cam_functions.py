import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def cam_minmax_norm(cam):
    # import pdb;pdb.set_trace()
    cam = cam - cam.min(dim=0)[0]
    cam_img = cam / cam.max(dim=0)[0]
    
    return cam_img


def make_cam(atts, cnn_feat, label_vec, P):

    atts = atts.view(cnn_feat.size(0),P["num_classes"],cnn_feat.size(2),cnn_feat.size(3)) # (batch, num_classes, 14, 14)
    atts_sp = atts[label_vec.type(torch.bool),:,:] # (batch, num_classes, 14, 14)

    atts_sp = atts_sp.unsqueeze(1) # (batch, 1, num_classes, 14, 14)

    cnn_feat = (cnn_feat*atts_sp).mean(dim=1) # (batch, 14, 14)
    cam = cam_minmax_norm(cnn_feat) # (batch, 14, 14)

    return cam
    
    
def make_resnet_cam_mask(image, cam_w, label_vec, P):

    
    cam_sp = cam_w[label_vec.type(torch.bool),:,:] # (batch, num_classes, 14, 14)
    cam = cam_minmax_norm(cam_sp) # (batch, 14, 14)
    try:
        cam = F.interpolate(cam, size=(image.size(2), image.size(3)), mode='bilinear', align_corners=False)
    except:
        cam = cam.unsqueeze(1) # (batch, 1, 14, 14)
        cam = F.interpolate(cam, size=(image.size(2), image.size(3)), mode='bilinear', align_corners=False) # (batch, 1, 448, 448)
        
    
    cam_mask = torch.where(cam > P["cam_threshold"], 1, 0)
    masked_image = image*cam_mask
    
    return masked_image

def make_nus_resnet_cam_mask(image, cam_w, label_vec, P, device):

    if label_vec.sum() == 0:
        cam_mask_mod = torch.ones(image.size(0),1,image.size(-2),image.size(-1)).to(device)
        masked_image = image*cam_mask_mod
        return masked_image
    cam_sp = cam_w[label_vec.type(torch.bool),:,:] # (batch, num_classes, 14, 14)
    cam = cam_minmax_norm(cam_sp) # (batch, 14, 14)
    try:
        cam = F.interpolate(cam, size=(image.size(2), image.size(3)), mode='bilinear', align_corners=False)
    except:
        cam = cam.unsqueeze(1) # (batch, 1, 14, 14)

        cam = F.interpolate(cam, size=(image.size(2), image.size(3)), mode='bilinear', align_corners=False) # (batch, 1, 448, 448)
    idx = label_vec.sum(dim=1)==1   
    
    cam_mask = torch.where(cam > P["cam_threshold"], 1, 0).float()
    cam_mask_mod = torch.ones(image.size(0),1,image.size(-2),image.size(-1)).to(device)
    
    cam_mask_mod[idx] = cam_mask
    masked_image = image*cam_mask_mod
    
    return masked_image

def make_cam_mask(image, cam, P):
    try:
        cam = F.interpolate(cam, size=(image.size(2), image.size(3)), mode='bilinear', align_corners=False)
    except:
        cam = cam.unsqueeze(1) # (batch, 1, 14, 14)

        cam = F.interpolate(cam, size=(image.size(2), image.size(3)), mode='bilinear', align_corners=False) # (batch, 1, 448, 448)
    
    # cam = cam.squeeze()
    cam_mask = torch.where(cam > P["cam_threshold"], 1, 0)
    masked_image = image*cam_mask
    
    return masked_image

def nus_cam_mask(image,atts,cnn_feat, label_vec, P, device):
    atts = atts.view(cnn_feat.size(0),P["num_classes"],cnn_feat.size(2),cnn_feat.size(3)) # (batch, num_classes, 14, 14)
    atts_sp = atts[label_vec.type(torch.bool),:,:]
    atts_sp = atts_sp.unsqueeze(1)
    idx = label_vec.sum(dim=1)==1
    cnn_feat = cnn_feat[idx]
    cnn_feat = (cnn_feat*atts_sp).mean(dim=1) # (batch, 14, 14)
    cam = cam_minmax_norm(cnn_feat)
    
    try:
        cam = F.interpolate(cam, size=(image.size(2), image.size(3)), mode='bilinear', align_corners=False)
    except:
        cam = cam.unsqueeze(1) # (batch, 1, 14, 14)

        cam = F.interpolate(cam, size=(image.size(2), image.size(3)), mode='bilinear', align_corners=False) # (batch, 1, 448, 448)
    
    cam_mask = torch.where(cam > P["cam_threshold"], 1, 0).float()
    cam_mask_mod = torch.ones(image.size(0),1,image.size(-2),image.size(-1)).to(device)
    
    cam_mask_mod[idx] = cam_mask
    masked_image = image*cam_mask_mod
    
    return masked_image