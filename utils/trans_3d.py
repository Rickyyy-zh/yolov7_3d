import torch
import numpy as np
import cv2
from visualize.show_2d3d_box import *

EPS = 1e-12
PI = 3.14159

def trans3dPoint2Pixel(t_3d, p2s, w2cs):
    assert w2cs.shape[0] == p2s.shape[0], 'wrong w2c numbers with batch'

    loc_3D_boxcenter = t_3d[:, 5:8]
    loc_3D_boxcenter[:, 1] = loc_3D_boxcenter[:, 1] - t_3d[:, 2] * 0.5
    loc_3D_boxcenter = loc_3D_boxcenter.to('cpu').numpy()
    r_y = t_3d[:, -1]
    alpha = t_3d[:, 1:2]
    center3d = torch.tensor([],device= t_3d.device)
    depth_t = torch.tensor([],device= t_3d.device)

    for i in range(p2s.shape[0]):
        # p2s[i] = p2s[i].to(t_3d.device)
        mask = (t_3d[:, 0] == i).to('cpu').numpy()
        center_pixel = p2s[i, :3, :3] @ (loc_3D_boxcenter[mask].T)
        # center_pixel_numpy = p2s[i, :3, :3].to('cpu').numpy() @ np.matrix(loc_3D_boxcenter[mask].to('cpu').numpy().T)
        depth = center_pixel[2, :]
        center_pixel = center_pixel / depth
        center_pixel = center_pixel.T[:, :2] #
        depth = torch.from_numpy(depth.T).to(t_3d.device)
        
        center3d = torch.cat((center3d, torch.from_numpy(center_pixel).to(t_3d.device)), 0)
        depth_t = torch.cat((depth_t, depth), 0)
    
    alpha_encode_1 = torch.sin(alpha)
    alpha_encode_2 = torch.cos(alpha)
    alpha_t = torch.cat((alpha_encode_1, alpha_encode_2), 1)
    return center3d, depth_t, alpha_t, t_3d[:, 2:5]
    
def decodeDepthOut(depth):
    return (1. / (torch.sigmoid(depth) + EPS)) - 1

def transAlpha2Yaw(alpha_vec, location):
    alphas = torch.atan(alpha_vec[:, 0] / (alpha_vec[:, 1] + EPS))
    theta = torch.atan(location[:, 0] / (location[:, 2] + EPS))
    
    cos_pos_idx = (alpha_vec[:, 1] >= 0).nonzero()
    cos_neg_idx = (alpha_vec[:, 1] < 0).nonzero()

    # alphas[cos_pos_idx] -= PI / 2
    # alphas[cos_neg_idx] += PI / 2

    ry = alphas + theta

    return alphas.unsqueeze(1), ry.unsqueeze(1)

def decodeLoc(center3D, depth, p2):
    depth = depth.unsqueeze(1)
    norm_points = torch.multiply(center3D, depth)
    points = torch.cat((norm_points, depth), 1)
    
    p2_inv = torch.from_numpy(np.linalg.inv(p2)).float().to(center3D.device)
    
    loc = points @ p2_inv[:3, :3].T

    return loc
    
def decodePred(oriOutput, p2, w2c, shape):
    out_2d = oriOutput[:, :4]
    dim = oriOutput[:, 10:13]
    conf_cls = oriOutput[:, 13:15]

    center3D = oriOutput[:, 4:6] # bottom center project in pixel
    depth = oriOutput[:, 6]
    loc = decodeLoc(center3D, decodeDepthOut(depth), p2)
    loc[:, 1] = loc[:, 1] + dim[:, 0] * 0.5

    alpha_vec = oriOutput[:, 8:10]
    alpha, ry = transAlpha2Yaw(alpha_vec, loc)
    
    return torch.cat((out_2d, dim, loc, ry, conf_cls), 1), alpha

