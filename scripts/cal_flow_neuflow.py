'''
Compute flow
Based on RAFT (https://github.com/princeton-vl/RAFT)
'''
import sys
import argparse
import os
import glob
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from os.path import join
import cv2

raft_path = join(os.path.dirname(__file__), '..', 'external', 'NeuFlow_v2')
if raft_path not in sys.path:
    sys.path.insert(1, raft_path)
from NeuFlow.neuflow import NeuFlow
from NeuFlow.backbone_v7 import ConvBlock

DEVICE = 'cuda'

image_width = 400 #768
image_height = 192 #432

def get_cuda_image(image_path):
    image = cv2.imread(image_path)

    image = cv2.resize(image, (image_width, image_height))

    image = torch.from_numpy(image).permute(2, 0, 1).half()
    return image[None].cuda()

def fuse_conv_and_bn(conv, bn):
    """Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/."""
    fusedconv = (
        torch.nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(conv.weight.device)
    )

    # Prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # Prepare spatial bias
    b_conv = torch.zeros(conv.weight.shape[0], device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')    
    parser.add_argument('--dir_data', type=str, help='dataset directory')
    parser.add_argument('--start_idx', type=int)
    parser.add_argument('--end_idx', type=int)
        
    args = parser.parse_args()
    
    if args.dir_data == None:
        this_dir = os.path.dirname(__file__)
        args.dir_data = join(this_dir, '..', 'data')
        
    if args.model == None:
        this_dir = os.path.dirname(__file__)
        args.model = join(this_dir, '..', 'external', 'RAFT', 'models', 'raft-kitti.pth')
    
    start_idx = args.start_idx
    end_idx = args.end_idx       
    out_dir = join(args.dir_data, 'prepared_data')
       
    device = torch.device('cuda')
    model = NeuFlow.from_pretrained("Study-is-happy/neuflow-v2").to(device)
    for m in model.modules():
        if type(m) is ConvBlock:
            m.conv1 = fuse_conv_and_bn(m.conv1, m.norm1)  # update conv
            m.conv2 = fuse_conv_and_bn(m.conv2, m.norm2)  # update conv
            delattr(m, "norm1")  # remove batchnorm
            delattr(m, "norm2")  # remove batchnorm
            m.forward = m.forward_fuse  # update forward

    model.eval()
    model.half()

    model.init_bhwd(1, image_height, image_width, 'cuda')

    im_list = np.array(np.sort(glob.glob(join(out_dir, '*im_full.jpg'))))
    
    N = len(im_list)
        
    print('Total sample number:', N)
    
    if start_idx == None:
        start_idx = 0
    
    if end_idx == None or end_idx > N - 1 :
        end_idx = N - 1
        
    for sample_idx in tqdm(range(start_idx, end_idx + 1)):
        
        f_im1 = im_list[sample_idx]
        
        im1 = np.array(Image.open(f_im1)).astype(np.uint8)
        
        f_im_next = f_im1[:-4] + '_next.jpg'
        f_im_prev = f_im1[:-4] + '_prev.jpg'

        if os.path.exists(f_im_next):
            im2 = np.array(Image.open(f_im_next)).astype(np.uint8) 
        else:
            im2 = np.array(Image.open(f_im_prev)).astype(np.uint8)
                
        im1 = np.pad(im1, ((2,2),(0,0),(0,0)), 'constant')
        im2 = np.pad(im2, ((2,2),(0,0),(0,0)), 'constant')
                        
        im1 = torch.from_numpy(im1).permute(2, 0, 1).float()
        im1 = im1[None,].to(DEVICE)   
        
        im2 = torch.from_numpy(im2).permute(2, 0, 1).float()
        im2 = im2[None,].to(DEVICE)   
    
        with torch.no_grad():
            flow_low, flow_up = model(im1, im2, iters=20, test_mode=True)
            flow = flow_up[0].permute(1,2,0).cpu().numpy()[2:-2,...]
            
            path_flow = f_im1[:-11] + 'full_flow.npy'        
            np.save(path_flow, flow)
        
    