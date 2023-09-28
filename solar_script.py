#!/usr/bin/env python3

#---U-NET INFERENCE MODULE FOR FAULT SOLAR PV PANELS SEGMENTATION---
# Images: 640x480 RGB (IR camera) --> resuts resized to 448x336 px
# Python: 3.8.5, torch: 1.7.1, np: 1.19.2, cv2: 4.5.3, ns: 7.1.1, json: 2.0.9

import torch
from torchvision import transforms as T
from torch.nn import functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, sampler
from torch import from_numpy, tensor
import numpy as np
import cv2
import os
import random
import natsort as ns
import json
import configparser

# to reproduce same results fixing the seed and hash
seed = 42
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# CPU/GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
Simple (plain-vanilla) UNet demo. @author: ptrblck
https://github.com/ptrblck/pytorch_misc/blob/master/unet_demo.py
"""
class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding,
                 stride):
        super(BaseConv, self).__init__()
        self.act = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding, stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size,padding, stride)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return x

class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(DownConv, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv_block = BaseConv(in_channels, out_channels, kernel_size, padding, stride)

    def forward(self, x):
        x = self.pool1(x)
        x = self.conv_block(x)
        return x

class UpConv(nn.Module):
    def __init__(self, in_channels, in_channels_skip, out_channels, kernel_size, padding, stride):
        super(UpConv, self).__init__()

        self.conv_trans1 = nn.ConvTranspose2d(
        in_channels, in_channels, kernel_size=2, padding=0, stride=2)
        
        self.conv_block = BaseConv(
            in_channels=in_channels + in_channels_skip,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride)

    def forward(self, x, x_skip):
        x = self.conv_trans1(x)
        x = torch.cat((x, x_skip), dim=1)
        x = self.conv_block(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, n_class, kernel_size, padding, stride):
        super(UNet, self).__init__()

        self.init_conv = BaseConv(in_channels, out_channels, kernel_size, padding, stride)
        self.down1 = DownConv(out_channels, 2 * out_channels, kernel_size, padding, stride)
        self.down2 = DownConv(2 * out_channels, 4 * out_channels, kernel_size, padding, stride)
        self.down3 = DownConv(4 * out_channels, 8 * out_channels, kernel_size, padding, stride)
        self.up3 = UpConv(8 * out_channels, 4 * out_channels, 4 * out_channels, kernel_size, padding, stride)
        self.up2 = UpConv(4 * out_channels, 2 * out_channels, 2 * out_channels, kernel_size, padding, stride)
        self.up1 = UpConv(2 * out_channels, out_channels, out_channels, kernel_size, padding, stride)
        self.out = nn.Conv2d(out_channels, n_class, kernel_size, padding, stride)

    def forward(self, x):
        # Encoder
        x = self.init_conv(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        # Decoder
        x_up = self.up3(x3, x2)
        x_up = self.up2(x_up, x1)
        x_up = self.up1(x_up, x)
        x_out = F.log_softmax(self.out(x_up), 1)
        return x_out
        
# Dataset class  
class SolarSetInference(Dataset):
   
    def __init__(self, pth):
        
        self.x_data = os.getcwd() + pth
        self.img_names = ns.natsorted(list(os.listdir(os.getcwd() + pth)))
        self.len = len(self.img_names)
        self.transforms = T.Compose([T.ToTensor()])
            
    def __getitem__(self, idx):
        
        img_name = self.img_names[idx]
        img_path = os.path.join(self.x_data, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (448, 336), interpolation=cv2.INTER_NEAREST)
        img = self.transforms(img)
             
        return img
  
    def __len__(self):
        
        return self.len

# Get parameters     
def get_params(cfg):

    """
    cfg.txt made for reading with Configparser lib:
    https://docs.python.org/3/library/configparser.html
    Parameters could be configured in cfg.txt:

    ['paths']
    img = 'img'            # folder with analyzed images
    mod = 'model.pth'      # filename of model state dict
    view = 'view'          # folder for viewing masks storage
    out = 'data'           # folder for out json data

    ['UNet']
    in_channels = 3        # number of channels in analyzed images (3 for RGB or HSV, 1 for grayscale)
    n_class = 7            # number of classes of objects need to detect by NN (bckgr + 6 PV failure types here)
    """

    config = configparser.ConfigParser()
    config.read(cfg)

    if config:
        img_dir = '/' + config['paths']['img']
        view_dir = '/' + config['paths']['view']
        mod_name = config['paths']['mod']
        out_name = config['paths']['out']
        in_channels=int(config['UNet']['in_channels'])
        n_class=int(config['UNet']['n_class'])
    else:
        raise FileNotFoundError ('Config file (cfg.txt) not found!')
    
    return {'img_dir': img_dir, 'view_dir': view_dir, 'mod_name': mod_name, 
            'out_name': out_name, 'in_channels': in_channels, 'n_class': n_class}
    
# Converting predicted masks to images
def converter(pred):
    
    '''
    Fn to convert predicted torch tensors to img arrays (numpy uint8 arrays).
    RGB-layout channel are constructed from pred tensor regions as follows:
    
    backgr == '0' == (0,0,0) --> black
    'single_cell' == '1' == (255,0,255) --> purple
    'hotspot' == '2' == (0,0,255) --> red
    'string' == '3' == (255,255,0) --> cyan
    'multi_cell' == '4' == (255,0,0) --> blue
    'vegetation' == '5' == (0,255,0) --> green
    'full_module' == '6' == (0,255,255) --> yellow
    
    Also 6 class label-associated binary masks are one-hot encoded and packed in 
    list for each img, these lists are wrapped in dict to bind file names as keys
    
    Returns: np.uint8, dict
    '''
    
    single_ch = pred
    
    single_cell = single_ch == 1
    hotspot = single_ch == 2
    string = single_ch == 3
    multi_cell = single_ch == 4
    vegetation = single_ch == 5
    full_module = single_ch == 6
    
    r = single_cell + string + multi_cell
    g = string + vegetation + full_module
    b = single_cell + hotspot + full_module
    
    image = torch.stack([b,g,r], axis=-1).int()*255
    
    data = [np.uint8(single_cell.numpy()*255), np.uint8(hotspot.numpy()*255), 
            np.uint8(string.numpy()*255), np.uint8(multi_cell.numpy()*255), 
            np.uint8(vegetation.numpy()*255), np.uint8(full_module.numpy()*255)]
    
    return np.uint8(image.numpy()), data
    
# Contours filtering
def cnt_filter (cnts, cutoff):
    
    '''
    Fn recieves cutoff param and list of counters finded by cv2.findContours, 
    filter (throws out noise contours with small perimeter lengths, i.e. 
    cv2.arcLength values) from the list. Small value is which < cutoff
    '''
    
    new_cnts = [i for i in cnts if cv2.arcLength(i ,True) >= cutoff]
    return new_cnts

# Exctracting statistics from predicted masks    
def img_processing(class_dict):
    
    '''
    Fn to extract data (class labels, coordinates (x, y)) from predicted OHE 
    (one-hot encoding) class-masks, which is the list of 6 class-associated binary 
    masks for each img, wrapped in 'class_dict' (bind to f_names). Also draws 
    centroids with (x, y) coordinates on black canvas (for debugging purposes).
    1. Draw 1 px black lines on the edge of mask to complete edge-most contours
    2. Apply built-in Open CV Canny edge detection algorithm
    3. Build contours on detcted edges with built-in Open CV cv2.findContours
    4. Filter contours by size (custom 'cnt_filter' fn above) to reject noise
    5. Find centroids of the filtered contours by built-in Open CV cv2.moments()
    6. Write aquired data to 'target_dict', which wrapped back to list for each img
    7. Fill 'global_dict' by these lists, binded to file names as dict keys
    '''

    label_list = ['single_cell', 'hotspot', 'string', 
              'multi_cell', 'vegetation', 'full_module']

    global_dict = {}

    for k, v in class_dict.items():

        local_list = []
        for i, l in zip(v, label_list):

            target_dict = {}

            # add 1 px blacklines on the edge of mask to complete edge-most contours
            i_black_edge = cv2.rectangle(i, (0, 0), (i.shape[1]-1, i.shape[0]-1), (0, 0, 0), 1)

            # Canny out: matrix-array, could be saved as pic
            edges = cv2.Canny(i_black_edge, 20, 1000, False)

            # findContours out: nested list-like array
            cnts, _ = cv2.findContours(edges, cv2. RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # custom contours arc_length filter, out: filtered list-like array
            cnts_f = cnt_filter(cnts, 60)
            
            # convex hull approximation to close some not closed contours
            cnts_f_h = [cv2.convexHull(i) for i in cnts_f]
            
            # Centroids calculation
            for i, c in enumerate(cnts_f_h):
                M = cv2.moments(c)
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                target_dict[str(i)] = (cx, cy)

            if target_dict:
                local_list.append({l: target_dict}) 

        global_dict.update({k: local_list})
    
    return global_dict

# Main loop for inference
if __name__ == '__main__':

    # ~1 img/min @Intel P6000, 4Gb
    # ~10 imgs/min @Intel i5, 8Gb
    # ~10 imgs/sec NVIDIA RTX 2080Ti

    # call params getter fn
    par = get_params('cfg.txt')

    # create view dir (for resulting masks) if needed
    if not os.path.exists(os.getcwd() + par['view_dir']):
        os.mkdir(os.getcwd() + par['view_dir'])

    # data Loader
    if os.listdir(os.getcwd() + par['img_dir']):
        loader = DataLoader(dataset=SolarSetInference(par['img_dir']),
                              batch_size=1, shuffle=False, num_workers=2)
    else:
        raise FileNotFoundError (f'Images in image directory not found!')
        
    # initialize U-Net model with params acquired
    model = UNet(in_channels=par['in_channels'], out_channels=64,
                 n_class=par['n_class'], kernel_size=3, padding=1, stride=1)

    # model state dict loading
    if par['mod_name'] + '.pth' in os.listdir(os.getcwd()):
        model.load_state_dict(torch.load(os.path.join(os.getcwd(),
                              par['mod_name'] + '.pth'), map_location=device))
    else:
        raise FileNotFoundError (f'Model state dictionary not found!')

    # instance for naming when writing
    a = SolarSetInference(par['img_dir'])

    # evaluation mode for inference
    model.eval()

    class_dict = {}

    with torch.no_grad():
        for i, img in enumerate(loader, 0):

            # generate masks with black bgr
            model, img = model.to(device), img.to(device)
            output = model.forward(img)
            _, predicted = (torch.max(output.detach().cpu(), 1))
            predicted = torch.squeeze(predicted)
            converted, data = converter(predicted) # converter fn above
            converted = cv2.cvtColor(converted, cv2.COLOR_BGR2RGB)

            if converted.any(): # if non-empty mask

                # generate 'for humans' masks: b&w images with color labels
                img = torch.squeeze(img.detach().cpu()*255).permute(1,2,0)
                img = np.uint8(img.numpy())
                img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                converted_bw = cv2.cvtColor(converted, cv2.COLOR_BGR2GRAY)
                _, converted_bw_thr = cv2.threshold(converted_bw, 10, 255,
                                                    cv2.THRESH_BINARY_INV)
                bw_holes = cv2.bitwise_and(img_bw, converted_bw_thr)
                bw_holes_3 = cv2.merge([bw_holes, bw_holes, bw_holes])
                masked = cv2.bitwise_or(bw_holes_3, converted)

                cv2.imwrite(os.getcwd() + '/' + par['view_dir'][1:] + '/m_' + a.img_names[i], masked)

                # write class-associated binary masks to list wrapped in dict (bind to file names)
                class_dict.update({a.img_names[i]: data})

    result = img_processing(class_dict)

    with open (par['out_name'] + '.json', 'w') as f: # writing final json file
        json.dump(result, f, indent=2)
