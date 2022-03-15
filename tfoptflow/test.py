from __future__ import absolute_import, division, print_function
from copy import deepcopy
from skimage.io import imread
from model_pwcnet import ModelPWCNet, _DEFAULT_PWCNET_TEST_OPTIONS
from visualize import flow_to_img
import time
import glob
import os
import cv2

# Build a list of image pairs to process
img_pairs = []
img_backward_pairs = []

file_path = './samples'
filename_list = glob.glob(os.path.join(file_path, '*.png')) + \
                glob.glob(os.path.join(file_path, '*.jpg'))
files = sorted(filename_list)
image1 = imread(files[0])
image2 = None
for i in range(len(files)-1):
    image2 = imread(files[i+1])
    img_pairs.append((image1, image2))
    img_backward_pairs.append((image2, image1))
    image1 = image2


# TODO: Set device to use for inference
# Here, we're using a GPU (use '/device:CPU:0' to run inference on the CPU)
gpu_devices = ['/device:GPU:0']  
controller = '/device:GPU:0'

# TODO: Set the path to the trained model (make sure you've downloaded it first from http://bit.ly/tfoptflow)
ckpt_path = './models//pwcnet-lg-6-2-multisteps-chairsthingsmix/pwcnet.ckpt-595000'

# Configure the model for inference, starting with the default options
nn_opts = deepcopy(_DEFAULT_PWCNET_TEST_OPTIONS)
nn_opts['verbose'] = True
nn_opts['ckpt_path'] = ckpt_path
nn_opts['batch_size'] = 1
nn_opts['gpu_devices'] = gpu_devices
nn_opts['controller'] = controller

# We're running the PWC-Net-large model in quarter-resolution mode
# That is, with a 6 level pyramid, and upsampling of level 2 by 4 in each dimension as the final flow prediction
nn_opts['use_dense_cx'] = True
nn_opts['use_res_cx'] = True
nn_opts['pyr_lvls'] = 6
nn_opts['flow_pred_lvl'] = 2

# The size of the images in this dataset are not multiples of 64, while the model generates flows padded to multiples
# of 64. Hence, we need to crop the predicted flows to their original size
nn_opts['adapt_info'] = (1, 436, 1024, 2)

# Instantiate the model in inference mode and display the model configuration
nn = ModelPWCNet(mode='test', options=nn_opts)
nn.print_config()

# Generate the predictions and display them
# forward
idx = 0
for pair in img_pairs:
    start = time.time()
    pred = nn.predict_from_img_pairs([pair], batch_size=1, verbose=False)
    print(time.time()-start)
    imagename = f'output/fw_{idx:05d}.png'
    print(pred[0].shape)
    cv2.imwrite(imagename, flow_to_img(pred[0]))
    idx += 1
#backword
idx = 0
for pair in img_backward_pairs:
    start = time.time()
    pred = nn.predict_from_img_pairs([pair], batch_size=1, verbose=False)
    print(time.time()-start)    
    imagename = f'output/bw_{idx:05d}.png'
    cv2.imwrite(imagename, flow_to_img(pred[0]))
    idx += 1
