import argparse
import cv2
import os
import time
import glob
import numpy as np

from openvino.inference_engine import IECore
from visualize import flow_to_img

import sample_utils

def load_to_IE(model_xml, pair):
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    ie = IECore()
    net = ie.read_network(model=model_xml, weights=model_bin)
    
    x = np.array([pair])
    # padding for 64 alignment
    print('frame.shape:', x.shape)
    x_adapt, _ = sample_utils.adapt_x(x)
    x_adapt = x_adapt.transpose((0, 3, 1, 2))    # B2HWC --> BC2HW
    print('adapt.shape:', x_adapt.shape)
    print(f"Input shape: {net.input_info['x_tnsr'].tensor_desc.dims}")

    # Call reshape
    net.reshape({'x_tnsr': x_adapt.shape})
    print(f"Input shape (new): {net.input_info['x_tnsr'].tensor_desc.dims}")

    exec_net = ie.load_network(network=net, device_name="CPU")
    print("IR successfully loaded into Inference Engine.")
    del net

    '''
    for item in exec_net.input_info:
        print('input:', item)
    for key in list(exec_net.outputs.keys()):
        print('output:', key)
    '''

    return exec_net

def load_images(args):
    # Get input image
    # Build a list of image pairs to process
    img_pairs = []
    file_path = args.input
    filename_list = glob.glob(os.path.join(file_path, '*.png')) + \
                    glob.glob(os.path.join(file_path, '*.jpg'))
    files = sorted(filename_list)
    image1 = cv2.imread(files[0])
    image2 = None
    for i in range(len(files)-1):
        image2 = cv2.imread(files[i+1])
        img_pairs.append((image1, image2))
        image1 = image2
    return img_pairs


def inference(args):
    '''
    Performs inference on an input image, given an ExecutableNetwork
    '''
    img_pairs = load_images(args)
    exec_net = load_to_IE(args.model, img_pairs[0])

    idx = 0
    for pair in img_pairs:
        # Repackage input image pairs as np.ndarray
        x = np.array([pair])

        # padding for 64 alignment
        print('frame.shape:', x.shape)
        x_adapt, x_adapt_info = sample_utils.adapt_x(x)
        print('adapt.shape:', x_adapt.shape)
        y_adapt_info = None
        if x_adapt_info is not None:
            y_adapt_info = (x_adapt_info[0], x_adapt_info[2], x_adapt_info[3], 2)
        input_image = x_adapt.transpose((0, 3, 1, 2))    # BHWC --> BCHW
            
        # inference
        start = time.time()
        y_hat = exec_net.infer({'x_tnsr':input_image})
        print(time.time()-start)

        # restore to orignal resolution, cut off the padding
        flow = y_hat['pwcnet/flow_pred']
        imagename = f'output/fw_ov{idx:04d}.png'
        flow = flow.transpose((0, 2, 3, 1))  # BCHW --> BHWC
        flow = sample_utils.postproc_y_hat(flow, y_adapt_info)
        flow = np.squeeze(flow, axis=0)
        print (flow.shape)
        cv2.imwrite(imagename, flow_to_img(flow))
        idx += 1

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Load an IR into the Inference Engine")
    # -- Create the descriptions for the commands
    model_desc = "location of the model XML file"
    input_desc = "location of the image input"

    parser.add_argument("--model", default='../model_ir/frozen.xml', help=model_desc)
    parser.add_argument("--input", default='./samples', help=input_desc)

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    inference(args)

if __name__ == "__main__":
    main()