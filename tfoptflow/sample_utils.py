###
# Sample mgmt
###
import numpy as np

def adapt_x(x, pyr_lvls=6):
    """Preprocess the input samples to adapt them to the network's requirements
    Here, x, is the actual data, not the x TF tensor.
    Args:
        x: input samples in list[(2,H,W,3)] or (N,2,H,W,3) np array form
    Returns:
        Samples ready to be given to the network (w. same shape as x)
        Also, return adaptation info in (N,2,H,W,3) format
    """
    # Ensure we're dealing with RGB image pairs
    assert (isinstance(x, np.ndarray) or isinstance(x, list))
    if isinstance(x, np.ndarray):
        assert (len(x.shape) == 5)
        assert (x.shape[1] == 2 and x.shape[4] == 3)
    else:
        assert (len(x[0].shape) == 4)
        assert (x[0].shape[0] == 2 or x[0].shape[3] == 3)

    # Bring image range from 0..255 to 0..1 and use floats (also, list[(2,H,W,3)] -> (batch_size,2,H,W,3))
    x_adapt = np.array(x, dtype=np.float32) if isinstance(x, list) else x.astype(np.float32)
    x_adapt /= 255.

    # Make sure the image dimensions are multiples of 2**pyramid_levels, pad them if they're not
    _, pad_h = divmod(x_adapt.shape[2], 2 ** pyr_lvls)
    if pad_h != 0:
        pad_h = 2 ** pyr_lvls - pad_h
    _, pad_w = divmod(x_adapt.shape[3], 2 ** pyr_lvls)
    if pad_w != 0:
        pad_w = 2 ** pyr_lvls - pad_w
    x_adapt_info = None
    if pad_h != 0 or pad_w != 0:
        padding = [(0, 0), (0, 0), (0, pad_h), (0, pad_w), (0, 0)]
        x_adapt_info = x_adapt.shape  # Save original shape
        x_adapt = np.pad(x_adapt, padding, mode='constant', constant_values=0.)

    return x_adapt, x_adapt_info

def postproc_y_hat(pred_flows, adapt_info=None):
    """Postprocess the results coming from the network during the test mode.
    Here, y_hat, is the actual data, not the y_hat TF tensor. Override as necessary.
    Args:
        y_hat: predictions, see set_output_tnsrs() for details
        adapt_info: adaptation information in (N,H,W,2) format
    Returns:
        Postprocessed labels
    """

    # Have the samples been padded to fit the network's requirements? If so, crop flows back to original size.
    if adapt_info is not None:
        pred_flows = pred_flows[:, 0:adapt_info[1], 0:adapt_info[2], :]

    return pred_flows