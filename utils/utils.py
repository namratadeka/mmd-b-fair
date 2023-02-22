import torch
import random
import numpy as np
from os.path import join


def seed_everything(seed=0, harsh=False):
    """
    Seeds all important random functions
    Args:
        seed (int, optional): seed value. Defaults to 0.
        harsh (bool, optional): torch backend deterministic. Defaults to False.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    if harsh:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
def dict_to_device(d_ten: dict, device):
    """
    Sets a dictionary to device
    Args:
        d_ten (dict): dictionary of tensors
        device (str): torch device
    Returns:
        dict: dictionary on device
    """
    for key, tensor in d_ten.items():
        if type(tensor) is torch.Tensor:
            d_ten[key] = d_ten[key].to(device)

    return d_ten

def zero_diag(K):
    K_ = K.clone()
    if len(K_.shape) > 2:
        for i in range(len(K_)):
            K_[i] = K_[i].fill_diagonal_(0.0)
    else:
        K_ = K_.fill_diagonal_(0.0)
    return K_

def save_model(model, opt, epoch, loss, outpath, ckpt_type):
    save_dict = {
        'state_dict': model.state_dict(),
        'optimizer': opt.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(save_dict, join(outpath, f'{ckpt_type}.pth'))

def copy_state_dict(cur_state_dict, pre_state_dict, prefix=""):
    """
        Load parameters
    Args:
        cur_state_dict (dict): current parameters
        pre_state_dict ([type]): load parameters
        prefix (str, optional): specific module names. Defaults to "".
    """

    def _get_params(key):
        key = prefix + key
        try:
            out = pre_state_dict[key]
        except Exception:
            try:
                out = pre_state_dict[key[7:]]
            except Exception:
                try:
                    out = pre_state_dict["module." + key]
                except Exception:
                    try:
                        out = pre_state_dict[key[14:]]
                    except Exception:
                        out = None
        return out

    for k in cur_state_dict.keys():
        v = _get_params(k)
        try:
            if v is None:
                print("parameter {} not found".format(k))
                # logging.info("parameter {} not found".format(k))
                continue
            cur_state_dict[k].copy_(v)
        except Exception:
            print("copy param {} failed".format(k))
            # logging.info("copy param {} failed".format(k))
            continue

def whiten(X, mn, std):
    mntile = np.tile(mn, (X.shape[0], 1))
    stdtile = np.maximum(np.tile(std, (X.shape[0], 1)), 1e-8)
    X = X - mntile
    X = np.divide(X, stdtile)
    return X

def reset_weights(network):
    for layers in network.children():
        for layer in layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
