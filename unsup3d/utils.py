import os
import sys
import glob
import yaml
import random
import numpy as np
import cv2
import torch
import pytorch3d
import zipfile


def setup_runtime(args):
    """Load configs, initialize CUDA, CuDNN and the random seeds."""

    # Setup CUDA
    cuda_device_id = args.gpu
    if cuda_device_id is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device_id)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    # Setup random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ## Load config
    cfgs = {}
    if args.config is not None and os.path.isfile(args.config):
        cfgs = load_yaml(args.config)

    cfgs['config'] = args.config
    cfgs['seed'] = args.seed
    cfgs['num_workers'] = args.num_workers
    cfgs['device'] = 'cuda:0' if torch.cuda.is_available() and cuda_device_id is not None else 'cpu'

    print(f"Environment: GPU {cuda_device_id} seed {args.seed} number of workers {args.num_workers}")
    return cfgs


def load_yaml(path):
    print(f"Loading configs from {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def dump_yaml(path, cfgs):
    print(f"Saving configs to {path}")
    xmkdir(os.path.dirname(path))
    with open(path, 'w') as f:
        return yaml.safe_dump(cfgs, f)


def xmkdir(path):
    """Create directory PATH recursively if it does not exist."""
    os.makedirs(path, exist_ok=True)


def clean_checkpoint(checkpoint_dir, keep_num=2):
    if keep_num > 0:
        names = list(sorted(
            glob.glob(os.path.join(checkpoint_dir, 'checkpoint*.pth'))
        ))
        if len(names) > keep_num:
            for name in names[:-keep_num]:
                print(f"Deleting obslete checkpoint file {name}")
                os.remove(name)


def archive_code(arc_path, filetypes=['.py', '.yml']):
    print(f"Archiving code to {arc_path}")
    # xmkdir(os.path.dirname(arc_path))
    # zipf = zipfile.ZipFile(arc_path, 'w', zipfile.ZIP_DEFLATED)
    # cur_dir = os.getcwd()
    # flist = []
    # for ftype in filetypes:
    #     flist.extend(glob.glob(os.path.join(cur_dir, '**', '*'+ftype), recursive=True))
    # [zipf.write(f, arcname=f.replace(cur_dir,'archived_code', 1)) for f in flist]
    # zipf.close()


def get_model_device(model):
    return next(model.parameters()).device


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def draw_bbox(im, size):
    b, c, h, w = im.shape
    h2, w2 = (h-size)//2, (w-size)//2
    marker = np.tile(np.array([[1.],[0.],[0.]]), (1,size))
    marker = torch.FloatTensor(marker)
    im[:, :, h2, w2:w2+size] = marker
    im[:, :, h2+size, w2:w2+size] = marker
    im[:, :, h2:h2+size, w2] = marker
    im[:, :, h2:h2+size, w2+size] = marker
    return im


def save_videos(out_fold, imgs, prefix='', suffix='', sep_folder=True, ext='.mp4', cycle=False):
    if sep_folder:
        out_fold = os.path.join(out_fold, suffix)
    xmkdir(out_fold)
    prefix = prefix + '_' if prefix else ''
    suffix = '_' + suffix if suffix else ''
    offset = len(glob.glob(os.path.join(out_fold, prefix+'*'+suffix+ext))) +1

    imgs = imgs.transpose(0,1,3,4,2)  # BxTxCxHxW -> BxTxHxWxC
    for i, fs in enumerate(imgs):
        if cycle:
            fs = np.concatenate([fs, fs[::-1]], 0)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # fourcc = cv2.VideoWriter_fourcc(*'avc1')
        vid = cv2.VideoWriter(os.path.join(out_fold, prefix+'%05d'%(i+offset)+suffix+ext), fourcc, 5, (fs.shape[2], fs.shape[1]))
        [vid.write(np.uint8(f[...,::-1]*255.)) for f in fs]
        vid.release()


def save_images(out_fold, imgs, prefix='', suffix='', sep_folder=True, ext='.png'):
    if sep_folder:
        out_fold = os.path.join(out_fold, suffix)
    xmkdir(out_fold)
    prefix = prefix + '_' if prefix else ''
    suffix = '_' + suffix if suffix else ''
    offset = len(glob.glob(os.path.join(out_fold, prefix+'*'+suffix+ext))) +1

    imgs = imgs.transpose(0,2,3,1)
    for i, img in enumerate(imgs):
        if 'depth' in suffix:
            im_out = np.uint16(img[...,::-1]*65535.)
        else:
            im_out = np.uint8(img[...,::-1]*255.)
        cv2.imwrite(os.path.join(out_fold, prefix+'%05d'%(i+offset)+suffix+ext), im_out)


def save_txt(out_fold, data, prefix='', suffix='', sep_folder=True, ext='.txt'):
    if sep_folder:
        out_fold = os.path.join(out_fold, suffix)
    xmkdir(out_fold)
    prefix = prefix + '_' if prefix else ''
    suffix = '_' + suffix if suffix else ''
    offset = len(glob.glob(os.path.join(out_fold, prefix+'*'+suffix+ext))) +1

    [np.savetxt(os.path.join(out_fold, prefix+'%05d'%(i+offset)+suffix+ext), d, fmt='%.6f', delimiter=', ') for i,d in enumerate(data)]


def compute_sc_inv_err(d_pred, d_gt, mask=None):
    b = d_pred.size(0)
    diff = d_pred - d_gt
    if mask is not None:
        diff = diff * mask
        avg = diff.view(b, -1).sum(1) / (mask.view(b, -1).sum(1))
        score = (diff - avg.view(b,1,1))**2 * mask
    else:
        avg = diff.view(b, -1).mean(1)
        score = (diff - avg.view(b,1,1))**2
    return score  # masked error maps


def compute_angular_distance(n1, n2, mask=None):
    dist = (n1*n2).sum(3).clamp(-1,1).acos() /np.pi*180
    return dist*mask if mask is not None else dist


def save_scores(out_path, scores, header=''):
    print('Saving scores to %s' %out_path)
    np.savetxt(out_path, scores, fmt='%.8f', delimiter=',\t', header=header)

def calculate_views_for_360_video(original_view, num_frames=8):
    views = []
    counter = -num_frames/2
    for i in range(num_frames):
        rotation_around_y = -(counter+i)*2*(np.pi/5)/num_frames
        new_view = original_view.detach().clone()
        new_view[:,0] = 0 # rotation around x axis
        new_view[:,1] = rotation_around_y # rotation around y axis
        new_view[:,2] = 0# rotation around z axis
        new_view[:,3] = 0 #x
        new_view[:,4] = 0 #y
        new_view[:,5] = 0.2 #z 0.6 for dogs
        views.append(new_view)
    
    return torch.stack(views)

def get_side_view(original_view, zoom_mode=0):
    if zoom_mode == 0:
        new_view = original_view.detach().clone()
        new_view[:,0] = 0 # rotation around x axis
        new_view[:,1] = -np.pi/2 # rotation around y axis
        new_view[:,2] = 0# rotation around z axis
        new_view[:,3] = 0 #x 
        new_view[:,4] = 0 #y
        new_view[:,5] = 0 #z zoom out a little bit
        return new_view
    if zoom_mode == 1:
        new_view = original_view.detach().clone()
        new_view[:,0] = 0 # rotation around x axis
        new_view[:,1] = -np.pi/2 # rotation around y axis
        new_view[:,2] = 0# rotation around z axis
        new_view[:,3] = 0.05 #x 
        new_view[:,4] = 0 #y
        new_view[:,5] = 0.2 #z zoom out a little bit
        return new_view


def get_gaussian_like_blub(kernel_size=32):
    sigma = 4

    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*np.pi*variance)) *\
                    torch.exp(
                        -torch.sum((xy_grid - mean)**2., dim=-1) /\
                        (2*variance)
                    )
    # Make sure sum of values in gaussian kernel equals 1.
    # gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    return gaussian_kernel*200

def normalize_tensor(in_feat,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=0,keepdim=True))
    return in_feat/(norm_factor+eps)