import torch
import os, glob
import numpy as np
import tifffile as tiff
import torch.nn as nn
import argparse
from utils.data_utils import imagesc


def get_aug(x0, aug, backward=False):  # X Y Z
    """Apply augmentation to the input volume.
    
    Args:
        x0: Input volume of shape (X, Y, Z)
        aug: Augmentation type (0=none, 1=transpose, 2=flip X, 3=flip Z, 4=flip both)
        backward: Whether to apply the inverse transformation
        
    Returns:
        Augmented volume
    """
    if not backward:
        if aug == 1:
            x0 = np.transpose(x0, (1, 0, 2))
        elif aug == 2:
            x0 = x0[:, ::-1, :]
        elif aug == 3:
            x0 = x0[:, :, ::-1]
        elif aug == 4:
            x0 = x0[:, ::-1, ::-1]
    else:
        if aug == 1:
            x0 = np.transpose(x0, (1, 0, 2))
        elif aug == 2:
            x0 = x0[:, ::-1, :]
        elif aug == 3:
            x0 = x0[:, :, ::-1]
        elif aug == 4:
            x0 = x0[:, ::-1, ::-1]
    return x0.copy()


def get_one(x0, aug, net, extra_dsp, mirror_padding):
    """Process a single volume with the network.
    
    Args:
        x0: Input volume
        aug: Augmentation type
        net: Neural network model
        extra_dsp: Extra downsampling factor
        mirror_padding: Mirror padding size
        residual: Whether to add residual connection
        
    Returns:
        Tuple of (input, output) volumes after processing
    """
    x0 = get_aug(x0, aug)
    x0 = torch.from_numpy(x0).unsqueeze(0).unsqueeze(0).float()  # (B, C, H, W, D)
    x0 = x0[:, :, 64:-64, 64:-64, :]
    print(x0.shape)

    # extra downsample
    x0 = x0[:, :, :, :, (extra_dsp // 2)::extra_dsp]
    x0 = torch.nn.Upsample(size=(x0.shape[2], x0.shape[3], x0.shape[4] * 2 * extra_dsp), mode='trilinear')(x0)

    # padding
    if mirror_padding > 0:
        padL = torch.flip(x0[:, :, :, :, :mirror_padding], [4])
        padR = torch.flip(x0[:, :, :, :, -mirror_padding:], [4])
        x0 = torch.cat([padL, x0, padR], 4)

    out = net(x0)['out0']  # (X, Y, Z)

    # unpadding
    if mirror_padding > 0:
        out = out[:, :, :, :, mirror_padding:-mirror_padding]
        x0 = x0[:, :, :, :, mirror_padding:-mirror_padding]
    
    x0 = x0.squeeze().detach().numpy()
    x0 = get_aug(x0, aug, backward=True)
    out = out.squeeze().detach().numpy()
    out = get_aug(out, aug, backward=True)
    return x0, out


def crop_and_norm(img):
    """Crop and normalize an image.
    
    Args:
        img: Input image
        
    Returns:
        Cropped and normalized image
    """
    print(img.shape)
    img = (img - img.min()) / (img.max() - img.min())
    img = torch.from_numpy(img).unsqueeze(1).repeat(1, 3, 1, 1)
    img = img.permute(1, 2, 3, 0).unsqueeze(0)
    img = img.permute(4, 1, 2, 3, 0).squeeze()
    img = img[:224, 0, ::].numpy()
    return img


def test_IsoLesion(sub, net, root, suffix, aug_list, extra_dsp, mirror_padding):
    """Test the model on a single subject.
    
    Args:
        sub: Path to subject file
        net: Neural network model
        root: Root directory for output
        suffix: Suffix for output filenames
        aug_list: List of augmentations to apply
        extra_dsp: Extra downsampling factor
    """
    subject_name = sub.split('/')[-1]
    x0 = tiff.imread(sub)  # (Z, X, Y)

    print(x0.min(), x0.max())

    x0 = np.transpose(x0, (1, 2, 0))  # (X, Y, Z)
    print(x0.shape)
    trd = [x0.min(), x0.max()]
    print(trd)

    # Normalization
    x0[x0 < trd[0]] = trd[0]
    x0[x0 > trd[1]] = trd[1]
    x0 = (x0 - trd[0]) / (trd[1] - trd[0])
    x0 = (x0 - 0.5) / 0.5

    # augmentations
    out_all = []
    for aug in aug_list:
        xup, out = get_one(x0, aug=aug, net=net, extra_dsp=extra_dsp, 
                          mirror_padding=mirror_padding)
        out_all.append(out)
    out = np.array(out_all).sum(0) / len(aug_list)

    # Create output directories
    os.makedirs(os.path.join(root, 'out', 'xy'), exist_ok=True)
    os.makedirs(os.path.join(root, 'out', 'xz'), exist_ok=True)
    os.makedirs(os.path.join(root, 'out', 'yz'), exist_ok=True)
    os.makedirs(os.path.join(root, 'out', 'xz2d'), exist_ok=True)
    os.makedirs(os.path.join(root, 'out', 'yz2d'), exist_ok=True)

    # Save outputs in different orientations
    # XY
    tiff.imwrite(os.path.join(root, 'out', 'xy', f"{suffix}{subject_name}.tif"), 
                np.transpose(out, (2, 0, 1)))
    # XZ
    tiff.imwrite(os.path.join(root, 'out', 'xz', f"{suffix}{subject_name}.tif"), 
                np.transpose(out, (1, 0, 2)))
    # YZ
    tiff.imwrite(os.path.join(root, 'out', 'yz', f"{suffix}{subject_name}.tif"), out)
    # XZ 2d
    tiff.imwrite(os.path.join(root, 'out', 'xz2d', f"{suffix}{subject_name}.tif"), 
                np.transpose(xup, (1, 0, 2)))
    # YZ 2d
    tiff.imwrite(os.path.join(root, 'out', 'yz2d', f"{suffix}{subject_name}.tif"), xup)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a trained model on TIFF images')
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name')
    
    # Optional arguments
    parser.add_argument('--root', type=str, default='/media/ExtHDD01/Dataset/paired_images',
                       help='Root directory for input/output')
    parser.add_argument('--extra_dsp', type=int, default=4,
                       help='Extra downsampling factor')
    parser.add_argument('--mirror_padding', type=int, default=0,
                       help='Mirror padding size')
    parser.add_argument('--suffix', type=str, default='',
                       help='Suffix for output filenames')
    parser.add_argument('--num_subjects', type=int, default=2,
                       help='Number of subjects to process')
    
    args = parser.parse_args()
    
    # Setup paths
    root = os.path.join(args.root, args.dataset)
    checkpoint_path = args.checkpoint
    
    # Ensure root directory exists
    if not os.path.exists(root):
        raise ValueError(f"Root directory {root} does not exist")
    
    # Find subjects
    subjects = sorted(glob.glob(os.path.join(root, 'ori', '*')))
    if not subjects:
        raise ValueError(f"No subjects found in {os.path.join(root, 'ori')}")
    
    # Create output directories
    os.makedirs(os.path.join(root, 'out'), exist_ok=True)
    
    # Setup augmentation list
    aug_list = [0, 1]
    
    # Process subjects
    print(f"Processing {min(args.num_subjects, len(subjects))} subjects...")
    for sub in subjects[:args.num_subjects]:
        print(f"Loading model from {checkpoint_path}")
        net = torch.load(checkpoint_path, map_location='cpu')
        print(f"Processing subject: {sub}")
        test_IsoLesion(
            sub=sub,
            net=net,
            root=root,
            suffix=args.suffix,
            aug_list=aug_list,
            extra_dsp=args.extra_dsp,
            mirror_padding=args.mirror_padding,
        )
    
    print(f"Results saved to {os.path.join(root, 'out')}")