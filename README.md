# IsotropicOAI

This is the code base for our paper "Self-Supervised 3D Super-Resolution and Diffusion-Based Inpainting for Enhanced
Abnormality Detection in Anisotropic Musculoskeletal MRI"
Authors: Jui-Yo Hsu, Pin-Hsun Lian, Tzu-Yi Chuang, Gary Han Chang


## Installation

### Prerequisites

- Python 3.7+
- CUDA-compatible GPU
- PyTorch 1.7+
- PyTorch Lightning 1.5+

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/MicroCoreNTU/IsotropicOAI.git
   cd IsotropicOAI
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Create a `.env` file in the `env` directory
   - Configure dataset paths and other environment-specific settings

## Project Structure

```
IsotropicOAI/
├── dataloader/           # Data loading utilities
├── env/                  # Environment configurations
│   ├── env               # Environment settings
│   ├── jsn/              # JSON configuration files
├── models/               # Model definitions
│   ├── base.py           # Base model class
│   ├── IsoREF.py         # IsoREF model implementation
├── networks/             # Network architectures
├── utils/                # Utility functions
├── train.py              # Training script
├── test.py               # Testing script
└── README.md             # This file
```

## Usage

### Training
Example:
```bash
python train.py --jsn default --prj womac4 --dataset womac4 --directions ori --models IsoREF --env default --n_epochs 801 --gdim 3d_3d --use_mlp --epoch_save 100 --lbNCE 10 --lamb 1 --l1how max
```

### Testing
Example:
```bash
python test.py --checkpoint checkpoints/womac4.pth --dataset womac4 --extra_dsp 1
```

## Command-Line Arguments

### Training Arguments

| Argument | Description                                          | Default  |
|----------|------------------------------------------------------|----------|
| `--jsn` | JSON configuration file name (without extension)     | Required |
| `--prj` | Project name (used for logging)                      | Required |
| `--models` | Model type to use (e.g., IsoREF)                     | Required |
| `--env` | Environment configuration name                       | Required |
| `--n_epochs` | Number of training epochs                            | 200      |
| `--epoch_save` | Save checkpoint every N epochs                       | 20       |
| `--batch_size` | Training batch size                                  | 1        |
| `--direction` | image folder name                                    | ori      |
| `--lr` | Learning rate                                        | 0.0002   |
| `--beta1` | Beta1 parameter for Adam optimizer                   | 0.5      |
| `--gdim` | Generator dimensions (encoder / decoder e.g., 3d_3d) | 3d_3d    |
| `--use_mlp` | Use MLP for contrastive learning                     | False    |
| `--nocut` | Disable contrastive learning                         | False    |
| `--lbNCE` | Weight for NCE loss                                  | 1.0      |
| `--lamb` | Weight for L1 loss                                   | 10.0     |
| `--l1how` | L1 loss aggregation method (dsp, mean or max)        | max      |
| `--lbNCE` | Weight for NCE loss                                  | 10.0     |
| `--gan_mode` | GAN loss mode                                        | vallina  |

### Testing Arguments

| Argument | Description | Default                               |
|----------|-------------|---------------------------------------|
| `--checkpoint` | Path to model checkpoint | Required                              |
| `--dataset` | Dataset name | Required                              |
| `--root` | Root directory for input/output | /media/ExtHDD01/Dataset/paired_images |
| `--extra_dsp` | Extra downsampling factor | １                                     |
| `--num_subjects` | Number of subjects to process | 2                                     |


### Environment Configuration

Environment settings in the `env/env` file define paths to datasets and logs:

```json
{
  "runpod": {
    "DATASET": "/path/to/datasets/",
    "LOGS": "/path/to/logs/"
  }
}
```
## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code for your research, please cite our paper:

```
@article{IsoScopeX2025,
  title={IsoScopeX: Deep Learning Framework for Isotropic Resolution Enhancement in Medical Imaging},
  author={Your Name and Co-authors},
  journal={Journal Name},
  year={2025}
}
```


