import argparse
import os
import json


def expandpath(path):
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def command_line_parser():
    parser = argparse.ArgumentParser(
        add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # -------------------------- wandb settings --------------------------
    parser.add_argument(
        "--project",
        type=str,
        default="LabelToTarget",
        help="Name for your run to wandb project.",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name for your run to easier identify it.",
    )

    # -------------------------- logging settings --------------------------
    parser.add_argument(
        "--log_dir",
        type=expandpath,
        default="/cluster/scratch/$USER/logs",
        help="Place for artifacts and logs",
    )
    parser.add_argument(
        "--use_wandb", type=str2bool, default=True, help="Use WandB for logging"
    )
    parser.add_argument(
        "--shared", type=str2bool, default=False, help="Push to shared wandb project"
    )
    # -------------------------- training settings --------------------------
    parser.add_argument(
        "--num_epochs_final",
        type=int,
        default=55,
        help="Number of training epochs for the final segmentation net",
    )
    parser.add_argument(
        "--num_epochs_labeltotarget", type=int, default=500, help="Number of training epochs"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=expandpath,
        help="Path to the LabelToTarget net's checkpoint (leave empty if should be trained)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of samples in a batch for training",
    )
    parser.add_argument(
        "--batch_size_validation",
        type=int,
        default=4,
        help="Number of samples in a batch for validation",
    )
    parser.add_argument(
        "--reconstruction_weight",
        type=float,
        default=10,
        help="Weight assigned to the reconstruction loss",
    )
    parser.add_argument(
        "--gan_noise",
        type=float,
        default=0.7,
        help="Noise used to generate images",
    )
    parser.add_argument(
        '--lr_scheduler_power_final', type=float, default=0.95, help='Poly learning rate power')

    # -------------------------- model settings --------------------------
    parser.add_argument(
        "--generator_filters",
        type=int,
        default=32,
        help="Filters for the CycleGAN generators",
    )

    parser.add_argument(
        "--discriminator_filters",
        type=int,
        default=32,
        help="Filters for the CycleGAN discriminator",
    )

    # -------------------------- data settings --------------------------
    parser.add_argument(
        "--dataset_root", type=expandpath, default="/cluster/scratch/$USER/dl_data/data", help="Path to dataset",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Size training images should be scaled to",
    )
    
    # -------------------------- hardware settings --------------------------
    parser.add_argument("--gpu", type=str2bool, default=True, help="GPU usage")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker threads fetching training data",
    )
    parser.add_argument(
        "--workers_validation",
        type=int,
        default=4,
        help="Number of worker threads fetching validation data",
    )

    cfg = parser.parse_args()

    print(json.dumps(cfg.__dict__, indent=4, sort_keys=True))

    return cfg
