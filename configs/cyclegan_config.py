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
        "--project", type=str, help="Name for your run to wandb project.",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name for your run to easier identify it.",
    )

    # -------------------------- logging settings --------------------------
    parser.add_argument(
        "--log_dir", type=expandpath, required=True, help="Place for artifacts and logs"
    )
    parser.add_argument(
        "--use_wandb", type=str2bool, default=False, help="Use WandB for logging"
    )

    # -------------------------- training settings --------------------------
    parser.add_argument(
        "--num_epochs", type=int, default=16, help="Number of training epochs"
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
        "--identity_weight",
        type=float,
        default=2,
        help="Weight assigned to the identity loss",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume training from checkpoint, which can also be an AWS link s3://...",
    )

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
        "--dataset_root", type=expandpath, required=True, help="Path to dataset"
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="domainA",
        choices=["domainA", "domainB"],
        help="Type of the target domain",
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
        default=1,
        help="Number of worker threads fetching training data",
    )
    parser.add_argument(
        "--workers_validation",
        type=int,
        default=1,
        help="Number of worker threads fetching validation data",
    )

    cfg = parser.parse_args()

    print(json.dumps(cfg.__dict__, indent=4, sort_keys=True))

    return cfg
