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
    parser = argparse.ArgumentParser(add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # -------------------------- wandb settings --------------------------
    parser.add_argument(
        "--project",
        type=str,
        default="Gogoll",
        help="Name for your run to wandb project.",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name for your run to easier identify it.",
    )

    # -------------------------- logging settings --------------------------
    parser.add_argument("--log_dir", type=expandpath, default="./logs", help="Place for artifacts and logs")
    parser.add_argument("--use_wandb", type=str2bool, default=False, help="Use WandB for logging")
    parser.add_argument("--shared", type=str2bool, default=False, help="Push to shared wandb project")

    # -------------------------- training settings --------------------------
    parser.add_argument("--num_epochs_seg", type=int, default=16, help="Number of training epochs for the segmentation net")
    parser.add_argument("--num_epochs_gogoll", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--seg_checkpoint_path", type=expandpath, help="Path to the source segmentation net's checkpoint (leave empty if should be trained)")
    parser.add_argument("--gogoll_checkpoint_path", type=expandpath, help="Path to the gogol net's checkpoint (leave empty if should be trained)")
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
        "--segmentation_weight",
        type=float,
        default=0.8,
        help="Weight assigned to the segmentation loss",
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
    parser.add_argument("--dataset_root", type=expandpath, default="./data", help="Path to dataset")
    parser.add_argument(
        "--domain",
        type=str,
        default="domainB",
        choices=["domainA", "domainB"],
        help="Type of the target domain",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Size training images should be scaled to",
    )
    parser.add_argument("--save_generated_images", type=str2bool, default=False, help="Save generated images at the end of training?")
    parser.add_argument("--max_generated_images_saved", type=int, default=100, help="Maximum number of images to generate and save at the end of training")
    parser.add_argument("--generated_dataset_save_root", type=expandpath, default="./output/data", help="Path where you save generated images")

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
        default=2,
        help="Number of worker threads fetching validation data",
    )

    cfg = parser.parse_args()

    print(json.dumps(cfg.__dict__, indent=4, sort_keys=True))

    return cfg
