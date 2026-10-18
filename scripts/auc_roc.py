import logging
import os
import sys

lib_path = os.path.abspath("").replace("scripts", "src")
sys.path.append(lib_path)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
import glob
import argparse
import torch
import matplotlib.pyplot as plt
from data.dataloader import build_train_test_dataset
from tqdm.auto import tqdm
from models import networks
from configs.base import Config
from utils.device import get_device
from sklearn.metrics import roc_curve, auc
import numpy as np


def eval(cfg, checkpoint_path, all_state_dict=True):
    device = get_device()
    network = getattr(networks, cfg.model_type)(cfg)
    network.to(device)

    # Build dataset
    _, test_ds = build_train_test_dataset(cfg)
    weight = torch.load(checkpoint_path, map_location=device)
    if all_state_dict:
        weight = weight["state_dict_network"]

    network.load_state_dict(weight, strict=False)
    network.eval()
    network.to(device)

    y_actu = []
    y_pred = []

    skipped = 0
    for every_test_list in tqdm(test_ds):
        input_ids, audio, label = every_test_list
        input_ids = input_ids.to(device)
        audio = audio.to(device)
        label = label.to(device)
        try:
            with torch.no_grad():
                output = network(input_ids, audio)[0]
                y_actu.append(label.detach().cpu().numpy()[0])
                y_pred.append(output.detach().cpu().numpy()[0])
        except RuntimeError as e:
            skipped += 1
            logging.warning(f"Skipping sample (audio len {audio.shape[-1]}): {e}")
    if skipped:
        logging.warning(f"Skipped {skipped} sample(s) that ran out of GPU memory.")
    return np.array(y_actu), np.array(y_pred)


def main(args):
    logging.info("Loading checkpoint")
    test_set = args.test_set if args.test_set is not None else "test.pkl"

    ckpt = args.checkpoint_path
    cfg_path = os.path.join(ckpt, "cfg.log")
    if args.latest:
        ckpt_path = glob.glob(os.path.join(ckpt, "weights", "*.pt"))
        if len(ckpt_path) != 0:
            ckpt_path = ckpt_path[0]
            all_state_dict = True
        else:
            ckpt_path = glob.glob(os.path.join(ckpt, "weights", "*.pth"))[0]
            all_state_dict = False
    else:
        ckpt_path = os.path.join(ckpt, "weights/best_acc/checkpoint_0_0.pt")
        all_state_dict = True
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(ckpt, "weights/best_acc/checkpoint_0.pth")
            all_state_dict = False

    cfg = Config()
    cfg.load(cfg_path)
    # Change to test set
    cfg.data_valid = test_set
    if args.data_root is not None:
        assert (
            args.data_name is not None
        ), "Change validation dataset requires data_name"
        cfg.data_root = args.data_root
        cfg.data_name = args.data_name

    y_true, y_pred = eval(cfg, ckpt_path, all_state_dict=all_state_dict)

    class_unique = sorted(set(y_true.tolist()))
    label_names = args.label_names.split(",") if args.label_names else None
    if label_names is None:
        if label_names is None and cfg.num_classes == 4:
            label_names = ["Angry", "Happy", "Sad", "Neutral"]
        elif cfg.num_classes == 5:
            label_names = ["Angry", "Happy", "Sad", "Neutral", "Surprise"]
        else:
            label_names = [str(c) for c in class_unique]

    plt.figure(figsize=(8, 5.5))
    for cls in class_unique:
        y_true_bin = y_true == cls
        y_score = y_pred[:, cls]
        fpr, tpr, _ = roc_curve(y_true_bin, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{label_names[cls]} (AUC = {roc_auc:.4f})")

    # Plot random guess line
    plt.plot([0, 1], [0, 1], "r--", label="Random Guess")

    # Set labels and title
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    out_path = f"AUC_ROC_{cfg.name}_{cfg.data_valid}.png"
    plt.savefig(out_path, format="png", dpi=300)
    plt.close()
    logging.info(f"Saved ROC curve to {out_path}")


def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-ckpt", "--checkpoint_path", type=str, required=True, help="path to checkpoint folder"
    )

    parser.add_argument(
        "-l",
        "--latest",
        action="store_true",
        help="whether to use latest weight or best weight",
    )

    parser.add_argument(
        "-t",
        "--test_set",
        type=str,
        default=None,
        help="name of testing set. Ex: test.pkl",
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="If want to change the validation dataset",
    )
    parser.add_argument(
        "--data_name", type=str, default=None, help="for changing validation dataset"
    )
    parser.add_argument(
        "--label_names",
        type=str,
        default=None,
        help="Comma-separated class names in label-index order, e.g. Angry,Happy,Sad,Neutral",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    main(args)
