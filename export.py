# coding: utf-8

import argparse
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from vocos.pretrained import instantiate_class


_LOGGER = logging.getLogger("export_torch_script")


class VocosGen(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, mels):
        x = self.backbone(mels)
        audio = self.head(x)
        return audio


def export_generator(config_path, checkpoint_path, output_dir, format):

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    class_module, class_name = config["model"]["class_path"].rsplit(".", 1)
    module = __import__(class_module, fromlist=[class_name])
    vocos_cls = getattr(module, class_name)

    params = config["model"]["init_args"]

    checkpoint = torch.load(checkpoint_path)
    if isinstance(checkpoint, dict):
        feature_extractor = instantiate_class(args=(), init=params["feature_extractor"])
        backbone = instantiate_class(args=(), init=params["backbone"])
        head = instantiate_class(args=(), init=params["head"])
        vocos = vocos_cls.load_from_checkpoint(
            checkpoint_path,
            map_location="cpu",
            feature_extractor=feature_extractor,
            backbone=backbone,
            head=head,
            sample_rate=params["sample_rate"],
            initial_learning_rate=params["initial_learning_rate"],
            num_warmup_steps=params["num_warmup_steps"],
            mel_loss_coeff=params["mel_loss_coeff"],
            mrd_loss_coeff=params["mrd_loss_coeff"],
        )
    else:
        vocos = checkpoint
    vocos = vocos.cpu().eval()
    vocos._jit_is_scripting = True

    # Reinitialize from state-dict to avoid copying unused components
    exp_backbone = instantiate_class(args=(), init=params["backbone"])
    exp_head = instantiate_class(args=(), init=params["head"])
    exp_backbone.load_state_dict(vocos.backbone.state_dict()) 
    exp_head.load_state_dict(vocos.head.state_dict())
    model = VocosGen(exp_backbone, exp_head)
    model = model.eval()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ext = "pt" if format == "sm" else "ckpt"
    export_filename = f"{Path(checkpoint_path).stem}.{ext}"
    export_path = os.path.join(output_dir, export_filename)

    if format == "sm":
        args = (torch.rand(1, vocos.backbone.input_channels, 64),)
        traced_script_module  = torch.jit.trace(model, args)
        traced_script_module .save(export_path)
    else:
        torch.save(model, export_path)
    return export_path


def main():
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser(
        prog="export",
        description="Export a model checkpoint to torch script",
    )

    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-i", "--ckpt", type=str, required=True)
    formats = [
        # Script module
        "sm",
        # Model checkpoint
        "ckpt"
    ]
    parser.add_argument("--format", type=str, choices=formats, default="sm")
    parser.add_argument("-o", "--out-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1234, help="random seed")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    _LOGGER.info("Exporting model")
    _LOGGER.info(f"Config path: `{args.config}`")
    _LOGGER.info(f"Using checkpoint: `{args.ckpt}`")
    export_path = export_generator(
        config_path=args.config,
        checkpoint_path=args.ckpt,
        output_dir=args.out_dir,
        format=args.format
    )
    _LOGGER.info(f"Exported model to: `{export_path}`")


if __name__ == '__main__':
    main()