#!/usr/bin/env python3
import os
import logging
import os.path as osp
from utils.inout import get_root_project

# logging
logging.basicConfig(level=logging.INFO)

import hydra
from omegaconf import DictConfig

# --- SAM2 model URLs (official) ---
MODEL_DICT = {
    "sam2_hiera_tiny":      "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
    "sam2_hiera_small":     "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
    "sam2_hiera_base_plus": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
    "sam2_hiera_large":     "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
}

# --- Config YAMLs from the official repo ---
CONFIG_URLS = {
    "sam2.1_hiera_t.yaml":  "https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2.1/sam2.1_hiera_t.yaml",
    "sam2.1_hiera_s.yaml":  "https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2.1/sam2.1_hiera_s.yaml",
    "sam2.1_hiera_b+.yaml": "https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2.1/sam2.1_hiera_b%2B.yaml",
    "sam2.1_hiera_l.yaml":  "https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
}

def _wget(url: str, dst_dir: str):
    os.makedirs(dst_dir, exist_ok=True)
    out = osp.join(dst_dir, url.split("/")[-1])
    cmd = f"wget -O '{out}' '{url}' --no-check-certificate"
    logging.info(f"Downloading: {url} -> {out}")
    rc = os.system(cmd)
    if rc != 0:
        raise RuntimeError(f"wget failed for {url}")
    return out

def download_model(url: str, output_path: str):
    return _wget(url, output_path)

def download_config_files(output_path: str):
    cfg_root = osp.join(output_path, "configs", "sam2.1")
    os.makedirs(cfg_root, exist_ok=True)
    for name, url in CONFIG_URLS.items():
        dst = osp.join(cfg_root, name)
        if osp.exists(dst):
            logging.info(f"Config exists, skipping: {dst}")
            continue
        _wget(url, cfg_root)
        logging.info(f"Downloaded {name}")

@hydra.main(
    version_base=None,
    config_path="./configs",
    config_name="download",
)
def download(cfg: DictConfig) -> None:
    # allow overriding via Hydra if you want: python download_sam2.py model_name=sam2_hiera_small
    model_name = getattr(cfg, "model_name", "sam2_hiera_large")

    save_dir = osp.join(get_root_project(), "checkpoints", "sam2")
    os.makedirs(save_dir, exist_ok=True)

    if model_name not in MODEL_DICT:
        raise KeyError(f"Unknown model_name '{model_name}'. Valid: {list(MODEL_DICT.keys())}")

    logging.info(f"Downloading SAM2 weights: {model_name}")
    download_model(MODEL_DICT[model_name], save_dir)

    logging.info("Downloading SAM2 config YAMLs")
    download_config_files(save_dir)

    logging.info(f"Done. Files saved under: {save_dir}")

if __name__ == "__main__":
    download()
