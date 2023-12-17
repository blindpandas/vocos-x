import tempfile
from pathlib import Path

import torch
import soundfile as sf

from matcha import MatchaMelSpectrogramFeatures
from vocos.dataset import DataConfig, VocosDataset
from vocos.pretrained import Vocos


# Set these paths first
EXPORTED_VOCOS_PT_PATH = "./checkpoints/vocos-hfc-female.pt"
OUTPUT_DIR = "./rewavs"
FILES_TO_VOCODE = [
    "wavs/1.wav",
    "wavs/2.wav",
    "wavs/3.wav",
    "wavs/4.wav",
    "wavs/5.wav",
]

##########
_tempdir = tempfile.TemporaryDirectory()
FILELIST_FILE = Path(_tempdir.name).joinpath("filelist.txt")
FILELIST_FILE.write_text("\n".join(FILES_TO_VOCODE), encoding="utf-8")

cfg = DataConfig(
    filelist_path=FILELIST_FILE,
    sampling_rate=22050,
    # 15 seconds
    num_samples=330750,
    batch_size=1,
    num_workers=1
)
dataset = VocosDataset(cfg=cfg, train=True)
feat_ex = MatchaMelSpectrogramFeatures(
    sample_rate=22050,
    n_fft=1024,
    hop_length=256,
    n_mels=80,
    f_min=0,
    f_max=8000,
    center=False,
    mel_mean=-6.38385,
    mel_std=2.541796
)
vocos = torch.jit.load(EXPORTED_VOCOS_PT_PATH)
vocos = vocos.eval()
outdir = Path(OUTPUT_DIR)
outdir.mkdir(exist_ok=True, parents=True)
for i in range(len(dataset)):
    aud = dataset[i].unsqueeze(0)
    mel = feat_ex(aud)
    reaud = vocos(mel).squeeze().detach ().numpy()
    sf.write(
        outdir.joinpath(f"{i + 1}_vocos.wav"),
        reaud,
        22050,
        "FLOAT"
    )