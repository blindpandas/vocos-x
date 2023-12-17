import torch
from torch import nn
from vocos.heads import FourierHead
from vocos.spectral_ops import ISTFT

from . import cnn_stft


class MatchaISTFTHead(FourierHead):
    def __init__(self, dim: int, n_fft: int, hop_length: int, padding: str = "same"):
        super().__init__()
        out_dim = n_fft + 2
        self.out = torch.nn.Linear(dim, out_dim)
        self.istft = ISTFT(n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.out(x).transpose(1, 2)
        mag, p = x.chunk(2, dim=1)
        # I asume this is here because it expects a log-normalized melspectogram 
        # mag = torch.exp(mag)
        mag = torch.clip(mag, max=1e2)  # safeguard to prevent excessively large magnitudes
        # wrapping happens here. These two lines produce real and imaginary value
        x = torch.cos(p)
        y = torch.sin(p)
        S = mag * (x + 1j * y)
        audio = self.istft(S)
        return audio


class ONNXExportableSTFTHead(FourierHead):

    def __init__(self, dim: int, n_fft: int, hop_length: int, win_length: int):
        super().__init__()
        out_dim = n_fft + 2
        self.out = torch.nn.Linear(dim, out_dim)
        self.istft = cnn_stft.STFT(
            filter_length=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.out(x).transpose(1, 2)
        mag, p = x.chunk(2, dim=1)
        # I asume this is here because it expects log-normalized melspectorgram  
        # mag = torch.exp(mag)
        mag = torch.clip(mag, max=1e2)  # safeguard to prevent excessively large magnitudes
        # wrapping happens here. These two lines produce real and imaginary value
        x = torch.cos(p)
        y = torch.sin(p)
        audio = self.istft.inverse(x, y)
        return audio


