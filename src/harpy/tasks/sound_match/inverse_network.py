"""Small optional Torch network; imported only by explicit learned operations."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from harpy.tasks.sound_match.inverse_features import (
    NOTE_BINS,
    architecture_configuration,
    conditioning,
    decode_prediction,
    log_spectrogram,
)


class InverseNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        configuration = architecture_configuration()
        layers = []
        for incoming, outgoing, kernel, stride, padding in configuration["convolutions"]:
            layers.extend(
                (
                    nn.Conv2d(incoming, outgoing, kernel, stride, padding, dtype=torch.float32),
                    nn.ReLU(),
                )
            )
        pooled = configuration["adaptive_average_pool"]
        layers.extend((nn.AdaptiveAvgPool2d(tuple(pooled)), nn.Flatten()))
        self.encoder = nn.Sequential(*layers)
        width = configuration["fusion_hidden"]
        flattened = configuration["convolutions"][-1][1] * pooled[0] * pooled[1]
        self.fusion = nn.Sequential(
            nn.Linear(flattened + configuration["conditioning_size"], width, dtype=torch.float32),
            nn.ReLU(),
        )
        self.notes = nn.Linear(width, NOTE_BINS, dtype=torch.float32)
        self.waveform = nn.Linear(width, configuration["waveform_logits"], dtype=torch.float32)
        self.scalars = nn.Linear(width, configuration["scalar_logits"], dtype=torch.float32)

    def forward(self, spectrogram, condition):
        encoded = self.encoder(spectrogram)
        hidden = self.fusion(torch.cat((encoded, condition), dim=1))
        return {
            "notes": self.notes(hidden),
            "waveforms": self.waveform(hidden),
            "scalars": self.scalars(hidden),
        }


def supervised_loss(predictions: dict, targets: dict):
    """Exactly equal eligible field weights, followed by an equal example mean."""
    counts = targets["counts"][:, None]
    notes = functional.binary_cross_entropy_with_logits(
        predictions["notes"],
        targets["notes"],
        pos_weight=(NOTE_BINS - counts) / counts,
        reduction="none",
    ).mean(dim=1)
    waveform = functional.cross_entropy(
        predictions["waveforms"], targets["waveforms"], reduction="none"
    )
    scalars = (torch.sigmoid(predictions["scalars"]) - targets["scalars"]).square()
    fields = torch.cat((notes[:, None], waveform[:, None], scalars), dim=1)
    eligible = targets["eligible"]
    return ((fields * eligible).sum(dim=1) / eligible.sum(dim=1).clamp_min(1)).mean()


def predict_patch(model: InverseNetwork, audio: np.ndarray, current_patch, editable_fields):
    features = torch.from_numpy(log_spectrogram(audio))[None, None]
    condition = torch.from_numpy(conditioning(current_patch, editable_fields))[None]
    with torch.inference_mode():
        heads = model(features, condition)
        notes = heads["notes"][0].cpu().numpy()
        waveform = heads["waveforms"][0].cpu().numpy()
        scalars = torch.sigmoid(heads["scalars"][0]).cpu().numpy()
    return decode_prediction(notes, waveform, scalars, current_patch, editable_fields)
