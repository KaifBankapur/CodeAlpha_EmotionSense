"""The emotion classifier: a 2-D CNN over MFCCs followed by a BiLSTM.

Shape of the problem
--------------------
MFCC features form a ``(n_features, n_frames)`` time-frequency matrix - a small
image whose x axis is time and whose y axis is cepstral coefficient. Two very
different kinds of structure live in it:

* **Local, spatial** patterns - harmonics, formant movement, bursts of energy.
  These are small translations-invariant motifs: exactly what a convolution is
  for. Treating them with an RNN would force the network to rediscover them one
  time step at a time.
* **Long-range, sequential** structure - the trajectory of the voice over the
  whole utterance (rising pitch for surprise, a sagging energy contour for
  sadness). These span hundreds of milliseconds, far beyond any convolution
  receptive field.

So the architecture splits the two jobs: convolutions read the local
time-frequency texture, the LSTM reads the sequence those convolutions have
summarised, and attention pooling picks out which moments actually mattered for
this particular utterance.

    (B, 1, F, T)
      -> 3x [Conv2d -> BN -> ReLU -> MaxPool(2,2) -> Dropout]
      -> (B, 128, F/8, T/8)
      -> mean over the frequency axis      -> (B, T/8, 128)
      -> Dropout
      -> BiLSTM(128 -> 128 per direction)  -> (B, T/8, 256)
      -> masked additive attention pooling -> (B, 256)
      -> Dropout -> Linear(256 -> 8)

Two deliberate choices worth defending in an interview:

**Frequency mean-pooling before the LSTM.** Flattening ``128 x 15 = 1920``
values per time step into the LSTM would work, but it also multiplies the
recurrent parameter count by eight for no measurable gain - the convolution
stack has already extracted frequency-specific evidence, so the LSTM mainly
needs to reason about *when* things happened. Averaging keeps the recurrent
parameters small, which matters on a 960-utterance training set.

**Masked attention pooling rather than last-timestep or plain mean.** Clips
shorter than the 3 s analysis window are zero-padded. A last-step head would
read pure silence for a large share of the training set; an unmasked mean
dilutes the signal with padding in proportion to how short the clip was.
The mask from :class:`ml.data.dataset.EmotionDataset` removes both problems.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import nn

from ml.config import ModelConfig

logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    """Conv2d -> BatchNorm -> ReLU -> MaxPool -> Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        dropout: float = 0.15,
        use_batch_norm: bool = True,
        pool: tuple[int, int] = (2, 2),
        stride: tuple[int, int] = (1, 1),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            # bias is redundant (and harmful) next to BatchNorm - it is folded
            # into the running statistics during training.
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=not use_batch_norm,
            )
        ]
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers += [
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool, stride=pool),
            nn.Dropout2d(dropout),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AdditiveAttentionPool(nn.Module):
    """Bahdanau-style (additive) attention pooling over time steps.

    Learns a query-free scoring function: each time step gets a score from a
    shared MLP, scores are masked to the valid region, softmaxed, and used to
    build a weighted sum. Intuitively the network learns "for anger, weight the
    high-energy frames; for sadness, weight the quiet trailing frames".

    Args:
        dim: feature width of the sequence, i.e. ``2 * hidden`` when bidirectional.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        hidden = max(8, dim // 2)
        self.project = nn.Linear(dim, hidden)
        self.score = nn.Linear(hidden, 1, bias=False)

    def forward(
        self, sequence: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:
            sequence: ``(B, T, D)``
            mask: ``(B, T)``, 1 for valid steps.

        Returns:
            ``(context, weights)`` with shapes ``(B, D)`` and ``(B, T)``.
        """
        scores = self.score(torch.tanh(self.project(sequence)))  # (B, T, 1)

        if mask is not None:
            # A row of all -inf would make softmax produce NaN, so the first
            # step is always kept valid as a backstop.
            safe_mask = mask.clone()
            safe_mask[:, 0] = 1.0
            scores = scores.masked_fill(safe_mask.unsqueeze(-1) < 0.5, float("-inf"))

        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights * sequence, dim=1)
        return context, weights.squeeze(-1)


def masked_mean_pool(
    sequence: torch.Tensor, mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Padding-aware mean pooling - the ablation counterpart of attention.

    Returns:
        ``(context, weights)`` so both pooling strategies share one interface.
    """
    if mask is None:
        context = sequence.mean(dim=1)
        uniform = torch.full(
            (sequence.size(0), sequence.size(1)),
            1.0 / sequence.size(1),
            device=sequence.device,
            dtype=sequence.dtype,
        )
        return context, uniform

    weights = mask / mask.sum(dim=1, keepdim=True).clamp(min=1e-6)
    context = torch.sum(weights.unsqueeze(-1) * sequence, dim=1)
    return context, weights


class EmotionCNNBiLSTM(nn.Module):
    """CNN + BiLSTM speech-emotion classifier.

    Args:
        cfg: architecture hyper-parameters.
        n_features: number of input feature planes (channels). Inferred from the
            saved :class:`~ml.config.FeatureConfig` at load time - never guessed.

    Forward signature:
        ``forward(features, frame_mask=None, return_attention=False) -> logits``
    """

    def __init__(self, cfg: ModelConfig | None = None, n_features: int = 120) -> None:
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.n_input_features = int(n_features)

        blocks = []
        in_channels = 1
        stem_stride = tuple(self.cfg.stem_stride or (1, 1))
        if len(stem_stride) != 2:
            raise ValueError(f"stem_stride must have 2 entries, got {stem_stride!r}.")
        for index, out_channels in enumerate(self.cfg.channels):
            blocks.append(
                ConvBlock(
                    in_channels,
                    out_channels,
                    dropout=self.cfg.conv_dropout,
                    use_batch_norm=self.cfg.use_batch_norm,
                    # Only the first block strides; the rest downsample by
                    # max-pooling so overlapping evidence is preserved.
                    stride=stem_stride if index == 0 else (1, 1),
                )
            )
            in_channels = out_channels
        self.conv = nn.Sequential(*blocks)
        self.sequence_dropout = nn.Dropout(self.cfg.head_dropout)

        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=self.cfg.lstm_hidden,
            num_layers=self.cfg.lstm_layers,
            batch_first=True,
            bidirectional=self.cfg.bidirectional,
            dropout=self.cfg.lstm_dropout if self.cfg.lstm_layers > 1 else 0.0,
        )

        sequence_dim = self.cfg.lstm_output_dim()
        self.pool = AdditiveAttentionPool(sequence_dim) if self.cfg.use_attention_pooling else None

        self.head_dropout = nn.Dropout(self.cfg.head_dropout)
        self.classifier = nn.Linear(sequence_dim, self.cfg.num_classes)

        self._init_weights()

    # -- setup ------------------------------------------------------------
    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, parameter in module.named_parameters():
                    if "weight" in name:
                        nn.init.xavier_uniform_(parameter)
                    elif "bias" in name:
                        nn.init.zeros_(parameter)
                        # Forget-gate bias of 1 is the standard trick: it starts
                        # the cell remembering rather than erasing, which speeds
                        # up early training considerably.
                        hidden = module.hidden_size
                        parameter.data[hidden : 2 * hidden] = 1.0

    # -- mask handling ----------------------------------------------------
    @staticmethod
    def _resize_mask(frame_mask: torch.Tensor, target_len: int) -> torch.Tensor:
        """Downsample a per-frame validity mask to the conv stack's time axis.

        ``adaptive_max_pool1d`` keeps a pooled bin valid if *any* frame inside it
        was valid, which is the semantics we want: pooling over a window should
        not discard the one real frame it contains.
        """
        mask = frame_mask.unsqueeze(1)  # (B, 1, T)
        return F.adaptive_max_pool1d(mask, target_len).squeeze(1)

    # -- forward ----------------------------------------------------------
    def forward(
        self,
        features: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Args:
            features: ``(B, 1, n_features, n_frames)``.
            frame_mask: ``(B, n_frames)``, 1 = real speech, 0 = padding.

        Returns:
            Raw logits ``(B, num_classes)``. Softmax is applied by the caller -
            keeping it out of the module means ``CrossEntropyLoss`` can use its
            numerically stable log-softmax path.
        """
        if features.dim() != 4:
            raise ValueError(
                f"Expected a 4-D input (B, 1, features, frames), got {tuple(features.shape)}."
            )

        x = self.conv(features)  # (B, C, F', T')
        x = x.mean(dim=2)  # frequency pooling -> (B, C, T')
        x = x.transpose(1, 2)  # (B, T', C) for the LSTM

        mask = self._resize_mask(frame_mask, x.size(1)) if frame_mask is not None else None
        x = self.sequence_dropout(x)

        sequence, _ = self.lstm(x)  # (B, T', 2H)

        if self.pool is not None:
            context, weights = self.pool(sequence, mask)
        else:
            context, weights = masked_mean_pool(sequence, mask)

        logits = self.classifier(self.head_dropout(context))
        if return_attention:
            return logits, weights
        return logits

    # -- introspection ----------------------------------------------------
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_breakdown(self) -> dict[str, int]:
        breakdown = {
            "conv": sum(p.numel() for p in self.conv.parameters()),
            "lstm": sum(p.numel() for p in self.lstm.parameters()),
            "pool": (
                sum(p.numel() for p in self.pool.parameters()) if self.pool is not None else 0
            ),
            "classifier": sum(p.numel() for p in self.classifier.parameters()),
        }
        breakdown["total"] = sum(breakdown.values())
        return breakdown

    def describe(self) -> str:
        pooling = "attention" if self.pool is not None else "masked-mean"
        direction = "bidirectional" if self.cfg.bidirectional else "unidirectional"
        return (
            f"EmotionCNNBiLSTM(input_features={self.n_input_features}, "
            f"conv_channels={self.cfg.channels}, "
            f"lstm_hidden={self.cfg.lstm_hidden} ({direction}), "
            f"pooling={pooling}, classes={self.cfg.num_classes}, "
            f"params={self.num_parameters():,})"
        )


def build_model(cfg: ModelConfig | None = None, n_features: int = 120) -> EmotionCNNBiLSTM:
    """Factory used by both the trainer and the inference loader."""
    return EmotionCNNBiLSTM(cfg or ModelConfig(), n_features=n_features)


__all__ = [
    "AdditiveAttentionPool",
    "ConvBlock",
    "EmotionCNNBiLSTM",
    "build_model",
    "masked_mean_pool",
]
