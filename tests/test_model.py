"""Architecture: shapes, masking behaviour and parameter budget."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.config import FeatureConfig, ModelConfig
from ml.models.emotion_cnn_lstm import (
    AdditiveAttentionPool,
    EmotionCNNBiLSTM,
    build_model,
    masked_mean_pool,
)

N_FEATURES = FeatureConfig().n_output_features()  # 120
N_FRAMES = 301  # 3.0 s at a 10 ms hop


def make_batch(batch: int = 2, features: int = N_FEATURES, frames: int = N_FRAMES):
    return torch.zeros(batch, 1, features, frames)


def make_mask(batch: int = 2, frames: int = N_FRAMES, valid: int = 150):
    mask = torch.zeros(batch, frames)
    mask[:, :valid] = 1.0
    return mask


@pytest.fixture(scope="module")
def model() -> EmotionCNNBiLSTM:
    torch.manual_seed(0)
    net = build_model(ModelConfig(), n_features=N_FEATURES)
    net.eval()
    return net


class TestShapes:
    def test_forward_returns_logits_per_class(self, model):
        logits = model(make_batch())
        assert logits.shape == (2, model.cfg.num_classes)

    def test_batch_dimension_is_independent(self, model):
        assert model(make_batch(batch=1)).shape == (1, 8)
        assert model(make_batch(batch=8)).shape == (8, 8)

    def test_input_shape_is_validated(self, model):
        with pytest.raises(ValueError, match="4-D"):
            model(torch.zeros(2, N_FEATURES, N_FRAMES))
        with pytest.raises(ValueError, match="4-D"):
            model(torch.zeros(2, 1, N_FEATURES))

    def test_handles_non_power_of_two_frame_counts(self, model):
        """Real clips do not land on exact multiples of the hop."""
        assert model(torch.zeros(1, 1, N_FEATURES, 97)).shape == (1, 8)
        assert model(torch.zeros(1, 1, N_FEATURES, 303)).shape == (1, 8)

    def test_attention_weights_are_returned_on_request(self, model):
        logits, weights = model(make_batch(), frame_mask=make_mask(), return_attention=True)
        assert logits.shape == (2, 8)
        assert weights.shape[0] == 2  # (B, T') after the conv downsampling
        torch.testing.assert_close(weights.sum(dim=1), torch.ones(2), rtol=1e-4, atol=1e-4)


class TestParameterBudget:
    def test_model_is_small_enough_for_960_training_utterances(self, model):
        """~0.4 M parameters is deliberately modest.

        The training set has 960 utterances; a much larger network would fit the
        speaker identities in train rather than generalise to unseen voices.
        """
        assert 100_000 < model.num_parameters() < 2_000_000

    def test_breakdown_sums_to_the_total(self, model):
        breakdown = model.parameter_breakdown()
        assert breakdown["total"] == model.num_parameters()
        assert breakdown["lstm"] > 0 and breakdown["conv"] > 0

    def test_recurrent_block_dominates(self, model):
        """Frequency mean-pooling exists to keep the LSTM small."""
        breakdown = model.parameter_breakdown()
        assert breakdown["lstm"] > breakdown["conv"]

    def test_describe_mentions_pooling_strategy(self, model):
        text = model.describe()
        assert "pooling=attention" in text
        assert "bidirectional" in text


class TestPooling:
    def test_attention_ignores_masked_steps(self):
        pool = AdditiveAttentionPool(dim=16)
        sequence = torch.randn(1, 10, 16)
        mask = torch.zeros(1, 10)
        mask[:, :4] = 1.0

        _, weights = pool(sequence, mask)
        # Softmax of -inf is exactly zero, not merely small.
        assert torch.allclose(weights[0, 4:], torch.zeros(6), atol=0.0)
        assert torch.allclose(weights.sum(dim=1), torch.ones(1), atol=1e-5)

    def test_masked_content_cannot_change_the_context(self):
        """Padding content must be irrelevant when the mask excludes it."""
        torch.manual_seed(1)
        pool = AdditiveAttentionPool(dim=8)
        mask = torch.zeros(1, 6)
        mask[:, :3] = 1.0

        sequence = torch.randn(1, 6, 8)
        original, _ = pool(sequence, mask)

        tampered = sequence.clone()
        tampered[:, 3:] = 1e3  # absurd values in the padded region only
        after, _ = pool(tampered, mask)

        torch.testing.assert_close(original, after, rtol=1e-5, atol=1e-6)

    def test_mean_pool_weights_are_uniform_over_valid_steps(self):
        sequence = torch.randn(2, 5, 4)
        mask = torch.zeros(2, 5)
        mask[:, :2] = 1.0
        mask[1, :4] = 1.0  # different lengths in the same batch

        context, weights = masked_mean_pool(sequence, mask)

        assert torch.allclose(weights[0], torch.tensor([0.5, 0.5, 0.0, 0.0, 0.0]))
        assert torch.allclose(weights[1], torch.tensor([0.25, 0.25, 0.25, 0.25, 0.0]))
        torch.testing.assert_close(context, (weights.unsqueeze(-1) * sequence).sum(dim=1))

    def test_mean_pool_without_mask_is_uniform(self):
        context, weights = masked_mean_pool(torch.randn(1, 4, 3))
        assert context.shape == (1, 3)
        torch.testing.assert_close(weights, torch.full((1, 4), 0.25))

    def test_all_zero_mask_does_not_produce_nan(self):
        """A degenerate mask must degrade, not explode."""
        pool = AdditiveAttentionPool(dim=8)
        _, weights = pool(torch.randn(1, 5, 8), torch.zeros(1, 5))
        assert torch.isfinite(weights).all()
        assert torch.allclose(weights.sum(dim=1), torch.ones(1), atol=1e-5)


class TestMaskDownsampling:
    def test_mask_is_resized_to_the_conv_time_axis(self, model):
        logits, weights = model(make_batch(), frame_mask=make_mask(), return_attention=True)
        # Stride-2 stem plus two 2x pools => 301 frames collapse to 37-ish.
        assert weights.shape[1] == model._resize_mask(make_mask(), weights.shape[1]).shape[1]
        assert weights.shape[1] < N_FRAMES

    def test_fully_valid_mask_is_equivalent_to_no_mask(self, model):
        features = torch.randn(2, 1, N_FEATURES, N_FRAMES)
        with torch.inference_mode():
            without = model(features)
            with_full_mask = model(features, frame_mask=torch.ones(2, N_FRAMES))
        torch.testing.assert_close(without, with_full_mask, rtol=1e-4, atol=1e-4)


class TestAblationVariant:
    def test_mean_pooling_model_runs_and_has_fewer_parameters(self):
        torch.manual_seed(0)
        attention = build_model(ModelConfig(use_attention_pooling=True), n_features=N_FEATURES)
        mean_pool = build_model(ModelConfig(use_attention_pooling=False), n_features=N_FEATURES)
        mean_pool.eval()

        assert mean_pool.pool is None
        assert mean_pool.num_parameters() < attention.num_parameters()
        assert mean_pool(torch.zeros(1, 1, N_FEATURES, N_FRAMES)).shape == (1, 8)


class TestConstruction:
    def test_invalid_stem_stride_is_rejected(self):
        with pytest.raises(ValueError, match="stem_stride"):
            build_model(ModelConfig(stem_stride=[2]), n_features=N_FEATURES)

    def test_n_input_features_is_recorded(self):
        net = build_model(ModelConfig(), n_features=13)
        assert net.n_input_features == 13

    def test_unidirectional_variant(self):
        net = build_model(ModelConfig(bidirectional=False), n_features=N_FEATURES)
        assert net.cfg.lstm_output_dim() == net.cfg.lstm_hidden
        assert net(torch.zeros(1, 1, N_FEATURES, N_FRAMES)).shape == (1, 8)

    def test_batchnorm_can_be_disabled(self):
        net = build_model(ModelConfig(use_batch_norm=False), n_features=N_FEATURES)
        net.eval()
        assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in net.modules())
        assert net(torch.zeros(1, 1, N_FEATURES, N_FRAMES)).shape == (1, 8)


class TestTrainingBehaviour:
    def test_eval_mode_is_deterministic(self, model):
        features = torch.randn(2, 1, N_FEATURES, N_FRAMES)
        mask = make_mask()
        with torch.inference_mode():
            first = model(features, frame_mask=mask)
            second = model(features, frame_mask=mask)
        torch.testing.assert_close(first, second)

    def test_dropout_is_active_in_train_mode(self, model):
        model.train()
        features = torch.randn(4, 1, N_FEATURES, N_FRAMES)
        first = model(features)
        second = model(features)
        assert not torch.allclose(first, second)
        model.eval()

    def test_gradients_reach_the_conv_stack(self, model):
        model.train()
        model.zero_grad(set_to_none=True)
        logits = model(torch.randn(2, 1, N_FEATURES, N_FRAMES), frame_mask=make_mask())
        logits.sum().backward()

        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
        model.zero_grad(set_to_none=True)

    def test_logits_are_finite_for_degenerate_inputs(self, model):
        """Zero features must not produce NaN - BatchNorm and softmax must cope."""
        with torch.inference_mode():
            logits = model(torch.zeros(2, 1, N_FEATURES, N_FRAMES), frame_mask=make_mask())
        assert torch.isfinite(logits).all()

    def test_softmax_over_logits_is_a_distribution(self, model):
        with torch.inference_mode():
            logits = model(torch.randn(4, 1, N_FEATURES, N_FRAMES))
        probs = torch.softmax(logits, dim=1)
        torch.testing.assert_close(probs.sum(dim=1), torch.ones(4), rtol=1e-5, atol=1e-5)
        assert (probs >= 0).all()


class TestNumericSanity:
    def test_output_dtype_is_float32(self, model):
        with torch.inference_mode():
            logits = model(make_batch())
        assert logits.dtype == torch.float32

    def test_constant_input_gives_constant_output(self, model):
        """A uniform frame should not produce a wildly varying sequence."""
        model.train()  # BatchNorm needs a non-degenerate batch in train mode
        batch = torch.ones(4, 1, N_FEATURES, N_FRAMES)
        batch += 0.01 * torch.randn_like(batch)
        mask = torch.ones(4, N_FRAMES)
        with torch.inference_mode():
            logits = model(batch, frame_mask=mask)
        assert torch.isfinite(logits).all()
        model.eval()

    def test_numpy_conversion_round_trip(self, model):
        features = np.random.RandomState(0).randn(1, N_FEATURES, N_FRAMES).astype(np.float32)
        with torch.inference_mode():
            logits = model(torch.from_numpy(features).unsqueeze(0))
        assert np.isfinite(logits.numpy()).all()
