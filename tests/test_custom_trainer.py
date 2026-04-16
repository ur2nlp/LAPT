"""Tests for custom_trainer module."""

import math

import pytest
import torch
import torch.nn.functional as F

from src.custom_trainer import floored_per_example_causal_lm_loss


def _log_prob_of_target(logits_row: torch.Tensor, target: int) -> float:
    """Return -log softmax probability assigned to `target` given a logits row."""
    log_probs = F.log_softmax(logits_row.float(), dim=-1)
    return float(-log_probs[target])


class TestFlooredPerExampleLoss:
    """Unit tests for floored_per_example_causal_lm_loss."""

    def test_rejects_floor_below_one(self):
        """
        A per-example floor of zero would divide by zero for empty examples and
        has no sensible interpretation for non-empty ones; reject it at the API.
        """
        logits = torch.zeros(1, 3, 4)
        labels = torch.tensor([[0, 1, 2]])
        with pytest.raises(ValueError, match="per_example_loss_floor"):
            floored_per_example_causal_lm_loss(logits, labels, per_example_loss_floor=0)

    def test_equal_length_examples_match_manual_computation(self):
        """
        With all examples equal length and no floor effect, the loss should equal
        the mean over examples of the per-example mean token CE.
        """
        torch.manual_seed(0)
        batch_size, seq_len, vocab_size = 2, 4, 5
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.tensor([[7, 1, 2, 3], [7, 4, 0, 2]])  # 7 is unused; will be shifted off

        # After the shift, targets are labels[:, 1:]: [[1,2,3],[4,0,2]],
        # predicted from logits[:, :-1, :] (positions 0,1,2).
        expected_per_example = []
        for b in range(batch_size):
            token_losses = [
                _log_prob_of_target(logits[b, t], int(labels[b, t + 1]))
                for t in range(seq_len - 1)
            ]
            expected_per_example.append(sum(token_losses) / len(token_losses))
        expected_loss = sum(expected_per_example) / batch_size

        actual = floored_per_example_causal_lm_loss(
            logits, labels, per_example_loss_floor=1
        )
        assert math.isclose(float(actual), expected_loss, rel_tol=1e-5)

    def test_heterogeneous_lengths_are_equally_weighted(self):
        """
        Without per-example normalization, a 3-token example would contribute 3x
        more gradient than a 1-token example. With k=1, each example contributes
        equally. This is the core property that motivates the feature.
        """
        torch.manual_seed(1)
        batch_size, seq_len, vocab_size = 2, 5, 6
        logits = torch.randn(batch_size, seq_len, vocab_size)
        # Example 0: three valid response tokens (targets at positions 1,2,3 after shift)
        # Example 1: one valid response token (target at position 3 after shift)
        labels = torch.tensor(
            [
                [-100, 2, 3, 4, -100],
                [-100, -100, -100, 1, -100],
            ]
        )

        # Shifted targets and positions:
        # example 0: positions 0,1,2 predict labels 2,3,4 (all valid)
        # example 1: positions 0,1,2 predict labels -100,-100,1 (only position 2 valid)
        example0_token_losses = [
            _log_prob_of_target(logits[0, 0], 2),
            _log_prob_of_target(logits[0, 1], 3),
            _log_prob_of_target(logits[0, 2], 4),
        ]
        example1_token_losses = [_log_prob_of_target(logits[1, 2], 1)]

        example0_loss = sum(example0_token_losses) / 3  # t_i = 3, k = 1
        example1_loss = sum(example1_token_losses) / 1  # t_i = 1, k = 1
        expected_loss = (example0_loss + example1_loss) / 2

        actual = floored_per_example_causal_lm_loss(
            logits, labels, per_example_loss_floor=1
        )
        assert math.isclose(float(actual), expected_loss, rel_tol=1e-5)

    def test_floor_caps_short_example_gradient_weight(self):
        """
        With k larger than any example's valid-token count, every denominator is
        `k`, so the loss equals the mean over examples of (sum of valid token
        losses / k). Verify against a hand-computed value.
        """
        torch.manual_seed(2)
        batch_size, seq_len, vocab_size = 2, 5, 5
        k = 10
        logits = torch.randn(batch_size, seq_len, vocab_size)
        # After labels[:, 1:] shift:
        # example 0: [2, -100, -100, -100]   -> one valid target (position 0)
        # example 1: [-100, 1, 4, -100]      -> two valid targets (positions 1, 2)
        labels = torch.tensor(
            [
                [-100, 2, -100, -100, -100],
                [-100, -100, 1, 4, -100],
            ]
        )

        example0_token_losses = [_log_prob_of_target(logits[0, 0], 2)]
        example1_token_losses = [
            _log_prob_of_target(logits[1, 1], 1),
            _log_prob_of_target(logits[1, 2], 4),
        ]
        example0_loss = sum(example0_token_losses) / k
        example1_loss = sum(example1_token_losses) / k
        expected_loss = (example0_loss + example1_loss) / 2

        actual = floored_per_example_causal_lm_loss(
            logits, labels, per_example_loss_floor=k
        )
        assert math.isclose(float(actual), expected_loss, rel_tol=1e-5)

    def test_floor_inactive_when_all_examples_long_enough(self):
        """
        When every example has `t_i >= k`, the floor has no effect and the loss
        equals the pure per-example mean (same as k=1).
        """
        torch.manual_seed(3)
        batch_size, seq_len, vocab_size = 3, 6, 7
        logits = torch.randn(batch_size, seq_len, vocab_size)
        # Every example has 5 valid shifted targets (positions 0..4 predict labels 1..5)
        labels = torch.tensor(
            [
                [0, 1, 2, 3, 4, 5],
                [0, 2, 4, 1, 3, 5],
                [0, 3, 1, 4, 2, 5],
            ]
        )

        loss_k1 = floored_per_example_causal_lm_loss(
            logits, labels, per_example_loss_floor=1
        )
        loss_k5 = floored_per_example_causal_lm_loss(
            logits, labels, per_example_loss_floor=5
        )
        assert math.isclose(float(loss_k1), float(loss_k5), rel_tol=1e-6)

    def test_examples_with_no_valid_tokens_are_ignored(self):
        """
        If an example has every label masked, including it in the mean would
        either inject a zero loss (pulling the average down) or divide by zero.
        Verify the ignored example does not change the loss vs. a single-example
        batch.
        """
        torch.manual_seed(4)
        seq_len, vocab_size = 4, 5
        logits_single = torch.randn(1, seq_len, vocab_size)
        labels_single = torch.tensor([[-100, 2, 3, -100]])

        logits_pair = torch.cat([logits_single, torch.randn(1, seq_len, vocab_size)], dim=0)
        labels_pair = torch.tensor(
            [
                [-100, 2, 3, -100],
                [-100, -100, -100, -100],  # entirely masked
            ]
        )

        loss_single = floored_per_example_causal_lm_loss(
            logits_single, labels_single, per_example_loss_floor=1
        )
        loss_pair = floored_per_example_causal_lm_loss(
            logits_pair, labels_pair, per_example_loss_floor=1
        )
        assert math.isclose(float(loss_single), float(loss_pair), rel_tol=1e-6)

    def test_gradients_flow_to_logits(self):
        """
        Ensure the loss remains differentiable through to the logits — important
        because the real code path relies on autograd for the backward pass.
        """
        torch.manual_seed(5)
        logits = torch.randn(2, 4, 5, requires_grad=True)
        labels = torch.tensor([[-100, 1, 2, 3], [-100, -100, 4, 0]])

        loss = floored_per_example_causal_lm_loss(logits, labels, per_example_loss_floor=1)
        loss.backward()

        assert logits.grad is not None
        # Ignored positions (including the final position, which is always ignored
        # after the shift) should receive zero gradient.
        assert torch.all(logits.grad[:, -1, :] == 0)
        # Example 1 positions 0 corresponds to label[1]=-100 after shift → zero grad
        assert torch.all(logits.grad[1, 0, :] == 0)
