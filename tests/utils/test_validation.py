"""Tests for probelab.utils.validation module."""

import pytest
import torch

from probelab.processing.activations import Activations, Axis
from probelab.processing.scores import ScoreAxis, Scores
from probelab.utils.validation import check_activations, check_scores


class TestCheckActivations:
    """Tests for check_activations function."""

    @pytest.fixture
    def activations_4d(self):
        """Create 4D activations [layer, batch, seq, hidden]."""
        return Activations.from_tensor(
            torch.randn(2, 4, 10, 32),  # [layer, batch, seq, hidden]
            layer_indices=[0, 1],
        )

    @pytest.fixture
    def activations_3d(self):
        """Create 3D activations [batch, seq, hidden] (single layer, axis removed)."""
        acts_4d = Activations.from_tensor(
            torch.randn(1, 4, 10, 32),
            layer_indices=[0],
        )
        return acts_4d.select(layer=0)  # Remove LAYER axis

    @pytest.fixture
    def activations_no_seq(self):
        """Create activations without SEQ axis."""
        acts = Activations.from_tensor(
            torch.randn(1, 4, 10, 32),
            layer_indices=[0],
        )
        return acts.select(layer=0).pool(dim="sequence", method="mean")

    def test_valid_activations_passes(self, activations_4d):
        """Basic validation passes for valid activations."""
        result = check_activations(activations_4d)
        assert result is activations_4d

    def test_type_check_fails_for_non_activations(self):
        """Raises TypeError for non-Activations input."""
        with pytest.raises(TypeError, match="Expected Activations"):
            check_activations(torch.randn(4, 10, 32))

    def test_type_check_fails_for_scores(self):
        """Raises TypeError when given Scores instead of Activations."""
        scores = Scores.from_sequence_scores(torch.randn(4, 2))
        with pytest.raises(TypeError, match="Expected Activations"):
            check_activations(scores)

    def test_require_layer_passes(self, activations_4d):
        """require_layer=True passes when LAYER axis present."""
        result = check_activations(activations_4d, require_layer=True)
        assert result is activations_4d

    def test_require_layer_fails(self, activations_3d):
        """require_layer=True fails when LAYER axis missing."""
        with pytest.raises(ValueError, match="Expected activations with LAYER axis"):
            check_activations(activations_3d, require_layer=True)

    def test_forbid_layer_passes(self, activations_3d):
        """forbid_layer=True passes when LAYER axis missing."""
        result = check_activations(activations_3d, forbid_layer=True)
        assert result is activations_3d

    def test_forbid_layer_fails(self, activations_4d):
        """forbid_layer=True fails when LAYER axis present."""
        with pytest.raises(ValueError, match="Expected single-layer activations"):
            check_activations(activations_4d, forbid_layer=True)

    def test_forbid_layer_error_message_includes_hint(self, activations_4d):
        """Error message includes helpful hint about SelectLayer."""
        with pytest.raises(ValueError, match="pre.SelectLayer"):
            check_activations(activations_4d, forbid_layer=True)

    def test_require_seq_passes(self, activations_3d):
        """require_seq=True passes when SEQ axis present."""
        result = check_activations(activations_3d, require_seq=True)
        assert result is activations_3d

    def test_require_seq_fails(self, activations_no_seq):
        """require_seq=True fails when SEQ axis missing."""
        with pytest.raises(ValueError, match="Expected activations with SEQ axis"):
            check_activations(activations_no_seq, require_seq=True)

    def test_forbid_seq_passes(self, activations_no_seq):
        """forbid_seq=True passes when SEQ axis missing."""
        result = check_activations(activations_no_seq, forbid_seq=True)
        assert result is activations_no_seq

    def test_forbid_seq_fails(self, activations_3d):
        """forbid_seq=True fails when SEQ axis present."""
        with pytest.raises(ValueError, match="Expected pooled activations without SEQ"):
            check_activations(activations_3d, forbid_seq=True)

    def test_conflicting_layer_requirements_fails(self, activations_4d):
        """Cannot both require and forbid LAYER axis."""
        with pytest.raises(ValueError, match="Cannot both require and forbid"):
            check_activations(activations_4d, require_layer=True, forbid_layer=True)

    def test_conflicting_seq_requirements_fails(self, activations_4d):
        """Cannot both require and forbid SEQ axis."""
        with pytest.raises(ValueError, match="Cannot both require and forbid"):
            check_activations(activations_4d, require_seq=True, forbid_seq=True)

    def test_ensure_finite_passes(self, activations_4d):
        """ensure_finite=True passes for finite values."""
        result = check_activations(activations_4d, ensure_finite=True)
        assert result is activations_4d

    def test_ensure_finite_fails_nan(self):
        """ensure_finite=True fails when NaN present."""
        tensor = torch.randn(1, 4, 10, 32)
        tensor[0, 0, 0, 0] = float("nan")
        acts = Activations.from_tensor(tensor, layer_indices=[0])
        with pytest.raises(ValueError, match="non-finite values"):
            check_activations(acts, ensure_finite=True)

    def test_ensure_finite_fails_inf(self):
        """ensure_finite=True fails when Inf present."""
        tensor = torch.randn(1, 4, 10, 32)
        tensor[0, 0, 0, 0] = float("inf")
        acts = Activations.from_tensor(tensor, layer_indices=[0])
        with pytest.raises(ValueError, match="non-finite values"):
            check_activations(acts, ensure_finite=True)

    def test_ensure_non_empty_passes(self, activations_4d):
        """ensure_non_empty=True passes for non-empty tensor."""
        result = check_activations(activations_4d, ensure_non_empty=True)
        assert result is activations_4d

    def test_estimator_name_in_error(self, activations_4d):
        """Estimator name appears in error messages."""
        with pytest.raises(ValueError, match="MyEstimator:"):
            check_activations(
                activations_4d, forbid_layer=True, estimator_name="MyEstimator"
            )

    def test_multiple_constraints(self, activations_3d):
        """Multiple constraints can be combined."""
        result = check_activations(
            activations_3d,
            forbid_layer=True,
            require_seq=True,
            ensure_finite=True,
            ensure_non_empty=True,
        )
        assert result is activations_3d


class TestCheckScores:
    """Tests for check_scores function."""

    @pytest.fixture
    def scores_seq(self):
        """Create token-level scores [batch, seq, 2]."""
        return Scores.from_token_scores(
            torch.randn(4, 10, 2),
            tokens_per_sample=torch.tensor([10, 8, 6, 10]),
        )

    @pytest.fixture
    def scores_no_seq(self):
        """Create sequence-level scores [batch, 2]."""
        return Scores.from_sequence_scores(torch.randn(4, 2))

    def test_valid_scores_passes(self, scores_seq):
        """Basic validation passes for valid scores."""
        result = check_scores(scores_seq)
        assert result is scores_seq

    def test_type_check_fails_for_non_scores(self):
        """Raises TypeError for non-Scores input."""
        with pytest.raises(TypeError, match="Expected Scores"):
            check_scores(torch.randn(4, 2))

    def test_type_check_fails_for_activations(self):
        """Raises TypeError when given Activations instead of Scores."""
        acts = Activations.from_tensor(torch.randn(4, 10, 32))
        with pytest.raises(TypeError, match="Expected Scores"):
            check_scores(acts)

    def test_require_seq_passes(self, scores_seq):
        """require_seq=True passes when SEQ axis present."""
        result = check_scores(scores_seq, require_seq=True)
        assert result is scores_seq

    def test_require_seq_fails(self, scores_no_seq):
        """require_seq=True fails when SEQ axis missing."""
        with pytest.raises(ValueError, match="Expected token-level scores with SEQ"):
            check_scores(scores_no_seq, require_seq=True)

    def test_forbid_seq_passes(self, scores_no_seq):
        """forbid_seq=True passes when SEQ axis missing."""
        result = check_scores(scores_no_seq, forbid_seq=True)
        assert result is scores_no_seq

    def test_forbid_seq_fails(self, scores_seq):
        """forbid_seq=True fails when SEQ axis present."""
        with pytest.raises(ValueError, match="Expected sequence-level scores"):
            check_scores(scores_seq, forbid_seq=True)

    def test_conflicting_seq_requirements_fails(self, scores_seq):
        """Cannot both require and forbid SEQ axis."""
        with pytest.raises(ValueError, match="Cannot both require and forbid"):
            check_scores(scores_seq, require_seq=True, forbid_seq=True)

    def test_ensure_finite_passes(self, scores_seq):
        """ensure_finite=True passes for finite values."""
        result = check_scores(scores_seq, ensure_finite=True)
        assert result is scores_seq

    def test_ensure_finite_fails_nan(self):
        """ensure_finite=True fails when NaN present."""
        tensor = torch.randn(4, 2)
        tensor[0, 0] = float("nan")
        scores = Scores.from_sequence_scores(tensor)
        with pytest.raises(ValueError, match="non-finite values"):
            check_scores(scores, ensure_finite=True)

    def test_ensure_finite_fails_inf(self):
        """ensure_finite=True fails when Inf present."""
        tensor = torch.randn(4, 2)
        tensor[0, 0] = float("inf")
        scores = Scores.from_sequence_scores(tensor)
        with pytest.raises(ValueError, match="non-finite values"):
            check_scores(scores, ensure_finite=True)

    def test_estimator_name_in_error(self, scores_seq):
        """Estimator name appears in error messages."""
        with pytest.raises(ValueError, match="MyTransform:"):
            check_scores(scores_seq, forbid_seq=True, estimator_name="MyTransform")

    def test_multiple_constraints(self, scores_no_seq):
        """Multiple constraints can be combined."""
        result = check_scores(
            scores_no_seq,
            forbid_seq=True,
            ensure_finite=True,
        )
        assert result is scores_no_seq
