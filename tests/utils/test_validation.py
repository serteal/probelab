"""Tests for probelab.utils.validation module."""

import pytest
import torch

from probelab.processing.activations import Activations
from probelab.utils.validation import check_activations


class TestCheckActivations:
    """Tests for check_activations function."""

    @pytest.fixture
    def activations_4d(self):
        """Create 4D activations [batch, layer, seq, hidden] via from_padded."""
        return Activations.from_padded(
            data=torch.randn(4, 2, 10, 32),
            detection_mask=torch.ones(4, 10),
            dims="blsh",
            layers=(0, 1),
        )

    @pytest.fixture
    def activations_3d(self):
        """Create 3D activations [batch, seq, hidden] (single layer, axis removed)."""
        acts_4d = Activations.from_padded(
            data=torch.randn(4, 1, 10, 32),
            detection_mask=torch.ones(4, 10),
            dims="blsh",
            layers=(0,),
        )
        return acts_4d.select_layers(0)  # Remove LAYER axis

    @pytest.fixture
    def activations_no_seq(self):
        """Create activations without SEQ axis."""
        acts = Activations.from_padded(
            data=torch.randn(4, 10, 32),
            detection_mask=torch.ones(4, 10),
            dims="bsh",
        )
        return acts.mean_pool()

    def test_valid_activations_passes(self, activations_4d):
        """Basic validation passes for valid activations."""
        result = check_activations(activations_4d)
        assert result is activations_4d

    def test_type_check_fails_for_non_activations(self):
        """Raises TypeError for non-Activations input."""
        with pytest.raises(TypeError, match="Expected Activations"):
            check_activations(torch.randn(4, 10, 32))

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
        with pytest.raises(ValueError, match="select_layers\\(layer_idx\\)"):
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
        tensor = torch.randn(4, 2, 10, 32)
        tensor[0, 0, 0, 0] = float("nan")
        acts = Activations.from_padded(
            data=tensor, detection_mask=torch.ones(4, 10),
            dims="blsh", layers=(0, 1),
        )
        with pytest.raises(ValueError, match="non-finite values"):
            check_activations(acts, ensure_finite=True)

    def test_ensure_finite_fails_inf(self):
        """ensure_finite=True fails when Inf present."""
        tensor = torch.randn(4, 2, 10, 32)
        tensor[0, 0, 0, 0] = float("inf")
        acts = Activations.from_padded(
            data=tensor, detection_mask=torch.ones(4, 10),
            dims="blsh", layers=(0, 1),
        )
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
