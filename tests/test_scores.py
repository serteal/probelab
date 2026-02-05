"""Tests for Scores container."""
import unittest
import torch
from probelab.processing.scores import Scores, ScoreAxis

class TestScoresConstruction(unittest.TestCase):
  def test_from_sequence_scores(self):
    t = torch.rand(4, 2)
    s = Scores.from_sequence_scores(t)
    self.assertEqual(s.shape, (4, 2))
    self.assertEqual(s.axes, (ScoreAxis.BATCH, ScoreAxis.CLASS))

  def test_from_sequence_scores_wrong_shape(self):
    t = torch.rand(4, 8, 2)
    with self.assertRaises(ValueError):
      Scores.from_sequence_scores(t)

  def test_from_token_scores_3d(self):
    t = torch.rand(4, 8, 2)
    tps = torch.tensor([8, 8, 8, 8])
    s = Scores.from_token_scores(t, tps)
    self.assertEqual(s.shape, (4, 8, 2))
    self.assertEqual(s.axes, (ScoreAxis.BATCH, ScoreAxis.SEQ, ScoreAxis.CLASS))

  def test_from_token_scores_2d_flattened(self):
    # 4 samples with [3, 2, 4, 1] tokens = 10 total
    t = torch.rand(10, 2)
    tps = torch.tensor([3, 2, 4, 1])
    s = Scores.from_token_scores(t, tps)
    self.assertEqual(s.shape, (4, 4, 2))  # max_seq = 4
    self.assertEqual(s.axes, (ScoreAxis.BATCH, ScoreAxis.SEQ, ScoreAxis.CLASS))

  def test_batch_indices(self):
    t = torch.rand(4, 2)
    s = Scores.from_sequence_scores(t, batch_indices=[10, 20, 30, 40])
    self.assertEqual(s.batch_indices, [10, 20, 30, 40])

  def test_class_axis_must_be_last(self):
    t = torch.rand(2, 4)
    with self.assertRaises(ValueError):
      Scores(t, axes=(ScoreAxis.CLASS, ScoreAxis.BATCH))

class TestScoresAxes(unittest.TestCase):
  def test_has_axis_batch(self):
    s = Scores.from_sequence_scores(torch.rand(4, 2))
    self.assertTrue(s.has_axis(ScoreAxis.BATCH))

  def test_has_axis_class(self):
    s = Scores.from_sequence_scores(torch.rand(4, 2))
    self.assertTrue(s.has_axis(ScoreAxis.CLASS))

  def test_has_axis_seq_token_level(self):
    s = Scores.from_token_scores(torch.rand(4, 8, 2), torch.tensor([8, 8, 8, 8]))
    self.assertTrue(s.has_axis(ScoreAxis.SEQ))

  def test_no_seq_axis_sequence_level(self):
    s = Scores.from_sequence_scores(torch.rand(4, 2))
    self.assertFalse(s.has_axis(ScoreAxis.SEQ))

class TestScoresPool(unittest.TestCase):
  def test_pool_mean_removes_seq(self):
    t = torch.rand(4, 8, 2)
    tps = torch.tensor([8, 8, 8, 8])
    s = Scores.from_token_scores(t, tps)
    p = s.pool("sequence", "mean")
    self.assertEqual(p.shape, (4, 2))
    self.assertFalse(p.has_axis(ScoreAxis.SEQ))

  def test_pool_max_removes_seq(self):
    t = torch.rand(4, 8, 2)
    tps = torch.tensor([8, 8, 8, 8])
    s = Scores.from_token_scores(t, tps)
    p = s.pool("sequence", "max")
    self.assertEqual(p.shape, (4, 2))

  def test_pool_last_token(self):
    t = torch.arange(64).reshape(4, 8, 2).float()
    tps = torch.tensor([3, 5, 2, 8])
    s = Scores.from_token_scores(t, tps)
    p = s.pool("sequence", "last_token")
    self.assertEqual(p.shape, (4, 2))
    # Check last valid token is selected
    self.assertTrue(torch.equal(p.scores[0], t[0, 2]))  # 3 tokens -> index 2
    self.assertTrue(torch.equal(p.scores[1], t[1, 4]))  # 5 tokens -> index 4
    self.assertTrue(torch.equal(p.scores[2], t[2, 1]))  # 2 tokens -> index 1
    self.assertTrue(torch.equal(p.scores[3], t[3, 7]))  # 8 tokens -> index 7

  def test_pool_mean_variable_length(self):
    t = torch.ones(4, 8, 2)
    tps = torch.tensor([2, 4, 6, 8])  # variable lengths
    s = Scores.from_token_scores(t, tps)
    p = s.pool("sequence", "mean")
    # Mean of 1s should be 1
    self.assertTrue(torch.allclose(p.scores, torch.ones(4, 2)))

  def test_pool_on_sequence_level_is_noop(self):
    t = torch.rand(4, 2)
    s = Scores.from_sequence_scores(t)
    p = s.pool("sequence", "mean")
    self.assertTrue(torch.equal(p.scores, t))

  def test_pool_preserves_batch_indices(self):
    t = torch.rand(4, 8, 2)
    tps = torch.tensor([8, 8, 8, 8])
    s = Scores.from_token_scores(t, tps, batch_indices=[5, 10, 15, 20])
    p = s.pool("sequence", "mean")
    self.assertEqual(p.batch_indices, [5, 10, 15, 20])

class TestScoresEMA(unittest.TestCase):
  def test_ema_reduces_to_sequence(self):
    t = torch.rand(4, 8, 2)
    tps = torch.tensor([8, 8, 8, 8])
    s = Scores.from_token_scores(t, tps)
    e = s.ema(alpha=0.5)
    self.assertEqual(e.shape, (4, 2))
    self.assertFalse(e.has_axis(ScoreAxis.SEQ))

  def test_ema_alpha_bounds(self):
    t = torch.rand(4, 8, 2)
    tps = torch.tensor([8, 8, 8, 8])
    s = Scores.from_token_scores(t, tps)
    with self.assertRaises(ValueError):
      s.ema(alpha=0.0)
    with self.assertRaises(ValueError):
      s.ema(alpha=1.5)

  def test_ema_alpha_1_is_max(self):
    t = torch.rand(4, 8, 2)
    tps = torch.tensor([8, 8, 8, 8])
    s = Scores.from_token_scores(t, tps)
    e = s.ema(alpha=1.0)
    m = s.pool("sequence", "max")
    # With alpha=1, EMA at each step equals current value, so max(EMA) = max(scores)
    self.assertTrue(torch.allclose(e.scores[:, 1], m.scores[:, 1]))

  def test_ema_on_sequence_level_is_noop(self):
    t = torch.rand(4, 2)
    s = Scores.from_sequence_scores(t)
    e = s.ema(alpha=0.5)
    self.assertTrue(torch.equal(e.scores, t))

class TestScoresRolling(unittest.TestCase):
  def test_rolling_reduces_to_sequence(self):
    t = torch.rand(4, 8, 2)
    tps = torch.tensor([8, 8, 8, 8])
    s = Scores.from_token_scores(t, tps)
    r = s.rolling(window_size=3)
    self.assertEqual(r.shape, (4, 2))
    self.assertFalse(r.has_axis(ScoreAxis.SEQ))

  def test_rolling_window_size_bounds(self):
    t = torch.rand(4, 8, 2)
    tps = torch.tensor([8, 8, 8, 8])
    s = Scores.from_token_scores(t, tps)
    with self.assertRaises(ValueError):
      s.rolling(window_size=0)

  def test_rolling_window_1_is_max(self):
    t = torch.rand(4, 8, 2)
    tps = torch.tensor([8, 8, 8, 8])
    s = Scores.from_token_scores(t, tps)
    r = s.rolling(window_size=1)
    m = s.pool("sequence", "max")
    self.assertTrue(torch.allclose(r.scores[:, 1], m.scores[:, 1]))

  def test_rolling_on_sequence_level_is_noop(self):
    t = torch.rand(4, 2)
    s = Scores.from_sequence_scores(t)
    r = s.rolling(window_size=3)
    self.assertTrue(torch.equal(r.scores, t))

class TestScoresDevice(unittest.TestCase):
  def test_to_cpu(self):
    t = torch.rand(4, 2)
    s = Scores.from_sequence_scores(t)
    c = s.to("cpu")
    self.assertEqual(c.device.type, "cpu")

  def test_to_preserves_data(self):
    t = torch.rand(4, 2)
    s = Scores.from_sequence_scores(t)
    c = s.to("cpu")
    self.assertTrue(torch.equal(c.scores, t))

  def test_to_moves_tokens_per_sample(self):
    t = torch.rand(4, 8, 2)
    tps = torch.tensor([8, 8, 8, 8])
    s = Scores.from_token_scores(t, tps)
    c = s.to("cpu")
    self.assertEqual(c.tokens_per_sample.device.type, "cpu")

class TestScoresProperties(unittest.TestCase):
  def test_batch_size(self):
    s = Scores.from_sequence_scores(torch.rand(5, 2))
    self.assertEqual(s.batch_size, 5)

  def test_shape(self):
    s = Scores.from_token_scores(torch.rand(4, 8, 2), torch.tensor([8, 8, 8, 8]))
    self.assertEqual(s.shape, (4, 8, 2))

  def test_repr(self):
    s = Scores.from_sequence_scores(torch.rand(4, 2))
    r = repr(s)
    self.assertIn("Scores", r)
    self.assertIn("BATCH", r)
    self.assertIn("CLASS", r)

class TestScoresEdgeCases(unittest.TestCase):
  def test_single_sample(self):
    s = Scores.from_sequence_scores(torch.rand(1, 2))
    self.assertEqual(s.batch_size, 1)

  def test_single_token(self):
    t = torch.rand(4, 1, 2)
    tps = torch.tensor([1, 1, 1, 1])
    s = Scores.from_token_scores(t, tps)
    p = s.pool("sequence", "mean")
    self.assertEqual(p.shape, (4, 2))

  def test_variable_length_with_zeros(self):
    # Some samples have 0 tokens
    t = torch.rand(4, 8, 2)
    tps = torch.tensor([4, 0, 6, 0])
    s = Scores.from_token_scores(t, tps)
    p = s.pool("sequence", "mean")
    self.assertEqual(p.shape, (4, 2))
    # Samples with 0 tokens should get 0 probability
    self.assertTrue(torch.allclose(p.scores[1], torch.tensor([0.5, 0.5]), atol=0.5))

if __name__ == '__main__':
  unittest.main()
