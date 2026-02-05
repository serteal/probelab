"""Tests for mask functions."""
import unittest
import torch
from probelab import masks
from probelab.masks import Mask, TokenMetadata

# =============================================================================
# Test Data
# =============================================================================

def _metadata(batch=2, seq=10, roles=None):
  """Create test metadata."""
  token_ids = torch.randint(0, 1000, (batch, seq))
  attention_mask = torch.ones(batch, seq)
  if roles is None:
    # Default: user(1), assistant(2) alternating per token
    role_ids = torch.zeros(batch, seq, dtype=torch.long)
    role_ids[:, ::2] = 1  # user
    role_ids[:, 1::2] = 2  # assistant
  else:
    role_ids = roles
  message_boundaries = torch.zeros(batch, seq, dtype=torch.long)
  return TokenMetadata(
    token_ids=token_ids,
    role_ids=role_ids,
    message_boundaries=message_boundaries,
    attention_mask=attention_mask,
  )

def _dialogues(n=2):
  """Create dummy dialogues."""
  return [None] * n

# =============================================================================
# Basic Masks
# =============================================================================

class TestAllMask(unittest.TestCase):
  def test_selects_all_non_padding(self):
    meta = _metadata()
    m = masks.all()
    result = m(_dialogues(), meta)
    self.assertEqual(result.shape, (2, 10))
    self.assertTrue(result.all())

  def test_respects_attention_mask(self):
    meta = _metadata()
    meta.attention_mask[0, 5:] = 0  # mask out last 5 tokens
    m = masks.all()
    result = m(_dialogues(), meta)
    self.assertEqual(result[0, :5].sum().item(), 5)
    self.assertEqual(result[0, 5:].sum().item(), 0)

class TestNoneMask(unittest.TestCase):
  def test_selects_nothing(self):
    meta = _metadata()
    m = masks.none()
    result = m(_dialogues(), meta)
    self.assertFalse(result.any())

# =============================================================================
# Role Masks
# =============================================================================

class TestRoleMasks(unittest.TestCase):
  def test_assistant_selects_correct_role(self):
    meta = _metadata()
    m = masks.assistant()
    result = m(_dialogues(), meta)
    # assistant=2, should select odd positions
    expected = (meta.role_ids == 2)
    self.assertTrue(torch.equal(result, expected))

  def test_user_selects_correct_role(self):
    meta = _metadata()
    m = masks.user()
    result = m(_dialogues(), meta)
    expected = (meta.role_ids == 1)
    self.assertTrue(torch.equal(result, expected))

  def test_system_selects_correct_role(self):
    roles = torch.zeros(2, 10, dtype=torch.long)
    roles[:, :3] = 0  # system
    roles[:, 3:] = 1  # user
    meta = _metadata(roles=roles)
    m = masks.system()
    result = m(_dialogues(), meta)
    self.assertEqual(result[:, :3].sum().item(), 6)

  def test_role_invalid_raises(self):
    with self.assertRaises(ValueError):
      masks.role("invalid")

# =============================================================================
# Position Masks
# =============================================================================

class TestNthMessage(unittest.TestCase):
  def test_selects_first_message(self):
    meta = _metadata()
    meta.message_boundaries = torch.zeros(2, 10, dtype=torch.long)
    meta.message_boundaries[:, 5:] = 1  # 2 messages
    m = masks.nth_message(0)
    result = m(_dialogues(), meta)
    self.assertTrue(result[:, :5].all())
    self.assertFalse(result[:, 5:].any())

  def test_selects_last_message_negative(self):
    meta = _metadata()
    meta.message_boundaries = torch.zeros(2, 10, dtype=torch.long)
    meta.message_boundaries[:, 5:] = 1
    m = masks.nth_message(-1)
    result = m(_dialogues(), meta)
    self.assertFalse(result[:, :5].any())
    self.assertTrue(result[:, 5:].all())

class TestLastNTokens(unittest.TestCase):
  def test_invalid_n_raises(self):
    with self.assertRaises(ValueError):
      masks.last_n_tokens(0)

class TestFirstNTokens(unittest.TestCase):
  def test_invalid_n_raises(self):
    with self.assertRaises(ValueError):
      masks.first_n_tokens(0)

class TestLastToken(unittest.TestCase):
  def test_returns_mask(self):
    m = masks.last_token()
    self.assertIsInstance(m, Mask)

# =============================================================================
# Composition
# =============================================================================

class TestMaskComposition(unittest.TestCase):
  def test_and(self):
    meta = _metadata()
    m1 = masks.all()
    m2 = masks.assistant()
    combined = m1 & m2
    result = combined(_dialogues(), meta)
    expected = masks.assistant()(_dialogues(), meta)
    self.assertTrue(torch.equal(result, expected))

  def test_or(self):
    meta = _metadata()
    m1 = masks.assistant()
    m2 = masks.user()
    combined = m1 | m2
    result = combined(_dialogues(), meta)
    self.assertTrue(result.all())  # All tokens are either user or assistant

  def test_not(self):
    meta = _metadata()
    m = ~masks.assistant()
    result = m(_dialogues(), meta)
    expected = masks.user()(_dialogues(), meta)  # Not assistant = user
    self.assertTrue(torch.equal(result, expected))

  def test_complex_composition(self):
    meta = _metadata()
    m = (masks.assistant() | masks.user()) & masks.all()
    result = m(_dialogues(), meta)
    self.assertTrue(result.all())

# =============================================================================
# AndMask, OrMask, NotMask
# =============================================================================

class TestExplicitCompositors(unittest.TestCase):
  def test_and_mask(self):
    meta = _metadata()
    m = masks.AndMask(masks.all(), masks.assistant())
    result = m(_dialogues(), meta)
    expected = masks.assistant()(_dialogues(), meta)
    self.assertTrue(torch.equal(result, expected))

  def test_or_mask(self):
    meta = _metadata()
    m = masks.OrMask(masks.assistant(), masks.user())
    result = m(_dialogues(), meta)
    self.assertTrue(result.all())

  def test_not_mask(self):
    meta = _metadata()
    m = masks.NotMask(masks.assistant())
    result = m(_dialogues(), meta)
    expected = masks.user()(_dialogues(), meta)
    self.assertTrue(torch.equal(result, expected))

# =============================================================================
# Mask Properties
# =============================================================================

class TestMaskProperties(unittest.TestCase):
  def test_mask_is_hashable(self):
    m1 = masks.all()
    m2 = masks.all()
    self.assertEqual(hash(m1), hash(m2))

  def test_mask_equality(self):
    m1 = masks.all()
    m2 = masks.all()
    m3 = masks.none()
    self.assertEqual(m1, m2)
    self.assertNotEqual(m1, m3)

  def test_mask_repr(self):
    m = masks.assistant()
    r = repr(m)
    self.assertIn("Mask", r)

  def test_mask_key(self):
    m = masks.assistant()
    self.assertEqual(m.key, ("role", "assistant", True))

# =============================================================================
# Edge Cases
# =============================================================================

class TestMaskEdgeCases(unittest.TestCase):
  def test_empty_batch(self):
    meta = TokenMetadata(
      token_ids=torch.zeros(0, 10, dtype=torch.long),
      role_ids=torch.zeros(0, 10, dtype=torch.long),
      message_boundaries=torch.zeros(0, 10, dtype=torch.long),
      attention_mask=torch.zeros(0, 10),
    )
    m = masks.all()
    result = m([], meta)
    self.assertEqual(result.shape, (0, 10))

  def test_empty_sequence(self):
    meta = TokenMetadata(
      token_ids=torch.zeros(2, 0, dtype=torch.long),
      role_ids=torch.zeros(2, 0, dtype=torch.long),
      message_boundaries=torch.zeros(2, 0, dtype=torch.long),
      attention_mask=torch.zeros(2, 0),
    )
    m = masks.all()
    result = m(_dialogues(), meta)
    self.assertEqual(result.shape, (2, 0))

  def test_all_padding(self):
    meta = _metadata()
    meta.attention_mask = torch.zeros(2, 10)
    m = masks.all()
    result = m(_dialogues(), meta)
    self.assertFalse(result.any())

  def test_single_sample(self):
    meta = _metadata(batch=1, seq=5)
    m = masks.assistant()
    result = m([None], meta)
    self.assertEqual(result.shape, (1, 5))

if __name__ == '__main__':
  unittest.main()
