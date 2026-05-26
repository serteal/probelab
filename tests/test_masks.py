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

def _text_metadata(text, *, token_ids=None, attention_mask=None, special_ids=None):
  """Create one-token-per-character metadata for text mask tests."""
  seq = len(text)
  if token_ids is None:
    token_ids = torch.arange(seq, dtype=torch.long).unsqueeze(0)
  elif token_ids.ndim == 1:
    token_ids = token_ids.unsqueeze(0)
  if attention_mask is None:
    attention_mask = torch.ones(1, seq, dtype=torch.bool)

  def char_to_token(batch_idx, char_pos):
    if batch_idx != 0 or char_pos < 0 or char_pos >= seq:
      return None
    return char_pos

  return TokenMetadata(
    token_ids=token_ids,
    role_ids=torch.ones(1, seq, dtype=torch.long),
    message_boundaries=torch.zeros(1, seq, dtype=torch.long),
    attention_mask=attention_mask,
    formatted_texts=(text,),
    char_to_token=char_to_token,
    special_token_ids=special_ids,
  )

def _expected(seq, positions):
  out = torch.zeros(1, seq, dtype=torch.bool)
  out[0, positions] = True
  return out

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

  def test_uses_message_boundaries_not_role_changes(self):
    meta = _metadata(batch=1, seq=6)
    meta.role_ids = torch.ones(1, 6, dtype=torch.long)
    meta.message_boundaries = torch.tensor([[0, 0, 1, 1, 1, 2]])

    result = masks.last_n_tokens(2)([None], meta)

    expected = torch.tensor([[True, True, False, True, True, True]])
    self.assertTrue(torch.equal(result, expected))

  def test_respects_attention_mask_when_selecting_last_tokens(self):
    meta = _metadata(batch=1, seq=5)
    meta.message_boundaries = torch.tensor([[0, 0, 0, 0, 0]])
    meta.attention_mask = torch.tensor([[True, True, False, True, False]])

    result = masks.last_n_tokens(2)([None], meta)

    expected = torch.tensor([[False, True, False, True, False]])
    self.assertTrue(torch.equal(result, expected))

class TestFirstNTokens(unittest.TestCase):
  def test_invalid_n_raises(self):
    with self.assertRaises(ValueError):
      masks.first_n_tokens(0)

  def test_uses_message_boundaries_not_role_changes(self):
    meta = _metadata(batch=1, seq=6)
    meta.role_ids = torch.ones(1, 6, dtype=torch.long)
    meta.message_boundaries = torch.tensor([[0, 0, 1, 1, 1, 2]])

    result = masks.first_n_tokens(2)([None], meta)

    expected = torch.tensor([[True, True, True, True, False, True]])
    self.assertTrue(torch.equal(result, expected))

class TestLastToken(unittest.TestCase):
  def test_selects_one_last_token_per_message(self):
    meta = _metadata(batch=1, seq=4)
    meta.role_ids = torch.ones(1, 4, dtype=torch.long)
    meta.message_boundaries = torch.tensor([[0, 0, 1, 1]])

    result = masks.last_token()([None], meta)

    expected = torch.tensor([[False, True, False, True]])
    self.assertTrue(torch.equal(result, expected))

class TestTextMasks(unittest.TestCase):
  def test_contains_selects_exact_text_span_case_insensitive(self):
    meta = _text_metadata("alpha beta gamma")

    result = masks.contains("BETA")([None], meta)

    self.assertTrue(torch.equal(result, _expected(len("alpha beta gamma"), range(6, 10))))

  def test_regex_selects_matching_span(self):
    meta = _text_metadata("alpha beta gamma")

    result = masks.regex(r"b.ta")([None], meta)

    self.assertTrue(torch.equal(result, _expected(len("alpha beta gamma"), range(6, 10))))

  def test_between_inclusive_and_exclusive(self):
    text = "pre <think>abc</think> post"
    meta = _text_metadata(text)

    inclusive = masks.thinking()([None], meta)
    exclusive = masks.thinking(inclusive=False)([None], meta)

    self.assertTrue(torch.equal(inclusive, _expected(len(text), range(4, 22))))
    self.assertTrue(torch.equal(exclusive, _expected(len(text), range(11, 14))))

  def test_after_and_before_bound_text_regions(self):
    text = "alpha beta gamma"
    meta = _text_metadata(text)

    after = masks.after("alpha ")([None], meta)
    before = masks.before(" gamma")([None], meta)

    self.assertTrue(torch.equal(after, _expected(len(text), range(6, len(text)))))
    self.assertTrue(torch.equal(before, _expected(len(text), range(0, 10))))

  def test_padding_expands_each_contiguous_region(self):
    meta = _text_metadata("abcdef")
    base = Mask(
      lambda _d, _m: torch.tensor([[False, False, True, True, False, False]]),
      ("base",),
    )

    result = masks.padding(base, before=1, after=1)([None], meta)

    self.assertTrue(torch.equal(result, torch.tensor([[False, True, True, True, True, False]])))

  def test_special_tokens_uses_metadata_ids_and_attention_mask(self):
    token_ids = torch.tensor([10, 20, 30, 20, 40])
    attention_mask = torch.tensor([[True, True, True, False, True]])
    meta = _text_metadata(
      "abcde",
      token_ids=token_ids,
      attention_mask=attention_mask,
      special_ids={20, 40},
    )

    result = masks.special_tokens()([None], meta)

    self.assertTrue(torch.equal(result, torch.tensor([[False, True, False, False, True]])))

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
# Mask Operators (&, |, ~)
# =============================================================================

class TestMaskOperators(unittest.TestCase):
  def test_and_operator(self):
    meta = _metadata()
    m = masks.all() & masks.assistant()
    result = m(_dialogues(), meta)
    expected = masks.assistant()(_dialogues(), meta)
    self.assertTrue(torch.equal(result, expected))

  def test_or_operator(self):
    meta = _metadata()
    m = masks.assistant() | masks.user()
    result = m(_dialogues(), meta)
    self.assertTrue(result.all())

  def test_not_operator(self):
    meta = _metadata()
    m = ~masks.assistant()
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

# =============================================================================
# Thinking Mask
# =============================================================================

class TestThinkingMask(unittest.TestCase):
  def test_correct_key_default(self):
    m = masks.thinking()
    self.assertEqual(m.key, ("between", "<think>", "</think>", True))

  def test_custom_delimiters(self):
    m = masks.thinking(start="<thinking>", end="</thinking>")
    self.assertEqual(m.key, ("between", "<thinking>", "</thinking>", True))

  def test_exclusive_mode(self):
    m = masks.thinking(inclusive=False)
    self.assertEqual(m.key, ("between", "<think>", "</think>", False))

if __name__ == '__main__':
  unittest.main()
