"""Offline tests for dataset dialogue builders."""

import unittest

from probelab.datasets.builders import build_from_messages, build_qa, normalize_role


class TestNormalizeRole(unittest.TestCase):
    def test_known_aliases(self):
        self.assertEqual(normalize_role("human"), "user")
        self.assertEqual(normalize_role("gpt"), "assistant")
        self.assertEqual(normalize_role("Doctor"), "assistant")

    def test_passthrough(self):
        self.assertEqual(normalize_role("user"), "user")


class TestBuildFromMessages(unittest.TestCase):
    def test_role_and_content_aliases(self):
        dialogue = build_from_messages(
            [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "yo"}]
        )
        self.assertEqual([m.role.value for m in dialogue], ["user", "assistant"])
        self.assertEqual([m.content for m in dialogue], ["hi", "yo"])


class TestBuildQA(unittest.TestCase):
    def test_default_aliases(self):
        d = build_qa({"question": "Q?", "answer": "A."})
        self.assertEqual([m.role.value for m in d], ["user", "assistant"])
        self.assertEqual(d[0].content, "Q?")
        self.assertEqual(d[1].content, "A.")

    def test_user_only(self):
        d = build_qa({"prompt": "only user"})
        self.assertEqual([m.role.value for m in d], ["user"])

    def test_system_prefix(self):
        d = build_qa(
            {"instruction": "do it", "output": "done", "system": "be nice"},
            system_keys=("system",),
        )
        self.assertEqual([m.role.value for m in d], ["system", "user", "assistant"])

    def test_returns_none_without_user(self):
        self.assertIsNone(build_qa({"answer": "no question here"}))

    def test_first_non_empty_wins(self):
        d = build_qa({"question": "", "prompt": "fallback"})
        self.assertEqual(d[0].content, "fallback")


if __name__ == "__main__":
    unittest.main()
