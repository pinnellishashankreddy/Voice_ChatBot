import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from response_engine import ResponseEngine

class TestResponseEngine(unittest.TestCase):
    """Test suite for ResponseEngine."""

    @classmethod
    def setUpClass(cls):
        cls.engine = ResponseEngine()

    def test_valid_intent_response(self):
        response = self.engine.get_response("working_hours", confidence=0.95, is_confident=True)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 5)

    def test_fallback_response_low_confidence(self):
        response = self.engine.get_response("working_hours", confidence=0.30, is_confident=False)
        self.assertIsInstance(response, str)
        self.assertTrue(any(word in response.lower() for word in ["rephrase", "understand", "different way", "catch"]))

    def test_unknown_intent_response(self):
        response = self.engine.get_response("non_existent_intent", confidence=0.90, is_confident=True)
        self.assertIsInstance(response, str)
        self.assertTrue(any(word in response.lower() for word in ["rephrase", "understand", "different way", "catch"]))

if __name__ == '__main__':
    unittest.main()
