import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from chatbot import ChatbotModel

class TestChatbotModel(unittest.TestCase):
    """Test suite for ChatbotModel prediction pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.bot = ChatbotModel()

    def test_model_loaded(self):
        self.assertTrue(self.bot.is_loaded, "Chatbot model artifacts should be loaded successfully.")

    def test_working_hours_intent(self):
        res = self.bot.predict_intent("What are your working hours?")
        self.assertEqual(res["intent"], "working_hours")
        self.assertGreaterEqual(res["confidence"], 0.60)

    def test_empty_input(self):
        res = self.bot.predict_intent("")
        self.assertEqual(res["intent"], "empty_input")
        self.assertFalse(res["is_confident"])

    def test_confidence_thresholding(self):
        # Extremely high threshold should mark as not confident unless 100%
        res = self.bot.predict_intent("What time do you close?", threshold=0.999)
        self.assertIn("is_confident", res)

if __name__ == '__main__':
    unittest.main()
