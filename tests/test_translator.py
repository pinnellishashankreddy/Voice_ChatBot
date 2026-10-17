import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import translator

class TestVoiceTranslator(unittest.TestCase):
    """Test suite for 4-Language Voice Translator module."""

    def test_supported_languages_config(self):
        langs = translator.SUPPORTED_LANGUAGES
        self.assertIn("English", langs)
        self.assertIn("Hindi", langs)
        self.assertIn("Tamil", langs)
        self.assertIn("Telugu", langs)

    def test_telugu_to_english_translation(self):
        res = translator.translate("నా పేరు రాహుల్.", "Telugu", "English")
        self.assertTrue(res["success"])
        self.assertIn("rahul", res["translated_text"].lower())

    def test_telugu_to_tamil_translation(self):
        res = translator.translate("నమస్కారం", "Telugu", "Tamil")
        self.assertTrue(res["success"])
        self.assertGreater(len(res["translated_text"]), 0)

    def test_english_to_hindi_translation(self):
        res = translator.translate("Hello, how are you?", "English", "Hindi")
        self.assertTrue(res["success"])
        self.assertGreater(len(res["translated_text"]), 0)

    def test_same_language_protection(self):
        res = translator.translate("Hello World", "English", "English")
        self.assertTrue(res["success"])
        self.assertEqual(res["translated_text"], "Hello World")

if __name__ == '__main__':
    unittest.main()
