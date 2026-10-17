import os
import json
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'intents.json')

class TestIntentDataset(unittest.TestCase):
    """Test suite for validating dataset integrity."""

    def test_dataset_exists(self):
        self.assertTrue(os.path.exists(DATASET_PATH), "dataset/intents.json must exist")

    def test_dataset_structure(self):
        with open(DATASET_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.assertIn("intents", data, "Root JSON must contain 'intents' key")
        intents_list = data["intents"]
        self.assertGreaterEqual(len(intents_list), 15, "Dataset must contain at least 15 intents")

        for item in intents_list:
            self.assertIn("intent", item, "Each entry must have an 'intent' field")
            self.assertIn("examples", item, "Each entry must have an 'examples' list")
            self.assertGreaterEqual(len(item["examples"]), 3, f"Intent '{item['intent']}' must have at least 3 examples")

if __name__ == '__main__':
    unittest.main()
