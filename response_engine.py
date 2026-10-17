import os
import json
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESPONSES_PATH = os.path.join(BASE_DIR, 'data', 'responses.json')

class ResponseEngine:
    """Manages response retrieval for intent predictions."""
    
    def __init__(self):
        self.responses = {}
        self._load_responses()

    def _load_responses(self):
        """Load intent response mappings from data/responses.json."""
        try:
            if os.path.exists(RESPONSES_PATH):
                with open(RESPONSES_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.responses = data.get('responses', {})
            else:
                print(f"[Warning] Responses file missing at {RESPONSES_PATH}")
        except Exception as e:
            print(f"[ResponseEngine] Error loading responses: {e}")

    def get_response(self, intent: str, confidence: float = 1.0, is_confident: bool = True):
        """
        Selects an appropriate response for the predicted intent.
        Returns fallback response if low confidence or unknown intent.
        """
        if not is_confident or intent == "unknown" or intent not in self.responses:
            fallback_options = self.responses.get(
                "fallback",
                ["I'm not completely sure I understood your question. Could you please rephrase it?"]
            )
            return random.choice(fallback_options)

        options = self.responses.get(intent, [])
        if not options:
            fallback_options = self.responses.get(
                "fallback",
                ["I'm not completely sure I understood your question. Could you please rephrase it?"]
            )
            return random.choice(fallback_options)

        return random.choice(options)
