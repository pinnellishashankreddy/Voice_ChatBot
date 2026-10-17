import os
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'chatbot_model.keras')
TOKENIZER_PATH = os.path.join(BASE_DIR, 'model', 'tokenizer.pkl')
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, 'model', 'label_encoder.pkl')
METADATA_PATH = os.path.join(BASE_DIR, 'model', 'model_metadata.json')

class ChatbotModel:
    """Deep Learning Intent Classification Chatbot Engine using BiLSTM."""
    
    def __init__(self, confidence_threshold=0.60):
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.metadata = None
        self.max_len = 20
        self.is_loaded = False
        self._load_artifacts()

    def _load_artifacts(self):
        """Load Keras model, Tokenizer, Label Encoder, and Metadata."""
        try:
            if not os.path.exists(MODEL_PATH):
                print(f"[Warning] Model file not found at {MODEL_PATH}")
                self.is_loaded = False
                return

            self.model = tf.keras.models.load_model(MODEL_PATH)
            
            with open(TOKENIZER_PATH, 'rb') as f:
                self.tokenizer = pickle.load(f)

            with open(LABEL_ENCODER_PATH, 'rb') as f:
                self.label_encoder = pickle.load(f)

            if os.path.exists(METADATA_PATH):
                with open(METADATA_PATH, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                    self.max_len = self.metadata.get('max_len', 20)
                    if 'default_confidence_threshold' in self.metadata:
                        self.confidence_threshold = self.metadata['default_confidence_threshold']

            self.is_loaded = True
            print("[ChatbotModel] Model and artifacts loaded successfully.")
        except Exception as e:
            print(f"[ChatbotModel] Error loading model artifacts: {e}")
            self.is_loaded = False

    def predict_intent(self, text: str, threshold: float = None):
        """
        Preprocess input text, pass through BiLSTM model, and return predicted intent & confidence.
        """
        if not text or not text.strip():
            return {
                "intent": "empty_input",
                "confidence": 0.0,
                "is_confident": False,
                "all_probabilities": {}
            }

        if not self.is_loaded:
            # Re-attempt loading
            self._load_artifacts()
            if not self.is_loaded:
                return {
                    "intent": "model_error",
                    "confidence": 0.0,
                    "is_confident": False,
                    "all_probabilities": {}
                }

        curr_threshold = threshold if threshold is not None else self.confidence_threshold

        # 1. Clean & tokenize
        clean_text = text.strip().lower()
        sequence = self.tokenizer.texts_to_sequences([clean_text])
        padded_sequence = pad_sequences(sequence, maxlen=self.max_len, padding='post', truncating='post')

        # 2. Deep Learning Inference (BiLSTM)
        probabilities = self.model.predict(padded_sequence, verbose=0)[0]

        # 3. Best Class & Confidence Calculation
        max_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[max_idx])
        predicted_intent = str(self.label_encoder.classes_[max_idx])

        # 4. Map probabilities to intent names
        all_probs = {
            str(cls_name): float(prob)
            for cls_name, prob in zip(self.label_encoder.classes_, probabilities)
        }

        is_confident = confidence >= curr_threshold

        return {
            "intent": predicted_intent if is_confident else "unknown",
            "raw_intent": predicted_intent,
            "confidence": confidence,
            "is_confident": is_confident,
            "threshold": curr_threshold,
            "all_probabilities": all_probs
        }
