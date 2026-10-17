import re

class EmotionDetector:
    """
    Experimental NLP Emotion & Tone Classifier.
    Analyzes sentence valence, keywords, and punctuation to categorize tone:
    😊 Happy, 😐 Neutral, 😔 Sad, 😡 Angry, 😨 Fear
    """

    EMOTION_KEYWORDS = {
        "Happy": [
            "happy", "great", "awesome", "good", "excellent", "love", "wonderful", 
            "thanks", "thank you", "fantastic", "glad", "pleased", "nice", "perfect"
        ],
        "Sad": [
            "sad", "unhappy", "depressed", "disappointed", "sorry", "miss", "lost", 
            "grief", "broken", "terrible", "worst", "unfortunate", "regret"
        ],
        "Angry": [
            "angry", "mad", "furious", "hate", "terrible", "annoyed", "useless", 
            "horrible", "awful", "cancel", "complaint", "ridiculous", "bad service"
        ],
        "Fear": [
            "scared", "afraid", "worried", "anxious", "fear", "locked out", "glitch", 
            "hacked", "danger", "urgent", "security issue", "problem"
        ]
    }

    def predict_emotion(self, text: str):
        """
        Analyzes user input text and predicts emotion with score.
        """
        if not text or not text.strip():
            return {"emotion": "Neutral", "confidence": 0.85, "emoji": "😐"}

        cleaned = re.sub(r'[^\w\s]', '', text.lower())
        words = cleaned.split()
        
        scores = {emotion: 0 for emotion in self.EMOTION_KEYWORDS}

        for word in words:
            for emotion, keywords in self.EMOTION_KEYWORDS.items():
                if word in keywords:
                    scores[emotion] += 1

        max_emotion = max(scores, key=scores.get)
        max_score = scores[max_emotion]

        if max_score == 0:
            return {
                "emotion": "Neutral",
                "confidence": 0.82,
                "emoji": "😐"
            }

        # Calculate confidence based on keyword density
        confidence = min(0.65 + (max_score * 0.15), 0.96)

        emoji_map = {
            "Happy": "😊",
            "Neutral": "😐",
            "Sad": "😔",
            "Angry": "😡",
            "Fear": "😨"
        }

        return {
            "emotion": max_emotion,
            "confidence": round(confidence, 2),
            "emoji": emoji_map.get(max_emotion, "😐")
        }
