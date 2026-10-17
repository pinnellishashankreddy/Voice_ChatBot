from collections import Counter

class AnalyticsTracker:
    """Calculates statistics and analytics metrics from session history."""

    @staticmethod
    def compute_analytics(history):
        """
        Calculates aggregate statistics from the conversation history array.
        """
        if not history:
            return {
                "total_messages": 0,
                "voice_messages": 0,
                "text_messages": 0,
                "voice_ratio": 0.0,
                "most_common_intent": "N/A",
                "avg_confidence": 0.0,
                "low_confidence_queries": 0,
                "positive_feedback": 0,
                "negative_feedback": 0,
                "intent_counts": {},
                "emotion_counts": {}
            }

        total_messages = len(history)
        voice_messages = sum(1 for t in history if t.get("input_type") == "voice")
        text_messages = total_messages - voice_messages
        voice_ratio = (voice_messages / total_messages) * 100 if total_messages > 0 else 0.0

        intents = [t.get("intent", "unknown") for t in history]
        intent_counter = Counter(intents)
        most_common_intent = intent_counter.most_common(1)[0][0] if intent_counter else "N/A"

        confidences = [t.get("confidence", 0.0) for t in history]
        avg_confidence = (sum(confidences) / len(confidences)) * 100 if confidences else 0.0

        low_confidence_queries = sum(1 for t in history if t.get("intent") == "unknown" or t.get("confidence", 1.0) < 0.60)

        pos_feedback = sum(1 for t in history if t.get("feedback") == "Positive")
        neg_feedback = sum(1 for t in history if t.get("feedback") == "Negative")

        emotions = [t.get("emotion", "Neutral") for t in history]
        emotion_counter = Counter(emotions)

        return {
            "total_messages": total_messages,
            "voice_messages": voice_messages,
            "text_messages": text_messages,
            "voice_ratio": round(voice_ratio, 1),
            "most_common_intent": most_common_intent,
            "avg_confidence": round(avg_confidence, 1),
            "low_confidence_queries": low_confidence_queries,
            "positive_feedback": pos_feedback,
            "negative_feedback": neg_feedback,
            "intent_counts": dict(intent_counter),
            "emotion_counts": dict(emotion_counter)
        }
