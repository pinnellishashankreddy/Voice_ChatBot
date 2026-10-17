import json
from datetime import datetime

class ContextManager:
    """Manages multi-turn conversation context, user memory, and export capabilities."""

    def __init__(self, max_history=50):
        self.max_history = max_history
        self.history = []
        self.session_start_time = datetime.now()

    def add_turn(self, input_type: str, user_text: str, bot_response: str, intent: str, confidence: float, emotion: str = "Neutral", feedback: str = None):
        """Record a single dialogue turn in context memory."""
        turn = {
            "id": len(self.history) + 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input_type": input_type, # 'voice' or 'text'
            "user_text": user_text,
            "bot_response": bot_response,
            "intent": intent,
            "confidence": round(confidence, 4),
            "emotion": emotion,
            "feedback": feedback
        }
        self.history.append(turn)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        return turn

    def update_feedback(self, turn_index: int, feedback: str):
        """Update feedback for a given turn."""
        if 0 <= turn_index < len(self.history):
            self.history[turn_index]["feedback"] = feedback

    def get_last_intent(self):
        """Returns the most recent intent recognized."""
        if self.history:
            return self.history[-1]["intent"]
        return None

    def clear_history(self):
        """Clear conversation history."""
        self.history = []

    def export_as_txt(self) -> str:
        """Export history in clean human-readable text transcript format."""
        lines = [
            "=" * 60,
            f"VocaMind Conversation Transcript",
            f"Session Start: {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Turns: {len(self.history)}",
            "=" * 60,
            ""
        ]

        for turn in self.history:
            lines.append(f"[{turn['timestamp']}] ({turn['input_type'].upper()})")
            lines.append(f"👤 USER: {turn['user_text']}")
            lines.append(f"🤖 VOCAMIND: {turn['bot_response']}")
            lines.append(f"   ↳ [Intent: {turn['intent']} | Confidence: {turn['confidence']*100:.1f}% | Emotion: {turn['emotion']}]")
            if turn.get('feedback'):
                lines.append(f"   ↳ [Feedback: {turn['feedback']}]")
            lines.append("-" * 50)

        return "\n".join(lines)

    def export_as_json(self) -> str:
        """Export history in JSON format."""
        export_data = {
            "session_start": self.session_start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_turns": len(self.history),
            "turns": self.history
        }
        return json.dumps(export_data, indent=2)
