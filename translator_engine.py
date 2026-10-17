from deep_translator import GoogleTranslator

class TranslationEngine:
    """Multilingual Speech & Response Translation Engine."""

    LANG_MAP = {
        "English (en-US)": "en",
        "Hindi (hi-IN)": "hi",
        "Telugu (te-IN)": "te",
        "Tamil (ta-IN)": "ta",
        "Spanish (es-ES)": "es",
        "French (fr-FR)": "fr"
    }

    @staticmethod
    def translate_to_english(text: str, source_language: str) -> str:
        """Translates user text from source language to English for model intent processing."""
        if not text or not text.strip():
            return ""

        lang_code = TranslationEngine.LANG_MAP.get(source_language, source_language)
        if lang_code == "en" or lang_code == "en-US":
            return text

        try:
            translator = GoogleTranslator(source='auto', target='en')
            translated = translator.translate(text.strip())
            return translated if translated else text
        except Exception as e:
            print(f"[TranslationEngine] Error translating to English: {e}")
            return text

    @staticmethod
    def translate_from_english(text: str, target_language: str) -> str:
        """Translates chatbot response from English into user's selected target language."""
        if not text or not text.strip():
            return ""

        lang_code = TranslationEngine.LANG_MAP.get(target_language, target_language)
        if lang_code == "en" or lang_code == "en-US":
            return text

        try:
            translator = GoogleTranslator(source='en', target=lang_code)
            translated = translator.translate(text.strip())
            return translated if translated else text
        except Exception as e:
            print(f"[TranslationEngine] Error translating from English: {e}")
            return text
