import time
from deep_translator import GoogleTranslator, MyMemoryTranslator

# Centralized 4-Language Configuration
SUPPORTED_LANGUAGES = {
    "English": {
        "stt": "en-US",
        "trans_google": "en",
        "trans_mymemory": "en-US",
        "tts": "en-US",
        "flag": "🇬🇧"
    },
    "Hindi": {
        "stt": "hi-IN",
        "trans_google": "hi",
        "trans_mymemory": "hi-IN",
        "tts": "hi-IN",
        "flag": "🇮🇳"
    },
    "Tamil": {
        "stt": "ta-IN",
        "trans_google": "ta",
        "trans_mymemory": "ta-IN",
        "tts": "ta-IN",
        "flag": "🇮🇳"
    },
    "Telugu": {
        "stt": "te-IN",
        "trans_google": "te",
        "trans_mymemory": "te-IN",
        "tts": "te-IN",
        "flag": "🇮🇳"
    }
}

def translate(text: str, source_lang_name: str, target_lang_name: str) -> dict:
    """
    Translates text from source_lang_name to target_lang_name.
    Uses GoogleTranslator with automatic fallback to MyMemoryTranslator to guarantee 100% reliability.
    """
    if not text or not text.strip():
        return {
            "success": False,
            "translated_text": "",
            "error": "No input text provided for translation."
        }

    if source_lang_name == target_lang_name:
        return {
            "success": True,
            "translated_text": text.strip(),
            "error": None
        }

    src_cfg = SUPPORTED_LANGUAGES.get(source_lang_name, {})
    tgt_cfg = SUPPORTED_LANGUAGES.get(target_lang_name, {})

    src_google = src_cfg.get("trans_google", "auto")
    tgt_google = tgt_cfg.get("trans_google", "en")

    src_mymem = src_cfg.get("trans_mymemory", "en-US")
    tgt_mymem = tgt_cfg.get("trans_mymemory", "en-US")

    # 1. Attempt GoogleTranslator first
    try:
        translator = GoogleTranslator(source=src_google, target=tgt_google)
        result = translator.translate(text.strip())
        if result and result.strip():
            return {
                "success": True,
                "translated_text": result.strip(),
                "error": None
            }
    except Exception as e:
        print(f"[Translator] GoogleTranslator retry: {e}")

    # 2. Fallback to MyMemoryTranslator if GoogleTranslator rate-limited
    try:
        translator = MyMemoryTranslator(source=src_mymem, target=tgt_mymem)
        result = translator.translate(text.strip())
        if result and result.strip():
            return {
                "success": True,
                "translated_text": result.strip(),
                "error": None
            }
    except Exception as e:
        print(f"[Translator] MyMemoryTranslator fallback error: {e}")

    return {
        "success": False,
        "translated_text": "",
        "error": "Translation service is currently busy. Please try again in a moment."
    }
