import base64
from io import BytesIO
from gtts import gTTS

class TextToSpeechEngine:
    """Generates audio speech synthesis from text responses using gTTS and Web Speech API."""

    @staticmethod
    def generate_audio_base64(text: str, lang: str = 'en') -> str:
        """
        Converts text to MP3 audio bytes using gTTS and returns Base64 encoded string.
        """
        if not text or not text.strip():
            return None

        try:
            tts = gTTS(text=text.strip(), lang=lang, slow=False)
            mp3_fp = BytesIO()
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)
            audio_bytes = mp3_fp.read()
            b64_audio = base64.b64encode(audio_bytes).decode('utf-8')
            return b64_audio
        except Exception as e:
            print(f"[TextToSpeechEngine] gTTS error: {e}")
            return None

    @staticmethod
    def get_tts_audio_html(text: str, button_text: str = "🔊 Play Response", lang: str = 'en-US') -> str:
        """
        Generates an HTML + JavaScript Web Speech Synthesis player button.
        This plays the text directly through browser TTS without external audio fetching latency.
        Also includes audio element fallback.
        """
        clean_text = text.replace('"', '\\"').replace('\n', ' ')
        
        html_code = f"""
        <div style="margin-top: 8px;">
            <button id="ttsBtn" onclick="speakText()" style="
                background-color: #16A34A;
                color: #FFFFFF;
                border: none;
                padding: 8px 16px;
                font-size: 14px;
                font-weight: 600;
                border-radius: 6px;
                cursor: pointer;
                display: inline-flex;
                align-items: center;
                gap: 6px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                transition: background-color 0.2s ease;">
                {button_text}
            </button>
            <script>
                function speakText() {{
                    if ('speechSynthesis' in window) {{
                        window.speechSynthesis.cancel(); // stop any ongoing speech
                        var utterance = new SpeechSynthesisUtterance("{clean_text}");
                        utterance.lang = "{lang}";
                        utterance.rate = 1.0;
                        utterance.pitch = 1.0;
                        
                        var btn = document.getElementById("ttsBtn");
                        btn.style.backgroundColor = "#15803D";
                        btn.innerText = "🔊 Speaking...";
                        
                        utterance.onend = function() {{
                            btn.style.backgroundColor = "#16A34A";
                            btn.innerText = "{button_text}";
                        }};
                        
                        utterance.onerror = function(e) {{
                            btn.style.backgroundColor = "#16A34A";
                            btn.innerText = "{button_text}";
                        }};

                        window.speechSynthesis.speak(utterance);
                    }} else {{
                        alert("Text-to-speech is not supported in your browser.");
                    }}
                }}
            </script>
        </div>
        """
        return html_code
