import speech_recognition as sr
import os
import tempfile

class SpeechToTextEngine:
    """Enhanced Multilingual Speech Recognition Engine."""

    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 250
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8

    def transcribe_audio_buffer(self, audio_buffer, language: str = 'en-US') -> dict:
        """
        Transcribes audio buffer using SpeechRecognition Google STT API in target BCP-47 language.
        """
        if not audio_buffer:
            return {
                "success": False,
                "text": "",
                "error": "No audio input received."
            }

        temp_wav_path = None
        try:
            raw_bytes = audio_buffer.read()
            if len(raw_bytes) < 300:
                return {
                    "success": False,
                    "text": "",
                    "error": "Audio clip too short. Please speak clearly for 2-3 seconds."
                }

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
                temp_wav.write(raw_bytes)
                temp_wav_path = temp_wav.name

            with sr.AudioFile(temp_wav_path) as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.2)
                audio_data = self.recognizer.record(source)
                
                # Transcribe in requested BCP-47 language (e.g. te-IN, hi-IN, ta-IN, en-US)
                text = self.recognizer.recognize_google(audio_data, language=language)
                
                if text and text.strip():
                    return {
                        "success": True,
                        "text": text.strip(),
                        "error": None
                    }
                else:
                    return {
                        "success": False,
                        "text": "",
                        "error": "Could not detect clear words. Please try again."
                    }

        except sr.UnknownValueError:
            return {
                "success": False,
                "text": "",
                "error": "Speech was not recognized clearly. Please speak louder and clearly."
            }
        except sr.RequestError as e:
            return {
                "success": False,
                "text": "",
                "error": f"Speech network error ({e}). Check internet connection."
            }
        except Exception as e:
            return {
                "success": False,
                "text": "",
                "error": f"Speech error: {str(e)}"
            }
        finally:
            if temp_wav_path and os.path.exists(temp_wav_path):
                try:
                    os.remove(temp_wav_path)
                except Exception:
                    pass

    @staticmethod
    def get_web_speech_component_html(language_code='en-US', lang_display_name='English'):
        """
        Returns Web Speech API component configured with exact BCP-47 code (e.g. te-IN, hi-IN, ta-IN).
        """
        html_code = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{
                    font-family: 'Inter', system-ui, -apple-system, sans-serif;
                    background: transparent;
                    margin: 0;
                    padding: 0;
                }}
                .mic-card {{
                    background: #1E293B;
                    border: 2px solid #38BDF8;
                    border-radius: 14px;
                    padding: 20px;
                    text-align: center;
                    box-shadow: 0 10px 25px rgba(0,0,0,0.4);
                }}
                .mic-btn {{
                    background: linear-gradient(135deg, #10B981 0%, #059669 100%);
                    color: #FFFFFF;
                    border: none;
                    padding: 16px 32px;
                    font-size: 17px;
                    font-weight: 800;
                    border-radius: 10px;
                    cursor: pointer;
                    display: inline-flex;
                    align-items: center;
                    gap: 12px;
                    box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4);
                    transition: all 0.25s ease-in-out;
                    text-transform: uppercase;
                }}
                .mic-btn:hover {{
                    transform: translateY(-2px) scale(1.03);
                    box-shadow: 0 10px 25px rgba(16, 185, 129, 0.6);
                }}
                .mic-btn.listening {{
                    background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
                    animation: pulseRing 1.5s infinite;
                }}
                @keyframes pulseRing {{
                    0% {{ box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.8); }}
                    70% {{ box-shadow: 0 0 0 16px rgba(239, 68, 68, 0); }}
                    100% {{ box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }}
                }}
                .status-text {{
                    color: #38BDF8;
                    font-size: 15px;
                    font-weight: 700;
                    margin-top: 14px;
                }}
                .transcript-display {{
                    background: #0F172A;
                    color: #FFFFFF;
                    border: 2px solid #10B981;
                    padding: 16px;
                    border-radius: 10px;
                    margin-top: 14px;
                    font-size: 18px;
                    font-weight: 800;
                    word-break: break-word;
                }}
            </style>
        </head>
        <body>
            <div class="mic-card">
                <button id="micBtn" class="mic-btn" onclick="toggleMic()">
                    <span id="micIcon">🎤</span> <span id="micText">SPEAK NOW IN {lang_display_name.upper()}</span>
                </button>
                <div id="status" class="status-text">Target Language: <strong>{language_code} ({lang_display_name})</strong></div>
                <div id="transcript" class="transcript-display" style="display:none;"></div>
            </div>

            <script>
                var recognition = null;
                var listening = false;

                if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {{
                    var SR = window.SpeechRecognition || window.webkitSpeechRecognition;
                    recognition = new SR();
                    recognition.continuous = false;
                    recognition.interimResults = true;
                    
                    // Set BCP-47 language tag EXACTLY (e.g. te-IN for Telugu, hi-IN for Hindi, en-US for English)
                    recognition.lang = '{language_code}';

                    recognition.onstart = function() {{
                        listening = true;
                        var btn = document.getElementById('micBtn');
                        btn.classList.add('listening');
                        document.getElementById('micText').innerText = 'LISTENING IN {lang_display_name.upper()}...';
                        document.getElementById('status').innerText = '🎙️ Recording speech in {lang_display_name}... Speak now!';
                        document.getElementById('status').style.color = '#F59E0B';
                    }};

                    recognition.onresult = function(event) {{
                        var text = '';
                        for (var i = event.resultIndex; i < event.results.length; ++i) {{
                            text += event.results[i][0].transcript;
                        }}
                        var box = document.getElementById('transcript');
                        box.style.display = 'block';
                        box.innerText = '📝 Recognized ({lang_display_name}): "' + text + '"';
                        
                        if (event.results[0].isFinal) {{
                            document.getElementById('status').innerText = '✅ Speech captured in {lang_display_name}! Copy or submit text below:';
                            document.getElementById('status').style.color = '#10B981';
                        }}
                    }};

                    recognition.onerror = function(e) {{
                        listening = false;
                        var btn = document.getElementById('micBtn');
                        btn.classList.remove('listening');
                        document.getElementById('micText').innerText = 'SPEAK NOW IN {lang_display_name.upper()}';
                        document.getElementById('status').innerText = '⚠️ Speech error: ' + e.error;
                        document.getElementById('status').style.color = '#EF4444';
                    }};

                    recognition.onend = function() {{
                        listening = false;
                        var btn = document.getElementById('micBtn');
                        btn.classList.remove('listening');
                        document.getElementById('micText').innerText = 'SPEAK NOW IN {lang_display_name.upper()}';
                    }};
                }} else {{
                    document.getElementById('status').innerText = '⚠️ Web Speech API unavailable in browser. Use direct microphone recorder tab.';
                    document.getElementById('status').style.color = '#EF4444';
                }}

                function toggleMic() {{
                    if (!recognition) return;
                    if (listening) {{
                        recognition.stop();
                    }} else {{
                        recognition.start();
                    }}
                }}
            </script>
        </body>
        </html>
        """
        return html_code
