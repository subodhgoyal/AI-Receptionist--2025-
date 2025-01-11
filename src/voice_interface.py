from google.cloud import speech_v1
from google.cloud import texttospeech
import os
import base64
import pyaudio
import wave
import threading
import queue
import time
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class VoiceInterface:
    def __init__(self):
        # Set Google API key from environment variable
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY is not set in the environment variables.")

        # Audio recording parameters
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.is_recording = False
        self.audio_queue = queue.Queue()

        # Create audio output directory if it doesn't exist
        self.audio_output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "audio_output")
        os.makedirs(self.audio_output_dir, exist_ok=True)

    def start_recording(self):
        """Start recording audio from microphone"""
        self.is_recording = True
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )
        print("Recording started...")
        self.recording_thread = threading.Thread(target=self._record_audio)
        self.recording_thread.start()

    def stop_recording(self):
        """Stop recording audio"""
        self.is_recording = False
        if hasattr(self, 'recording_thread'):
            self.recording_thread.join()
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'audio'):
            self.audio.terminate()
        print("Recording stopped.")

    def _record_audio(self):
        """Record audio in a separate thread"""
        while self.is_recording:
            data = self.stream.read(self.CHUNK, exception_on_overflow=False)
            self.audio_queue.put(data)

    def transcribe_audio(self):
        """Transcribe recorded audio using Google Speech-to-Text with API key"""
        import requests
        import base64

        url = f"https://speech.googleapis.com/v1/speech:recognize?key={self.api_key}"

        # Collect audio from the queue and save it to a temporary file
        audio_data = b"".join(list(self.audio_queue.queue))
        temp_audio_file = os.path.join(self.audio_output_dir, "temp_audio.wav")
        with wave.open(temp_audio_file, "wb") as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(pyaudio.PyAudio().get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(audio_data)

        # Convert audio to base64
        with open(temp_audio_file, "rb") as audio_file:
            audio_content = base64.b64encode(audio_file.read()).decode("utf-8")

        payload = {
            "config": {
                "encoding": "LINEAR16",
                "sampleRateHertz": 16000,
                "languageCode": "en-US",
            },
            "audio": {"content": audio_content},
        }

        response = requests.post(url, json=payload)
        if response.status_code == 200:
            result = response.json()
            return result.get("results", [{}])[0].get("alternatives", [{}])[0].get("transcript", "")
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return ""

    def text_to_speech(self, text, output_file=None):
        """Convert text to speech using Google Text-to-Speech with API key"""
        import requests

        url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={self.api_key}"

        payload = {
            "input": {"text": text},
            "voice": {
                "languageCode": "en-US",
                "name": "en-US-Neural2-F",
                "ssmlGender": "FEMALE",
            },
            "audioConfig": {
                "audioEncoding": "LINEAR16",
                "speakingRate": 0.9,
                "pitch": 0.0,
            },
        }

        response = requests.post(url, json=payload)
        if response.status_code == 200:
            audio_content = response.json()["audioContent"]

            if not output_file:
                output_file = os.path.join(
                    self.audio_output_dir,
                    f"response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
                )

            with open(output_file, "wb") as out:
                out.write(base64.b64decode(audio_content))

            return output_file
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None

    def play_audio_response(self, audio_file):
        """Play the generated audio response"""
        wf = wave.open(audio_file, 'rb')
        p = pyaudio.PyAudio()

        stream = p.open(
            format=p.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True
        )

        data = wf.readframes(self.CHUNK)
        while data:
            stream.write(data)
            data = wf.readframes(self.CHUNK)

        stream.stop_stream()
        stream.close()
        p.terminate()
