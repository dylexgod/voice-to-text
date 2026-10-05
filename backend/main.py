from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import tempfile
import os
import traceback

app = FastAPI(title="Voice to Text API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading Whisper model...")

model = WhisperModel(
    "tiny",
    device="cpu",
    compute_type="int8"
)

print("Whisper model loaded.")


@app.get("/")
def home():
    return {
        "message": "Voice-to-Text API is running"
    }


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):

    temp_path = None

    try:

        print(f"Received file: {audio.filename}")
        print(f"Content type: {audio.content_type}")

        # Save uploaded audio
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".webm"
        ) as temp_file:

            audio_data = await audio.read()

            print(f"Audio size: {len(audio_data)} bytes")

            temp_file.write(audio_data)

            temp_path = temp_file.name

        print("Starting transcription...")

        segments, info = model.transcribe(
            temp_path,
            beam_size=5
        )

        text = " ".join(
            segment.text.strip()
            for segment in segments
        )

        print(f"Transcription: {text}")

        return {
            "text": text,
            "language": info.language
        }

    except Exception as e:

        print("\n========== BACKEND ERROR ==========")
        traceback.print_exc()
        print("===================================\n")

        return {
            "error": str(e)
        }

    finally:

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
