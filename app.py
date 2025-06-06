from fastapi import FastAPI, Request, Form
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import openai
import os
import json
import base64
from dotenv import load_dotenv
from elevenlabs import Voice, VoiceSettings, generate as elevenlabs_generate, set_api_key
from backend.memory_store import memory, MAX_HISTORY
from backend.prompts import JANET_PROMPT
import asyncio
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

# Check for required environment variables
required_env_vars = {
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "ELEVENLABS_API_KEY": os.getenv("ELEVENLABS_API_KEY")
}

missing_vars = [var for var, value in required_env_vars.items() if not value]
if missing_vars:
    raise EnvironmentError(
        f"Missing required environment variables: {', '.join(missing_vars)}. "
        "Please set these in your Render environment variables."
    )

# Initialize API keys
openai.api_key = required_env_vars["OPENAI_API_KEY"]
set_api_key(required_env_vars["ELEVENLABS_API_KEY"])

# Create a thread pool for CPU-bound tasks
thread_pool = ThreadPoolExecutor(max_workers=1)

# ElevenLabs voice configuration
VOICE_CONFIG = {
    "voice_id": "pMsXgVXv3BLzUgSXRplE",
    "settings": VoiceSettings(
        stability=0.1,
        similarity_boost=0.3,
        style=0.3,
        use_speaker_boost=False
    )
}

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
templates = Jinja2Templates(directory="templates")

# Only mount static files if the directory exists
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

def generate_audio(text):
    """Generate audio in a separate thread to avoid blocking"""
    try:
        print(f"Starting audio generation for text: {text[:50]}...")
        voice = Voice(
            voice_id=VOICE_CONFIG["voice_id"],
            settings=VOICE_CONFIG["settings"]
        )
        print("Voice configured, calling ElevenLabs API...")
        audio = elevenlabs_generate(
            text=text,
            voice=voice,
            model="eleven_monolingual_v1"
        )
        print(f"Audio generated successfully, length: {len(audio)} bytes")
        encoded = base64.b64encode(audio).decode('utf-8')
        print(f"Audio encoded to base64, length: {len(encoded)}")
        return encoded
    except Exception as e:
        print(f"Audio generation error: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return None

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    user_id = body.get("user_id", "default_user")
    user_message = body.get("message", "")

    if not user_message:
        return JSONResponse({
            "error": "No message provided"
        })

    try:
        # Update memory with user input
        memory[user_id]["messages"].append({"role": "user", "content": user_message})
        memory[user_id]["messages"] = memory[user_id]["messages"][-MAX_HISTORY:]

        # Construct full chat context
        full_messages = [{"role": "system", "content": JANET_PROMPT}] + memory[user_id]["messages"]

        # Call OpenAI
        print("Calling OpenAI API...")
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=full_messages,
            temperature=0.9,
        )
        assistant_reply = response.choices[0].message.content
        print(f"OpenAI response received: {assistant_reply[:50]}...")

        # Save assistant response to memory
        memory[user_id]["messages"].append({"role": "assistant", "content": assistant_reply})
        memory[user_id]["messages"] = memory[user_id]["messages"][-MAX_HISTORY:]

        # Generate audio in a separate thread
        print("Starting audio generation...")
        loop = asyncio.get_event_loop()
        audio_base64 = await loop.run_in_executor(
            thread_pool,
            generate_audio,
            assistant_reply
        )
        print(f"Audio generation completed, result: {'success' if audio_base64 else 'failed'}")

        return JSONResponse({
            "text": assistant_reply,
            "audio": audio_base64
        })

    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return JSONResponse({
            "error": "An error occurred while processing your request"
        })
