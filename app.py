from fastapi import FastAPI, Request, Form
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import openai
import os
import json
from dotenv import load_dotenv
from elevenlabs import Voice, VoiceSettings, generate as elevenlabs_generate
from backend.memory_store import memory, MAX_HISTORY
from backend.prompts import JANET_PROMPT

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
os.environ["ELEVENLABS_API_KEY"] = os.getenv("ELEVENLABS_API_KEY")

# ElevenLabs voice configuration
VOICE_CONFIG = {
    "voice_id": "pMsXgVXv3BLzUgSXRplE",
    "settings": VoiceSettings(
        stability=0.3,
        similarity_boost=0.75,
        style=0.9,
        use_speaker_boost=True
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
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

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

    # Update memory with user input
    memory[user_id]["messages"].append({"role": "user", "content": user_message})
    memory[user_id]["messages"] = memory[user_id]["messages"][-MAX_HISTORY:]

    # Construct full chat context
    full_messages = [{"role": "system", "content": JANET_PROMPT}] + memory[user_id]["messages"]

    try:
        # Call OpenAI
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=full_messages,
            temperature=0.9,
        )
        assistant_reply = response.choices[0].message.content

        # Generate audio using ElevenLabs with configured voice
        voice = Voice(
            voice_id=VOICE_CONFIG["voice_id"],
            settings=VOICE_CONFIG["settings"]
        )
        
        audio = elevenlabs_generate(
            text=assistant_reply,
            voice=voice,
            model="eleven_monolingual_v1"
        )

        # Save assistant response to memory
        memory[user_id]["messages"].append({"role": "assistant", "content": assistant_reply})
        memory[user_id]["messages"] = memory[user_id]["messages"][-MAX_HISTORY:]

        return JSONResponse({
            "text": assistant_reply,
            "audio": audio
        })

    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse({
            "error": "An error occurred while processing your request"
        })
