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
import time

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
thread_pool = ThreadPoolExecutor(max_workers=1)  # Single worker for sequential processing

# ElevenLabs voice configuration - optimized for speed
VOICE_CONFIG = {
    "voice_id": "pMsXgVXv3BLzUgSXRplE",
    "settings": VoiceSettings(
        stability=0.1,  # Lower for faster generation
        similarity_boost=0.3,  # Lower for faster generation
        style=0.3,  # Lower for faster generation
        use_speaker_boost=False  # Disabled for faster generation
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

# Session management
active_sessions = set()
SESSION_TIMEOUT = 3600  # 1 hour timeout

# Periodically clean up old sessions
def cleanup_old_sessions():
    """Remove sessions older than SESSION_TIMEOUT"""
    current_time = time.time()
    for session_id in list(active_sessions):
        if session_id in memory and current_time - memory[session_id].get('last_activity', 0) > SESSION_TIMEOUT:
            print(f"Cleaning up old session: {session_id}")
            del memory[session_id]
            active_sessions.discard(session_id)

# Schedule session cleanup to run every 5 minutes
async def schedule_session_cleanup():
    while True:
        cleanup_old_sessions()
        await asyncio.sleep(300)  # 5 minutes

# Start session cleanup when app starts
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(schedule_session_cleanup())

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
    try:
        body = await request.json()
        user_id = body.get("user_id", "default_user")
        user_message = body.get("message", "")

        print(f"Received chat request - User ID: {user_id}, Message: {user_message[:50]}...")

        if not user_message:
            print("Error: No message provided")
            return JSONResponse({
                "error": "No message provided"
            })

        if not user_id:
            print("Error: No user_id provided")
            return JSONResponse({
                "error": "No user_id provided"
            })

        # Initialize or update session
        if user_id not in memory:
            print(f"Initializing memory for new user: {user_id}")
            memory[user_id] = {
                "messages": [],
                "last_activity": time.time()
            }
        active_sessions.add(user_id)
        memory[user_id]["last_activity"] = time.time()

        # Update memory with user input
        memory[user_id]["messages"].append({"role": "user", "content": user_message})
        memory[user_id]["messages"] = memory[user_id]["messages"][-MAX_HISTORY:]
        print(f"Current message count for user {user_id}: {len(memory[user_id]['messages'])}")

        # Construct full chat context
        full_messages = [{"role": "system", "content": JANET_PROMPT}] + memory[user_id]["messages"]
        print(f"Total messages in context: {len(full_messages)}")

        # Call OpenAI with timeout
        print("Calling OpenAI API...")
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    openai.chat.completions.create,
                    model="gpt-4",
                    messages=full_messages,
                    temperature=0.9,
                ),
                timeout=10.0  # 10 second timeout
            )
            assistant_reply = response.choices[0].message.content
            print(f"OpenAI response received: {assistant_reply[:50]}...")
        except asyncio.TimeoutError:
            print("OpenAI API call timed out")
            return JSONResponse({
                "error": "Response generation timed out. Please try again."
            })
        except Exception as e:
            print(f"OpenAI API error: {str(e)}")
            return JSONResponse({
                "error": "Error getting response from OpenAI"
            })

        # Save assistant response to memory
        memory[user_id]["messages"].append({"role": "assistant", "content": assistant_reply})
        memory[user_id]["messages"] = memory[user_id]["messages"][-MAX_HISTORY:]

        # Generate audio with retry mechanism
        print("Starting audio generation...")
        audio_base64 = None
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                audio_base64 = await asyncio.wait_for(
                    asyncio.to_thread(generate_audio, assistant_reply),
                    timeout=15.0  # 15 second timeout
                )
                if audio_base64:
                    print(f"Audio generation completed successfully on attempt {attempt + 1}")
                    break
            except asyncio.TimeoutError:
                print(f"Audio generation attempt {attempt + 1} timed out")
                if attempt < max_retries - 1:
                    print(f"Retrying audio generation in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
            except Exception as e:
                print(f"Audio generation error in attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    print(f"Retrying audio generation in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)

        if not audio_base64:
            print("All audio generation attempts failed")

        response_data = {
            "text": assistant_reply,
            "audio": audio_base64
        }
        print("Sending response to client")
        return JSONResponse(response_data)

    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return JSONResponse({
            "error": "An error occurred while processing your request"
        })

@app.post("/end_session")
async def end_session(request: Request):
    try:
        body = await request.json()
        user_id = body.get("user_id")
        
        if not user_id:
            return JSONResponse({"error": "No user_id provided"}, status_code=400)
            
        if user_id in memory:
            # Clean up session data
            del memory[user_id]
            active_sessions.discard(user_id)
            print(f"Session {user_id} ended successfully")
            return JSONResponse({"status": "success"})
            
        print(f"Attempted to end non-existent session: {user_id}")
        return JSONResponse({"status": "session not found"}, status_code=404)
        
    except Exception as e:
        print(f"Error ending session: {str(e)}")
        return JSONResponse({
            "error": "Failed to end session",
            "details": str(e)
        }, status_code=500)
