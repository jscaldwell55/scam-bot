from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os
import base64
from dotenv import load_dotenv
from elevenlabs import Voice, VoiceSettings, generate as elevenlabs_generate, set_api_key
from backend.memory_store import memory, MAX_HISTORY
from backend.prompts import JANET_PROMPT
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Initialize OpenAI client (updated for new API)
client = OpenAI(api_key=required_env_vars["OPENAI_API_KEY"])

# Initialize ElevenLabs
set_api_key(required_env_vars["ELEVENLABS_API_KEY"])

# Create a thread pool for CPU-bound tasks
thread_pool = ThreadPoolExecutor(max_workers=2)

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

# Templates
templates = Jinja2Templates(directory="templates")

# Mount static files if they exist
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Session management
SESSION_TIMEOUT = 3600  # 1 hour

def cleanup_old_sessions():
    """Remove sessions older than SESSION_TIMEOUT"""
    current_time = time.time()
    sessions_to_remove = []
    
    for session_id, session_data in memory.items():
        if current_time - session_data.get('last_activity', 0) > SESSION_TIMEOUT:
            sessions_to_remove.append(session_id)
    
    for session_id in sessions_to_remove:
        logger.info(f"Cleaning up old session: {session_id}")
        del memory[session_id]

async def periodic_cleanup():
    """Run cleanup every 5 minutes"""
    while True:
        await asyncio.sleep(300)
        cleanup_old_sessions()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(periodic_cleanup())

def generate_audio(text: str) -> str:
    """Generate audio using ElevenLabs"""
    try:
        logger.info(f"Generating audio for text: {text[:50]}...")
        voice = Voice(
            voice_id=VOICE_CONFIG["voice_id"],
            settings=VOICE_CONFIG["settings"]
        )
        
        audio = elevenlabs_generate(
            text=text,
            voice=voice,
            model="eleven_monolingual_v1"
        )
        
        # Convert to base64
        encoded = base64.b64encode(audio).decode('utf-8')
        logger.info(f"Audio generated successfully, base64 length: {len(encoded)}")
        return encoded
        
    except Exception as e:
        logger.error(f"Audio generation error: {str(e)}", exc_info=True)
        return None

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/chat")
async def chat(request: Request):
    try:
        body = await request.json()
        user_id = body.get("user_id", "default_user")
        user_message = body.get("message", "")

        logger.info(f"Chat request - User: {user_id}, Message: {user_message[:50]}...")

        # Validate input
        if not user_message:
            return JSONResponse(
                {"error": "No message provided"}, 
                status_code=400
            )

        if not user_id:
            return JSONResponse(
                {"error": "No user_id provided"}, 
                status_code=400
            )

        # Initialize or get session
        if user_id not in memory:
            logger.info(f"Creating new session for user: {user_id}")
            memory[user_id] = {
                "messages": [],
                "last_activity": time.time()
            }
        
        # Update last activity
        memory[user_id]["last_activity"] = time.time()

        # Add user message to history
        memory[user_id]["messages"].append({
            "role": "user", 
            "content": user_message
        })
        
        # Keep only recent messages
        memory[user_id]["messages"] = memory[user_id]["messages"][-MAX_HISTORY:]

        # Prepare messages for OpenAI
        messages = [
            {"role": "system", "content": JANET_PROMPT}
        ] + memory[user_id]["messages"]

        # Call OpenAI API
        logger.info("Calling OpenAI API...")
        try:
            completion = await asyncio.wait_for(
                asyncio.to_thread(
                    client.chat.completions.create,
                    model="gpt-4",
                    messages=messages,
                    temperature=0.9,
                    max_tokens=150  # Limit response length
                ),
                timeout=15.0
            )
            
            assistant_reply = completion.choices[0].message.content
            logger.info(f"OpenAI response: {assistant_reply[:50]}...")
            
        except asyncio.TimeoutError:
            logger.error("OpenAI API timeout")
            return JSONResponse(
                {"error": "Response generation timed out. Please try again."},
                status_code=504
            )
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}", exc_info=True)
            return JSONResponse(
                {"error": "Error generating response"},
                status_code=500
            )

        # Add assistant response to memory
        memory[user_id]["messages"].append({
            "role": "assistant", 
            "content": assistant_reply
        })
        memory[user_id]["messages"] = memory[user_id]["messages"][-MAX_HISTORY:]

        # Generate audio asynchronously
        logger.info("Generating audio...")
        audio_task = asyncio.create_task(
            asyncio.to_thread(generate_audio, assistant_reply)
        )
        
        # Set a timeout for audio generation
        try:
            audio_base64 = await asyncio.wait_for(audio_task, timeout=20.0)
        except asyncio.TimeoutError:
            logger.warning("Audio generation timed out")
            audio_base64 = None

        # Return response
        response_data = {
            "text": assistant_reply,
            "audio": audio_base64
        }
        
        logger.info("Sending response to client")
        return JSONResponse(response_data)

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        return JSONResponse(
            {"error": "An error occurred while processing your request"},
            status_code=500
        )

@app.post("/end_session")
async def end_session(request: Request):
    try:
        body = await request.json()
        user_id = body.get("user_id")
        
        if not user_id:
            return JSONResponse(
                {"error": "No user_id provided"}, 
                status_code=400
            )
        
        if user_id in memory:
            del memory[user_id]
            logger.info(f"Session {user_id} ended successfully")
            return JSONResponse({"status": "success"})
        
        logger.warning(f"Attempted to end non-existent session: {user_id}")
        return JSONResponse(
            {"status": "session not found"}, 
            status_code=404
        )
        
    except Exception as e:
        logger.error(f"Error ending session: {str(e)}", exc_info=True)
        return JSONResponse(
            {"error": "Failed to end session"},
            status_code=500
        )