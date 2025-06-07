from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import openai
import os
import json
import base64
from dotenv import load_dotenv
from elevenlabs import Voice, VoiceSettings, generate as elevenlabs_generate, set_api_key, APIError as ElevenLabsAPIError
from backend.memory_store import memory, MAX_HISTORY # Your existing memory store
from backend.prompts import JANET_PROMPT # Your updated Janet prompt
import asyncio
from concurrent.futures import ThreadPoolExecutor
import traceback # For detailed error logging

# --- Global Configuration & Initialization ---
load_dotenv()

# Check for required environment variables
required_env_vars_map = {
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "ELEVENLABS_API_KEY": os.getenv("ELEVENLABS_API_KEY")
}

missing_vars = [var_name for var_name, var_value in required_env_vars_map.items() if not var_value]
if missing_vars:
    raise EnvironmentError(
        f"Missing required environment variables: {', '.join(missing_vars)}. "
        "Please set these in your .env file or environment variables."
    )

openai.api_key = required_env_vars_map["OPENAI_API_KEY"]
set_api_key(required_env_vars_map["ELEVENLABS_API_KEY"])

# Create a thread pool for CPU-bound/blocking I/O tasks
# Increased workers for better concurrency, adjust based on load and API limits.
# (Your original was max_workers=1, this is a suggested improvement)
thread_pool = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)

# ElevenLabs voice configuration (using your original values, consider tuning as per previous suggestions)
JANET_VOICE_ID = "pMsXgVXv3BLzUgSXRplE" # Your Voice ID
VOICE_CONFIG = {
    "voice_id": JANET_VOICE_ID,
    "settings": VoiceSettings(
        stability=0.3,       # Example: Slightly more expressive
        similarity_boost=0.7,  # Example: Higher similarity
        style=0.2,           # Example: Subtle style
        use_speaker_boost=True # Example: Often good for clarity
    )
}

app = FastAPI()

# Add CORS middleware (allow all for development, restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Or specify your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
# Assumes 'templates' and 'static' directories are at the same level as this app.py
current_dir = os.path.dirname(__file__)
templates_dir = os.path.join(current_dir, "templates")
static_dir = os.path.join(current_dir, "static")

if not os.path.exists(templates_dir):
    print(f"Warning: Templates directory not found at {templates_dir}")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
else:
    print(f"Warning: Static directory not found at {static_dir}, not mounting.")

# Ensure templates_dir is valid before creating Jinja2Templates instance
templates = Jinja2Templates(directory=templates_dir if os.path.exists(templates_dir) else ".")


def run_openai_call_sync(messages_for_api):
    """Synchronous function for OpenAI API call to be run in a thread."""
    print(f"OpenAI: Calling API with {len(messages_for_api)} messages. Last user msg: '{messages_for_api[-1]['content'][:50]}...'")
    try:
        response = openai.chat.completions.create(
            model="gpt-4-turbo-preview", # Faster and often good enough for chat
            messages=messages_for_api,
            temperature=0.85, # Adjusted for a bit more creativity but still focused
            max_tokens=350,  # Limit response length to manage latency and cost
        )
        assistant_reply = response.choices[0].message.content
        if not assistant_reply:
            print("OpenAI Warning: Received an empty content string.")
            return "..." # Return a placeholder if empty to avoid None issues downstream
        return assistant_reply
    except openai.APIError as e:
        print(f"OpenAI API Error: Status {e.status_code} - {e.message}")
        # Re-raise to be caught and handled specifically in the main endpoint
        raise
    except Exception as e:
        print(f"OpenAI: Generic Call Error: {type(e).__name__} - {str(e)}")
        print(traceback.format_exc())
        # Re-raise to be caught by the main endpoint's error handler
        raise


def generate_audio_sync(text_to_speak):
    """Synchronous function for ElevenLabs audio generation to be run in a thread."""
    if not text_to_speak or text_to_speak == "...": # Don't try to generate audio for placeholders
        print("ElevenLabs: Skipping audio generation for empty or placeholder text.")
        return None
    try:
        print(f"ElevenLabs: Starting audio generation for text (first 50 chars): '{text_to_speak[:50]}...'")
        voice_obj = Voice(
            voice_id=VOICE_CONFIG["voice_id"],
            settings=VOICE_CONFIG["settings"]
        )
        audio_data = elevenlabs_generate(
            text=text_to_speak,
            voice=voice_obj,
            model="eleven_multilingual_v2" # Good general-purpose model
        )
        print(f"ElevenLabs: Audio generated successfully, {len(audio_data)} bytes.")
        encoded_audio = base64.b64encode(audio_data).decode('utf-8')
        # print(f"ElevenLabs: Audio encoded to base64, length: {len(encoded_audio)} chars.") # Can be very long
        return encoded_audio
    except ElevenLabsAPIError as e:
        print(f"ElevenLabs API Error: Status {e.status_code} - {str(e)}")
        # Log the error but return None, frontend will handle missing audio
        return None
    except Exception as e:
        print(f"ElevenLabs: Audio generation error: {type(e).__name__} - {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return None

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # Ensure 'index.html' is in the 'templates' directory
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat_endpoint(request: Request):
    try:
        body = await request.json()
    except json.JSONDecodeError:
        print("API Error: Invalid JSON in request body")
        return JSONResponse(
            {"error": "Invalid JSON payload. Please send valid JSON."},
            status_code=400
        )

    user_id = body.get("user_id")
    user_message_content = body.get("message", "").strip()

    print(f"API Chat Request: User ID: {user_id}, Message (first 50): '{user_message_content[:50]}...'")

    if not user_message_content:
        print("API Error: No message content provided by user.")
        return JSONResponse(
            {"error": "Message content cannot be empty."},
            status_code=400
        )
    if not user_id:
        print("API Error: No user_id (session ID) provided.")
        return JSONResponse(
            {"error": "User ID (session ID) is required for conversation continuity."},
            status_code=400
        )

    # --- Session Management (from backend.memory_store) ---
    # IMPORTANT: As noted before, your `memory_store.memory` dict is for DEVELOPMENT ONLY.
    # For production with multiple workers, use a shared store like Redis.
    if user_id not in memory:
        print(f"Memory: Initializing for new user_id: {user_id}")
        memory[user_id] = {"messages": []}
    
    session_messages = memory[user_id]["messages"]
    session_messages.append({"role": "user", "content": user_message_content})
    memory[user_id]["messages"] = session_messages[-MAX_HISTORY:] 
    print(f"Memory: User '{user_id}' now has {len(memory[user_id]['messages'])} messages in history.")

    # Construct full chat context for OpenAI
    messages_for_api = [{"role": "system", "content": JANET_PROMPT}] + memory[user_id]["messages"]
    print(f"OpenAI: Total messages in context being sent: {len(messages_for_api)}")

    loop = asyncio.get_event_loop()
    assistant_reply_text = None
    audio_base64_data = None
    warning_message = None

    try:
        # --- Call OpenAI (Non-blocking for event loop) ---
        assistant_reply_text = await loop.run_in_executor(
            thread_pool,
            run_openai_call_sync,
            messages_for_api
        )
        print(f"OpenAI: Response received (first 50): '{assistant_reply_text[:50]}...'")

        # Save assistant response to memory
        memory[user_id]["messages"].append({"role": "assistant", "content": assistant_reply_text})
        memory[user_id]["messages"] = memory[user_id]["messages"][-MAX_HISTORY:]
        print(f"Memory: Assistant reply saved. User '{user_id}' history count: {len(memory[user_id]['messages'])}.")

        # --- Generate Audio (Non-blocking for event loop) ---
        if assistant_reply_text:
            audio_base64_data = await loop.run_in_executor(
                thread_pool,
                generate_audio_sync,
                assistant_reply_text
            )
            if not audio_base64_data:
                warning_message = "Janet's voice seems to be unavailable right now, but here's her message."
                print("API Warning: Audio generation failed or returned no data.")
            else:
                print("API Info: Audio generation successful.")
        else: # Should not happen if run_openai_call_sync returns "..."
            print("API Warning: No assistant text to generate audio for.")
            warning_message = "Janet seems to be speechless!"
            assistant_reply_text = "..." # Ensure there's always some text

    except openai.APIError as e:
        print(f"API Error: OpenAI API issue: {str(e)}")
        return JSONResponse(
            {"error": f"Janet is having trouble thinking right now (OpenAI: {e.message}). Please try again."},
            status_code=503 # Service Unavailable
        )
    except Exception as e: # Catch errors from run_openai_call or generate_audio_sync if re-raised
        print(f"API Error: Unhandled exception during backend processing: {type(e).__name__} - {str(e)}")
        print(traceback.format_exc())
        return JSONResponse(
            {"error": "Sorry, Janet is a bit overwhelmed right now. Please try again later."},
            status_code=500 # Internal Server Error
        )

    response_payload = {
        "text": assistant_reply_text,
        "audio": audio_base64_data
    }
    if warning_message:
        response_payload["warning"] = warning_message
    
    print("API: Sending successful response to client.")
    return JSONResponse(response_payload)
