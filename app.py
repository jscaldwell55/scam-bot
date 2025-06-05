from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import openai
import os
import json
from dotenv import load_dotenv
from backend.memory_store import memory, MAX_HISTORY
from backend.prompts import JANET_PROMPT

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

@app.post("/vapi")
async def vapi_webhook(request: Request):
    body = await request.json()
    user_id = body.get("user_id", "default_user")
    user_message = body.get("transcript", "")

    # Append user message to memory
    memory[user_id]["messages"].append({"role": "user", "content": user_message})
    memory[user_id]["messages"] = memory[user_id]["messages"][-MAX_HISTORY:]

    # Generate assistant response
    full_messages = [{"role": "system", "content": JANET_PROMPT}] + memory[user_id]["messages"]
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=full_messages,
        temperature=0.9,
    )

    assistant_reply = response.choices[0].message.content

    # Save reply to memory
    memory[user_id]["messages"].append({"role": "assistant", "content": assistant_reply})
    memory[user_id]["messages"] = memory[user_id]["messages"][-MAX_HISTORY:]

    return JSONResponse({
        "type": "message",
        "message": assistant_reply
    })


@app.post("/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    user_id = body.get("user_id", "default_user")
    model = body.get("model", "gpt-4o")
    messages = body.get("messages", [])
    temperature = body.get("temperature", 0.9)
    tools = body.get("tools", [])
    tool_choice = body.get("tool_choice", "auto")

    # Pull user history
    history = memory[user_id]["messages"]
    full_messages = [{"role": "system", "content": JANET_PROMPT}] + history + messages

    try:
        # OpenAI call
        response = openai.chat.completions.create(
            model=model,
            messages=full_messages,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
        )
        reply = response.choices[0].message.content or "Sorry, I didn’t catch that."
    except Exception as e:
        print(f"[OpenAI Error] {e}")
        reply = "Hmm... Janet ran into a glitch. Try again?"

    # Store recent messages in memory
    if len(messages) >= 2:
        memory[user_id]["messages"].extend(messages[-2:])
        memory[user_id]["messages"] = memory[user_id]["messages"][-MAX_HISTORY:]

    # Return message (not reply) for Vapi
    return JSONResponse({"message": reply})
