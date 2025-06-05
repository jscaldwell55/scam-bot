from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import openai
import os
import json
from dotenv import load_dotenv
from memory_store import memory
from prompts import JANET_PROMPT

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

@app.post("/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    user_id = body.get("user_id", "default_user")  # 👈 Use real user_id if available

    model = body.get("model", "gpt-4o")
    messages = body.get("messages", [])
    temperature = body.get("temperature", 0.9)
    tools = body.get("tools", [])
    tool_choice = body.get("tool_choice", "auto")

    # Build full context
    history = memory[user_id]
    full_messages = [{"role": "system", "content": JANET_PROMPT}] + history + messages

    def format_sse(data):
        return f"data: {json.dumps(data)}\n\n"

    def stream_openai():
        response = openai.chat.completions.create(
            model=model,
            messages=full_messages,
            temperature=temperature,
            stream=True,
            tools=tools,
            tool_choice=tool_choice,
        )
        for chunk in response:
            yield format_sse(chunk.dict())
        yield "data: [DONE]\n\n"

    # Save last user + assistant messages
    if len(messages) >= 2:
        memory[user_id].extend(messages[-2:])
        memory[user_id] = memory[user_id][-12:]  # keep memory lean

    return StreamingResponse(stream_openai(), media_type="text/event-stream")
