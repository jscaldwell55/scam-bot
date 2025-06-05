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

    # Load user memory
    user_data = memory[user_id]
    history = user_data["messages"]
    trust_score = user_data["trust_score"]
    facts = user_data["facts"]

    # Build conversation context
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

    # Save new user and assistant messages to memory
    if len(messages) >= 2:
        user_data["messages"].extend(messages[-2:])
        user_data["messages"] = user_data["messages"][-MAX_HISTORY:]

    return StreamingResponse(stream_openai(), media_type="text/event-stream")
