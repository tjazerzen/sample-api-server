import asyncio
import os

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

app = FastAPI()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

class CompletionRequest(BaseModel):
    prompt: str

async def fake_video_streamer():
    for _ in range(10):
        yield b"some fake video bytes\n"
        await asyncio.sleep(1)

@app.get("/check-health")
def check_health():
    return {"message": "Hello, World!"}

@app.get("/stream")
async def stream_video():
    return StreamingResponse(fake_video_streamer())


async def stream_generator(prompt: str):
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for event in stream:
        if event.choices[0].delta.content:
            yield f"data: {event.choices[0].delta.content}\n\n"

@app.post("/complete")
async def complete(request: CompletionRequest):
    return StreamingResponse(stream_generator(request.prompt), media_type="text/event-stream")