"""
Chatbot App - GPT-4o-mini + VibeVoice TTS
Start with: python chatbot_app/server.py
Listens on http://localhost:8000
"""

import os
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import AsyncOpenAI

# ── config ──────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL  = os.getenv("OPENAI_MODEL", "gpt-4o-mini-2024-07-18")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set. "
                       "Run: set OPENAI_API_KEY=your-key-here")
TTS_SERVICE   = os.getenv("TTS_SERVICE_URL", "http://localhost:8001")
TTS_SPEAKER   = os.getenv("TTS_SPEAKER", "Carter")
# ────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Chatbot with TTS")
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# serve static frontend
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


class ChatRequest(BaseModel):
    message: str
    speaker: str = TTS_SPEAKER


@app.get("/")
async def index():
    return FileResponse(os.path.join(static_dir, "index.html"))


@app.post("/chat/text")
async def chat_text(req: ChatRequest):
    """Returns GPT reply as plain text (for debugging)."""
    reply = await get_gpt_reply(req.message)
    return {"reply": reply}


@app.post("/chat/audio")
async def chat_audio(req: ChatRequest):
    """Returns GPT reply synthesized as WAV audio."""
    reply = await get_gpt_reply(req.message)

    # call TTS service
    async with httpx.AsyncClient(timeout=120) as client:
        try:
            tts_resp = await client.post(
                f"{TTS_SERVICE}/synthesize",
                json={"text": reply, "speaker": req.speaker},
            )
            tts_resp.raise_for_status()
        except httpx.ConnectError:
            raise HTTPException(503, "TTS service is not running. Start tts_service/server.py first.")
        except Exception as e:
            raise HTTPException(500, f"TTS error: {e}")

    return StreamingResponse(
        iter([tts_resp.content]),
        media_type="audio/wav",
        headers={
            "X-Reply-Text": reply[:200],   # send text in header too
            "Content-Disposition": "inline; filename=reply.wav",
        },
    )


@app.get("/tts/voices")
async def voices():
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            r = await client.get(f"{TTS_SERVICE}/voices")
            return r.json()
        except Exception:
            return {"voices": [], "error": "TTS service unavailable"}


async def get_gpt_reply(user_message: str) -> str:
    response = await openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful, concise voice assistant. Keep replies under 3 sentences."},
            {"role": "user", "content": user_message},
        ],
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    uvicorn.run("chatbot_app.server:app", host="0.0.0.0", port=8000, reload=False)
