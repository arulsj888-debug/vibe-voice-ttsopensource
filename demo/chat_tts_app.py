"""
Chat TTS App: User text → GPT-4o-mini → VibeVoice TTS → Audio
"""
import os
import copy
import glob
import time
import torch
import tempfile
import numpy as np
from pathlib import Path
from openai import OpenAI
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
# ── Config ──────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL  = os.getenv("OPENAI_MODEL", "gpt-4o-mini-2024-07-18")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set. "
                       "Run: set OPENAI_API_KEY=your-key-here")
MODEL_PATH    = os.getenv("TTS_MODEL_PATH", "microsoft/VibeVoice-Realtime-0.5B")
SPEAKER_NAME  = os.getenv("TTS_SPEAKER", "Carter")
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load TTS model once at startup ──────────────────────────────────────────
print(f"Loading VibeVoice TTS on {DEVICE}...")
from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor

processor = VibeVoiceStreamingProcessor.from_pretrained(MODEL_PATH)

try:
    tts_model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        attn_implementation="flash_attention_2",
    )
except Exception:
    tts_model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        attn_implementation="sdpa",
    )

tts_model.eval()
tts_model.set_ddpm_inference_steps(num_steps=5)

# Load voice preset — prefer exact match, then partial match, then first available
voices_dir = Path(__file__).parent / "voices" / "streaming_model"
voice_files = {Path(f).stem.lower(): f for f in glob.glob(str(voices_dir / "*.pt"))}

def find_voice(name: str) -> str:
    name = name.lower()
    if name in voice_files:
        return voice_files[name]
    # partial match: e.g. "carter" matches "en-carter_man"
    for key, path in voice_files.items():
        if name in key:
            return path
    return list(voice_files.values())[0]

voice_path = find_voice(SPEAKER_NAME)
print(f"Using voice: {voice_path}")
voice_preset = torch.load(voice_path, map_location=DEVICE, weights_only=False)

# ── OpenAI client ────────────────────────────────────────────────────────────
openai_client = OpenAI(api_key=OPENAI_API_KEY)
conversation_history = []

print("✅ Ready!")

# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def gpt_response(user_message: str) -> str:
    conversation_history.append({"role": "user", "content": user_message})
    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful, concise assistant. Keep responses under 3 sentences."},
            *conversation_history,
        ],
        max_tokens=200,
    )
    reply = response.choices[0].message.content.strip()
    conversation_history.append({"role": "assistant", "content": reply})
    return reply


def text_to_speech(text: str, vpreset) -> bytes:
    text = text.replace("'", "'").replace('"', '"').replace('"', '"')
    inputs = processor.process_input_with_cached_prompt(
        text=text,
        cached_prompt=voice_preset,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(DEVICE)

    with torch.no_grad():
        outputs = tts_model.generate(
            **inputs,
            max_new_tokens=None,
            cfg_scale=1.5,
            tokenizer=processor.tokenizer,
            generation_config={"do_sample": False},
            verbose=False,
            all_prefilled_outputs=copy.deepcopy(vpreset),
        )

    # Save to temp file then read bytes
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    processor.save_audio(outputs.speech_outputs[0], output_path=tmp_path)
    with open(tmp_path, "rb") as f:
        audio_bytes = f.read()
    os.unlink(tmp_path)
    return audio_bytes


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


@app.post("/chat")
async def chat(payload: dict):
    user_msg = payload.get("message", "").strip()
    speaker  = payload.get("speaker", SPEAKER_NAME)
    if not user_msg:
        return {"error": "Empty message"}

    # Reload voice if speaker changed
    vpath = find_voice(speaker)
    vpreset = torch.load(vpath, map_location=DEVICE, weights_only=False)

    t0 = time.time()
    gpt_text = gpt_response(user_msg)
    t1 = time.time()
    audio_bytes = text_to_speech(gpt_text, vpreset)
    t2 = time.time()

    import base64
    audio_b64 = base64.b64encode(audio_bytes).decode()
    return {
        "reply": gpt_text,
        "audio_b64": audio_b64,
        "gpt_ms": round((t1 - t0) * 1000),
        "tts_ms": round((t2 - t1) * 1000),
    }


@app.post("/clear")
async def clear_history():
    conversation_history.clear()
    return {"status": "cleared"}


# ── HTML Frontend ─────────────────────────────────────────────────────────────
HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VibeVoice Chat</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', sans-serif; background: #0f0f13; color: #e0e0e0; height: 100vh; display: flex; flex-direction: column; }
  header { padding: 16px 24px; background: #1a1a24; border-bottom: 1px solid #2a2a3a; display: flex; align-items: center; gap: 12px; }
  header h1 { font-size: 1.2rem; font-weight: 600; color: #a78bfa; }
  header span { font-size: 0.8rem; color: #666; }
  #chat { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 16px; }
  .msg { max-width: 75%; padding: 12px 16px; border-radius: 16px; line-height: 1.5; font-size: 0.95rem; }
  .user { align-self: flex-end; background: #4f46e5; color: #fff; border-bottom-right-radius: 4px; }
  .assistant { align-self: flex-start; background: #1e1e2e; border: 1px solid #2a2a3a; border-bottom-left-radius: 4px; }
  .assistant .text { margin-bottom: 8px; }
  .audio-player { width: 100%; height: 36px; }
  .meta { font-size: 0.72rem; color: #555; margin-top: 4px; }
  .thinking { align-self: flex-start; color: #666; font-style: italic; font-size: 0.9rem; animation: pulse 1.2s infinite; }
  @keyframes pulse { 0%,100%{opacity:.4} 50%{opacity:1} }
  #input-area { padding: 16px 20px; background: #1a1a24; border-top: 1px solid #2a2a3a; display: flex; gap: 10px; }
  #msg-input { flex: 1; background: #0f0f13; border: 1px solid #2a2a3a; border-radius: 10px; padding: 12px 16px; color: #e0e0e0; font-size: 0.95rem; outline: none; resize: none; height: 48px; }
  #msg-input:focus { border-color: #4f46e5; }
  button { background: #4f46e5; color: #fff; border: none; border-radius: 10px; padding: 0 20px; cursor: pointer; font-size: 0.95rem; font-weight: 500; transition: background .2s; }
  button:hover { background: #6366f1; }
  button:disabled { background: #333; cursor: not-allowed; }
  #clear-btn { background: #2a2a3a; font-size: 0.8rem; padding: 0 14px; }
  #clear-btn:hover { background: #3a3a4a; }
  .speaker-select { background: #0f0f13; border: 1px solid #2a2a3a; border-radius: 8px; color: #e0e0e0; padding: 4px 8px; font-size: 0.85rem; }
</style>
</head>
<body>
<header>
  <h1>🎙️ VibeVoice Chat</h1>
  <span>GPT-4o-mini → VibeVoice TTS</span>
  <div style="margin-left:auto;display:flex;gap:8px;align-items:center">
    <span style="font-size:0.8rem;color:#666">Speaker:</span>
    <select class="speaker-select" id="speaker-select">
      <option value="Carter">Carter</option>
      <option value="Davis">Davis</option>
      <option value="Emma">Emma</option>
      <option value="Frank">Frank</option>
      <option value="Grace">Grace</option>
      <option value="Mike">Mike</option>
    </select>
  </div>
</header>
<div id="chat">
  <div class="msg assistant"><div class="text">👋 Hi! Ask me anything and I'll respond with voice.</div></div>
</div>
<div id="input-area">
  <textarea id="msg-input" placeholder="Type your message..." rows="1"></textarea>
  <button id="send-btn" onclick="sendMessage()">Send</button>
  <button id="clear-btn" onclick="clearHistory()">Clear</button>
</div>
<script>
const chat = document.getElementById('chat');
const input = document.getElementById('msg-input');
const sendBtn = document.getElementById('send-btn');

input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

function addMsg(role, html) {
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  div.innerHTML = html;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

async function sendMessage() {
  const text = input.value.trim();
  if (!text || sendBtn.disabled) return;
  input.value = '';
  sendBtn.disabled = true;

  addMsg('user', text);
  const thinking = addMsg('thinking', '⏳ Thinking...');

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, speaker: document.getElementById('speaker-select').value })
    });
    const data = await res.json();
    thinking.remove();

    const audioSrc = `data:audio/wav;base64,${data.audio_b64}`;
    addMsg('assistant', `
      <div class="text">${data.reply}</div>
      <audio class="audio-player" controls autoplay src="${audioSrc}"></audio>
      <div class="meta">GPT: ${data.gpt_ms}ms · TTS: ${data.tts_ms}ms</div>
    `);
  } catch(e) {
    thinking.remove();
    addMsg('assistant', `<div class="text" style="color:#f87171">Error: ${e.message}</div>`);
  }
  sendBtn.disabled = false;
  input.focus();
}

async function clearHistory() {
  await fetch('/clear', { method: 'POST' });
  chat.innerHTML = '<div class="msg assistant"><div class="text">🔄 Conversation cleared.</div></div>';
}
</script>
</body>
</html>"""


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
