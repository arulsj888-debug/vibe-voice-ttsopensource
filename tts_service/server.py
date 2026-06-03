"""
VibeVoice TTS Service - runs the model and exposes a REST API.
Start with: python tts_service/server.py
Listens on http://localhost:8001
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import glob
import io
import tempfile
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor

# ── config ──────────────────────────────────────────────────────────────────
MODEL_PATH = "microsoft/VibeVoice-Realtime-0.5B"
VOICES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "demo", "voices", "streaming_model")
DEFAULT_SPEAKER = "en-carter_man"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="VibeVoice TTS Service")

# globals filled at startup
processor = None
model = None
voice_presets: dict = {}


def load_voices():
    presets = {}
    if not os.path.exists(VOICES_DIR):
        return presets
    for pt in glob.glob(os.path.join(VOICES_DIR, "**", "*.pt"), recursive=True):
        name = os.path.splitext(os.path.basename(pt))[0].lower()
        presets[name] = os.path.abspath(pt)
    return dict(sorted(presets.items()))


def get_voice_path(speaker: str, presets: dict) -> str:
    speaker = speaker.lower()
    if speaker in presets:
        return presets[speaker]
    for k, v in presets.items():
        if speaker in k or k in speaker:
            return v
    return list(presets.values())[0]


@app.on_event("startup")
async def startup():
    global processor, model, voice_presets
    print(f"[TTS] Loading model from {MODEL_PATH} on {DEVICE} ...")
    voice_presets = load_voices()
    print(f"[TTS] Found {len(voice_presets)} voices: {', '.join(voice_presets.keys())}")

    processor = VibeVoiceStreamingProcessor.from_pretrained(MODEL_PATH)

    dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    attn = "flash_attention_2" if DEVICE == "cuda" else "sdpa"
    try:
        model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            MODEL_PATH, torch_dtype=dtype, device_map=DEVICE, attn_implementation=attn
        )
    except Exception:
        model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            MODEL_PATH, torch_dtype=dtype, device_map=DEVICE, attn_implementation="sdpa"
        )
    model.eval()
    model.set_ddpm_inference_steps(num_steps=5)
    print("[TTS] Model ready.")


class TTSRequest(BaseModel):
    text: str
    speaker: str = "Carter"


@app.post("/synthesize")
async def synthesize(req: TTSRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded yet")

    text = req.text.replace("'", "'").replace("\u201c", '"').replace("\u201d", '"')
    voice_path = get_voice_path(req.speaker, voice_presets)
    print(f"[TTS] Synthesizing: speaker={req.speaker}, chars={len(text)}")

    cached = torch.load(voice_path, map_location=DEVICE, weights_only=False)
    inputs = processor.process_input_with_cached_prompt(
        text=text, cached_prompt=cached,
        padding=True, return_tensors="pt", return_attention_mask=True,
    )
    inputs = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=None, cfg_scale=1.5,
            tokenizer=processor.tokenizer,
            generation_config={"do_sample": False},
            verbose=False,
            all_prefilled_outputs=copy.deepcopy(cached),
        )

    audio_tensor = outputs.speech_outputs[0]

    # save to temp file then stream back
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    processor.save_audio(audio_tensor, output_path=tmp_path)
    with open(tmp_path, "rb") as f:
        audio_bytes = f.read()
    os.unlink(tmp_path)

    return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/wav",
                             headers={"Content-Disposition": "inline; filename=speech.wav"})


@app.get("/voices")
async def list_voices():
    return {"voices": list(voice_presets.keys())}


@app.get("/health")
async def health():
    return {"status": "ok", "device": DEVICE, "model_loaded": model is not None}


if __name__ == "__main__":
    uvicorn.run("tts_service.server:app", host="0.0.0.0", port=8001, reload=False)
