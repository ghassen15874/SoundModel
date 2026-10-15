#!/usr/bin/env python3
"""
Multi-Model Arabic & Darija TTS Studio Server
Specialized for Short Explainer Videos, Reels, & TikTok:
  1. Moroccan Darija & Studio Neural (Jamal & Mouna - 24kHz)
  2. Piper Neural Offline (ONNX 22kHz, Kareem)
  3. Google Neural TTS (gTTS MSA)
  4. Facebook MMS-TTS (VITS 16kHz Offline)
"""

import os
import sys
import uuid
import glob
from flask import Flask, render_template, request, jsonify
from infer import ArabicTTS, NEURAL_VOICES, VOICES_CATALOG, DARIJA_EXPLAINER_PRESETS

app = Flask(__name__, template_folder="templates", static_folder="static")

AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

print("\n" + "=" * 60)
print("🚀 Initializing Darija & Arabic TTS Studio...")
print("=" * 60)
tts_manager = ArabicTTS(preload_mms=False)
print("✓ Studio Engine Ready!\n")


def cleanup_old_files(max_files=100):
    try:
        files = glob.glob(os.path.join(AUDIO_DIR, "*.*"))
        # Exclude explainer demo samples from deletion
        files = [f for f in files if "darija_explainer" not in os.path.basename(f)]
        if len(files) > max_files:
            files.sort(key=os.path.getmtime)
            for f in files[:-max_files]:
                try:
                    os.remove(f)
                except OSError:
                    pass
    except Exception as e:
        print(f"[WARN] Error during cleanup: {e}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/voices")
def get_voices():
    return jsonify({
        "catalog": VOICES_CATALOG,
        "all_voices": [{"id": vid, "name": name} for vid, name in NEURAL_VOICES.items()]
    })


@app.route("/api/presets")
def get_presets():
    return jsonify({"presets": DARIJA_EXPLAINER_PRESETS})


@app.route("/api/generate", methods=["POST"])
def generate_speech():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    engine = data.get("engine", "neural")
    voice = data.get("voice", "ar-MA-JamalNeural")
    speed = float(data.get("speed", 1.10))

    if not text:
        return jsonify({"success": False, "error": "الرجاء كتابة نص التوليد أولاً."}), 400

    ext = "wav" if engine in ["piper", "mms", "silma", "arabic_f5"] else "mp3"
    filename = f"{engine}_{uuid.uuid4().hex[:8]}.{ext}"
    output_path = os.path.join(AUDIO_DIR, filename)

    try:
        if engine == "neural":
            res = tts_manager.synthesize_neural(text, output_path=output_path, voice=voice, speed=speed)
        elif engine == "silma":
            res = tts_manager.synthesize_silma(text, output_path=output_path, speed=speed)
        elif engine == "arabic_f5":
            res = tts_manager.synthesize_arabic_f5(text, output_path=output_path, speed=speed)
        elif engine == "piper":
            res = tts_manager.synthesize_piper(text, output_path=output_path, speed=speed)
        elif engine == "google":
            res = tts_manager.synthesize_google(text, output_path=output_path)
        elif engine == "mms":
            res = tts_manager.synthesize_mms(text, output_path=output_path, speed=speed)
        else:
            return jsonify({"success": False, "error": f"محرك غير معروف: {engine}"}), 400

        cleanup_old_files()

        return jsonify({
            "success": True,
            "engine": engine,
            "model_name": res.get("model_name", engine),
            "voice": voice if engine == "neural" else None,
            "text": text,
            "filename": filename,
            "audio_url": f"/static/audio/{filename}",
            "duration": res["duration"],
            "inference_time": res["inference_time"],
            "sample_rate": res["sample_rate"]
        })
    except Exception as e:
        print(f"[ERROR] Generation failed ({engine}): {e}", flush=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/health")
def health():
    return jsonify({
        "status": "healthy",
        "engines": ["neural", "silma", "arabic_f5", "piper", "google", "mms"],
        "default_voice": "ar-MA-JamalNeural"
    })


@app.route("/api/model_status")
def model_status():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    silma_path = os.path.join(base_dir, "models", "silma", "model.pt")
    silma_parts = os.path.join(base_dir, "models", "silma", "model.pt.parts.json")
    f5_path = os.path.join(base_dir, "models", "arabic_f5", "model_547500_8_18.pt")
    f5_parts = os.path.join(base_dir, "models", "arabic_f5", "model_547500_8_18.pt.parts.json")

    silma_ready = os.path.exists(silma_path) and os.path.getsize(silma_path) > 100 * 1024 * 1024
    silma_pct = 100 if silma_ready else 0
    if not silma_ready and os.path.exists(silma_parts):
        try:
            import json
            d = json.load(open(silma_parts))
            silma_pct = round(len(d) / 621 * 100, 1)
        except Exception:
            pass

    f5_ready = os.path.exists(f5_path) and os.path.getsize(f5_path) > 100 * 1024 * 1024
    f5_pct = 100 if f5_ready else 0
    if not f5_ready and os.path.exists(f5_parts):
        try:
            import json
            d = json.load(open(f5_parts))
            f5_pct = round(len(d) / 1345 * 100, 1)
        except Exception:
            pass

    return jsonify({
        "silma": {"ready": silma_ready, "progress": silma_pct},
        "arabic_f5": {"ready": f5_ready, "progress": f5_pct},
        "neural": {"ready": True, "progress": 100},
        "piper": {"ready": True, "progress": 100},
        "google": {"ready": True, "progress": 100},
        "mms": {"ready": True, "progress": 100}
    })


def run_server():
    port = int(os.environ.get("PORT", 7860))
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("127.0.0.1", port)) == 0:
            print(f"[WARN] Port {port} is occupied, trying {port + 1}...")
            port += 1

    print(f"\n🎧 Darija Explainer Studio running at: http://localhost:{port}")
    print(f"👉 Open in browser: http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    run_server()
