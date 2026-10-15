#!/usr/bin/env python3
"""
Advanced Arabic & Darija Text-to-Speech (TTS) Studio Suite
Designed for Short Explainer Videos, TikTok, Reels, & Content Creation.
"""

import os
import sys
import time
import asyncio
import warnings
import argparse
import subprocess

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import scipy.io.wavfile as wavfile
import soundfile as sf
import torch
import torchaudio

def _sf_load(uri, frame_offset=0, num_frames=-1, normalize=True, channels_first=True, **kw):
    data, sr = sf.read(uri, always_2d=True, dtype='float32')
    tensor = torch.from_numpy(data)
    if channels_first:
        tensor = tensor.t()
    return tensor, sr

def _sf_save(uri, src, sample_rate, channels_first=True, **kw):
    if channels_first and src.ndim > 1:
        src = src.t()
    data = src.detach().cpu().numpy()
    sf.write(uri, data, sample_rate)

torchaudio.load = _sf_load
torchaudio.save = _sf_save

from transformers import VitsModel, AutoTokenizer
import edge_tts
from gtts import gTTS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPER_MODEL_PATH = os.path.join(BASE_DIR, "models", "piper", "ar_JO-kareem-medium.onnx")

# Voices categorized with Darija/Maghrebi featured prominently
VOICES_CATALOG = {
    "darija": {
        "ar-MA-JamalNeural": "🇲🇦 جمال (دارجة مغربية - رجالي طبيعي للشروحات والفيديوهات)",
        "ar-MA-MounaNeural": "🇲🇦 مونا (دارجة مغربية - نسائي عفوي للمحتوى القصير)",
        "ar-DZ-IsmaelNeural": "🇩🇿 إسماعيل (لهجة جزائرية/مغاربية - رجالي)",
        "ar-DZ-AminaNeural": "🇩🇿 أمينة (لهجة جزائرية/مغاربية - نسائي)",
        "ar-TN-HediNeural": "🇹🇳 هادي (لهجة تونسية/مغاربية - رجالي)",
        "ar-TN-ReemNeural": "🇹🇳 ريم (لهجة تونسية/مغاربية - نسائي)"
    },
    "mashreq": {
        "ar-SA-HamedNeural": "🇸🇦 حامد (سعودي فخم ورسمي - رجالي)",
        "ar-SA-ZariyahNeural": "🇸🇦 زارية (سعودي نقي - نسائي)",
        "ar-EG-SalmaNeural": "🇪🇬 سلمى (مصري جذاب - نسائي)",
        "ar-EG-ShakirNeural": "🇪🇬 شاكر (مصري هادئ - رجالي)",
        "ar-AE-HamdanNeural": "🇦🇪 حمدان (إماراتي رسمي - رجالي)",
        "ar-AE-FatimaNeural": "🇦🇪 فاطمة (إماراتي راقي - نسائي)",
        "ar-JO-TaimNeural": "🇯🇴 تيم (أردني - رجالي)",
        "ar-JO-SanaNeural": "🇯🇴 سناء (أردني - نسائي)"
    }
}

# Flatten for quick lookup
NEURAL_VOICES = {}
for category, voices in VOICES_CATALOG.items():
    NEURAL_VOICES.update(voices)


DARIJA_EXPLAINER_PRESETS = [
    {
        "title": "🔥 مقدمة تشويقية لفيديو قصير (Hook)",
        "text": "سلام عليكم الدراري! فهاد الفيديو القصير غادي نوريكم واحد البلان واعر بزاف اللي غادي يسهل عليكم الخدمة!"
    },
    {
        "title": "💡 شرح خطوة بخطوة (Step-by-Step)",
        "text": "أول حاجة كديرها هي كتدخل لهاد السيت، كتكليكي هنايا، ومن بعد كتخرج ليك النتيجة ناضية فثواني معدودة بلا تعقيد."
    },
    {
        "title": "🚀 سر أو نصيحة سريعة (Quick Tip)",
        "text": "واش كنتي عارف بلي تقدر تصاوب هادشي كامل بالذكاء الاصطناعي فدقيقة وحدة؟ تبع معايا حتى اللخر باش تفهم مزيان!"
    },
    {
        "title": "🎯 خاتمة ودعوة للمتابعة (Call to Action)",
        "text": "إلا عجبك هاد الفيديو وبغيتي تشوف شروحات خرى بحال هادي، دير جيم وبارطاجي مع صحابك وما تنساش تابونا فلاشين!"
    }
]


class ArabicTTS:
    def __init__(self, mms_model_id: str = "facebook/mms-tts-ara", device: str = None, preload_mms: bool = False):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mms_model_id = mms_model_id
        self.mms_tokenizer = None
        self.mms_model = None
        self.mms_sample_rate = 16000
        
        if preload_mms:
            self._init_mms()

    def _init_mms(self):
        if self.mms_model is None:
            print(f"[INFO] Initializing Facebook MMS-TTS ({self.mms_model_id}) on {self.device}...", flush=True)
            t0 = time.time()
            try:
                self.mms_tokenizer = AutoTokenizer.from_pretrained(self.mms_model_id, local_files_only=True)
                self.mms_model = VitsModel.from_pretrained(self.mms_model_id, local_files_only=True).to(self.device)
            except Exception:
                self.mms_tokenizer = AutoTokenizer.from_pretrained(self.mms_model_id)
                self.mms_model = VitsModel.from_pretrained(self.mms_model_id).to(self.device)
            self.mms_model.eval()
            self.mms_sample_rate = self.mms_model.config.sampling_rate
            print(f"[INFO] Facebook MMS-TTS ready in {time.time()-t0:.2f}s! Native rate: {self.mms_sample_rate} Hz", flush=True)

    def synthesize_mms(self, text: str, output_path: str = None, speed: float = 1.0) -> dict:
        self._init_mms()
        text = text.strip()
        if not text:
            raise ValueError("Input text cannot be empty.")
            
        t_start = time.time()
        inputs = self.mms_tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            output = self.mms_model(**inputs).waveform
            
        waveform = output.squeeze().cpu().numpy()
        inference_time = time.time() - t_start
        
        effective_rate = self.mms_sample_rate
        if speed != 1.0 and 0.5 <= speed <= 2.0:
            effective_rate = int(self.mms_sample_rate * speed)
            
        duration = len(waveform) / effective_rate
        
        if not output_path:
            output_path = "output_mms.wav"
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        wavfile.write(output_path, rate=effective_rate, data=waveform)
            
        return {
            "engine": "mms",
            "model_name": "Facebook MMS-TTS (VITS)",
            "duration": duration,
            "inference_time": inference_time,
            "sample_rate": effective_rate,
            "output_path": output_path
        }

    async def synthesize_neural_async(self, text: str, output_path: str = None, voice: str = "ar-MA-JamalNeural", speed: float = 1.10) -> dict:
        """
        Synthesize using Azure Neural Studio.
        Default voice is ar-MA-JamalNeural (Moroccan Darija Male) at 1.10x speed.
        """
        text = text.strip()
        if not text:
            raise ValueError("Input text cannot be empty.")
            
        t_start = time.time()
        rate_pct = int((speed - 1.0) * 100)
        rate_str = f"+{rate_pct}%" if rate_pct >= 0 else f"{rate_pct}%"
        
        communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate_str)
        
        if not output_path:
            output_path = "output_neural.mp3"
            
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        await communicate.save(output_path)
        inference_time = time.time() - t_start
        
        file_size = os.path.getsize(output_path)
        duration = max(1.0, file_size / (48000 / 8))
        
        voice_desc = NEURAL_VOICES.get(voice, voice)
        return {
            "engine": "neural",
            "model_name": f"{voice_desc}",
            "voice": voice,
            "duration": duration,
            "inference_time": inference_time,
            "sample_rate": 24000,
            "output_path": output_path
        }

    def synthesize_neural(self, text: str, output_path: str = None, voice: str = "ar-MA-JamalNeural", speed: float = 1.10) -> dict:
        return asyncio.run(self.synthesize_neural_async(text, output_path, voice=voice, speed=speed))

    def synthesize_piper(self, text: str, output_path: str = None, speed: float = 1.0) -> dict:
        text = text.strip()
        if not text:
            raise ValueError("Input text cannot be empty.")
            
        if not os.path.exists(PIPER_MODEL_PATH):
            raise FileNotFoundError(f"Piper model not found at {PIPER_MODEL_PATH}")

        if not output_path:
            output_path = "output_piper.wav"
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        piper_bin = os.path.join(BASE_DIR, "venv", "bin", "piper")
        if not os.path.exists(piper_bin):
            piper_bin = "piper"

        t_start = time.time()
        cmd = [
            piper_bin,
            "--model", PIPER_MODEL_PATH,
            "--output_file", output_path
        ]
        
        if speed != 1.0 and 0.5 <= speed <= 2.0:
            cmd.extend(["--length_scale", str(1.0 / speed)])

        subprocess.run(cmd, input=text, text=True, capture_output=True, check=True)
        inference_time = time.time() - t_start

        rate, data = wavfile.read(output_path)
        duration = len(data) / rate

        return {
            "engine": "piper",
            "model_name": "Piper Neural Offline (Kareem - 22kHz)",
            "duration": duration,
            "inference_time": inference_time,
            "sample_rate": rate,
            "output_path": output_path
        }

    def synthesize_google(self, text: str, output_path: str = None) -> dict:
        text = text.strip()
        if not text:
            raise ValueError("Input text cannot be empty.")

        t_start = time.time()
        if not output_path:
            output_path = "output_google.mp3"
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        tts = gTTS(text=text, lang="ar")
        tts.save(output_path)
        inference_time = time.time() - t_start

        file_size = os.path.getsize(output_path)
        duration = max(1.0, file_size / (32000 / 8))

        return {
            "engine": "google",
            "model_name": "Google Neural Arabic (MSA)",
            "duration": duration,
            "inference_time": inference_time,
            "sample_rate": 24000,
            "output_path": output_path
        }

    def synthesize_silma(self, text: str, output_path: str = None, speed: float = 1.0, nfe_step: int = 16) -> dict:
        text = text.strip()
        if not text:
            raise ValueError("Input text cannot be empty.")

        ckpt_file = os.path.join(BASE_DIR, "models", "silma", "model.pt")
        vocab_file = os.path.join(BASE_DIR, "models", "silma", "vocab.txt")
        ref_wav = os.path.join(BASE_DIR, "models", "silma", "ref.wav")
        ref_text = "ويدقق النظر في القرآن الكريم وسائر الكتب السماوية ويتبع مسالك الرسل العظام عليهم الصلاة والسلام."

        if not os.path.exists(ckpt_file) or os.path.getsize(ckpt_file) < 100 * 1024 * 1024:
            raise FileNotFoundError("SILMA TTS checkpoint is still currently downloading from Hugging Face. Please wait a few moments until download completes.")

        if not output_path:
            output_path = "output_silma.wav"
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        t_start = time.time()
        from silma_tts.api import SilmaTTS
        vocos_dir = os.path.join(BASE_DIR, "models", "vocos")
        if not os.path.exists(os.path.join(vocos_dir, "pytorch_model.bin")):
            vocos_dir = None

        if not hasattr(self, "_silma_instance") or self._silma_instance is None:
            print("[INFO] Loading SILMA TTS diffusion model...", flush=True)
            self._silma_instance = SilmaTTS(
                ckpt_file=ckpt_file,
                vocab_file=vocab_file,
                vocoder_local_path=vocos_dir,
                device=self.device,
                enable_normalizer=False,
                force_tashkeel=False
            )

        self._silma_instance.infer(
            ref_file=ref_wav,
            ref_text=ref_text,
            gen_text=text,
            file_wave=output_path,
            speed=speed,
            nfe_step=nfe_step,
            force_tashkeel=False,
            normalize_numbers=False
        )
        inference_time = time.time() - t_start

        rate, data = wavfile.read(output_path)
        duration = len(data) / rate

        return {
            "engine": "silma",
            "model_name": "SILMA TTS v1 (Natural Conversational Arabic Diffusion)",
            "duration": duration,
            "inference_time": inference_time,
            "sample_rate": rate,
            "output_path": output_path
        }

    def synthesize_arabic_f5(self, text: str, output_path: str = None, speed: float = 1.0, nfe_step: int = 16) -> dict:
        text = text.strip()
        if not text:
            raise ValueError("Input text cannot be empty.")

        ckpt_file = os.path.join(BASE_DIR, "models", "arabic_f5", "model_547500_8_18.pt")
        cfg_file = os.path.join(BASE_DIR, "models", "arabic_f5", "F5TTS_Base_8_18.yaml")
        vocab_file = os.path.join(BASE_DIR, "models", "arabic_f5", "vocab.txt")
        ref_wav = os.path.join(BASE_DIR, "models", "arabic_f5", "reference.wav")

        if not os.path.exists(ckpt_file) or os.path.getsize(ckpt_file) < 100 * 1024 * 1024:
            raise FileNotFoundError("Arabic-F5-TTS-v2 checkpoint is still currently downloading from Hugging Face. Please wait a few moments until download completes.")

        if not output_path:
            output_path = "output_arabic_f5.wav"
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        out_dir = os.path.dirname(os.path.abspath(output_path))
        out_name = os.path.basename(output_path)

        f5_bin = os.path.join(BASE_DIR, "venv", "bin", "f5-tts_infer-cli")
        if not os.path.exists(f5_bin):
            f5_bin = "f5-tts_infer-cli"

        t_start = time.time()
        cmd = [
            f5_bin,
            "-mc", cfg_file,
            "-p", ckpt_file,
            "-v", vocab_file,
            "-r", ref_wav,
            "-s", "",
            "-t", text,
            "-o", out_dir,
            "-w", out_name,
            "--nfe_step", str(nfe_step),
            "--speed", str(speed)
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        inference_time = time.time() - t_start

        rate, data = wavfile.read(output_path)
        duration = len(data) / rate

        return {
            "engine": "arabic_f5",
            "model_name": "Arabic-F5-TTS-v2 (Natural Human Arabic Speech Cloning)",
            "duration": duration,
            "inference_time": inference_time,
            "sample_rate": rate,
            "output_path": output_path
        }

    def play_audio(self, file_path: str):
        if not os.path.exists(file_path):
            print(f"[ERROR] Audio file not found: {file_path}", flush=True)
            return
            
        players = ["pw-play", "aplay", "paplay", "mpv", "ffplay"]
        for p in players:
            try:
                subprocess.run([p, file_path], check=True)
                return
            except (subprocess.SubprocessError, FileNotFoundError):
                continue
        print(f"[INFO] Saved: {file_path}. Open with your audio player.", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Arabic & Darija Multi-Model TTS Studio")
    parser.add_argument("-t", "--text", type=str, help="Text to synthesize (Darija or Arabic)")
    parser.add_argument("-f", "--file", type=str, help="Path to text file containing text")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output audio file path")
    parser.add_argument("-e", "--engine", choices=["neural", "silma", "arabic_f5", "piper", "google", "mms"], default="neural",
                        help="Engine choice")
    parser.add_argument("-v", "--voice", type=str, default="ar-MA-JamalNeural",
                        help="Voice ID (default: ar-MA-JamalNeural for Moroccan Darija)")
    parser.add_argument("-s", "--speed", type=float, default=1.10,
                        help="Speed factor (default: 1.10 for fast video explainer tempo)")
    parser.add_argument("-p", "--play", action="store_true", help="Play audio after generating")
    parser.add_argument("--darija-demo", action="store_true", help="Generate all Darija video explainer demo samples")
    parser.add_argument("--list-voices", action="store_true", help="List all available voices")

    args = parser.parse_args()

    if args.list_voices:
        print("\n🇲🇦 أصوات الدارجة والمغاربية (Moroccan Darija & Maghrebi):")
        print("-" * 65)
        for vid, desc in VOICES_CATALOG["darija"].items():
            print(f"  • {vid:<24} | {desc}")
        print("\n🌍 أصوات اللهجات المشرقية والفصحى:")
        print("-" * 65)
        for vid, desc in VOICES_CATALOG["mashreq"].items():
            print(f"  • {vid:<24} | {desc}")
        print("-" * 65)
        return

    tts = ArabicTTS()

    if args.darija_demo:
        os.makedirs("output_darija", exist_ok=True)
        print("\n" + "=" * 60)
        print("🎬 Generating Moroccan Darija Explainer Samples...")
        print("=" * 60)
        for i, item in enumerate(DARIJA_EXPLAINER_PRESETS, 1):
            out_file = f"output_darija/darija_explainer_{i}.mp3"
            print(f"\n[{i}/4] {item['title']}")
            print(f"  النص: {item['text']}")
            res = tts.synthesize_neural(item['text'], output_path=out_file, voice="ar-MA-JamalNeural", speed=1.10)
            print(f"  ✓ Saved to: {out_file} (Duration: {res['duration']:.2f}s)")
        print("\n[SUCCESS] All Darija explainer samples generated in output_darija/!")
        return

    text_to_synthesize = args.text
    if not text_to_synthesize and args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text_to_synthesize = f.read().strip()
    if not text_to_synthesize:
        text_to_synthesize = "سلام عليكم! فهاد الفيديو القصير غادي نوريكم واحد البلان واعر بزاف اللي غادي يسهل عليكم الخدمة!"

    print(f"\n[ENGINE] {args.engine.upper()} | [VOICE] {args.voice} | [SPEED] {args.speed}x")
    print(f"[INPUT] {text_to_synthesize}", flush=True)

    if args.engine == "neural":
        res = tts.synthesize_neural(text_to_synthesize, output_path=args.output, voice=args.voice, speed=args.speed)
    elif args.engine == "silma":
        res = tts.synthesize_silma(text_to_synthesize, output_path=args.output, speed=args.speed)
    elif args.engine == "arabic_f5":
        res = tts.synthesize_arabic_f5(text_to_synthesize, output_path=args.output, speed=args.speed)
    elif args.engine == "piper":
        res = tts.synthesize_piper(text_to_synthesize, output_path=args.output, speed=args.speed)
    elif args.engine == "google":
        res = tts.synthesize_google(text_to_synthesize, output_path=args.output)
    else:
        res = tts.synthesize_mms(text_to_synthesize, output_path=args.output, speed=args.speed)

    print(f"[MODEL] {res['model_name']}")
    print(f"[DONE] Saved to: {res['output_path']}")
    print(f"[STATS] Duration: {res['duration']:.2f}s | Gen Time: {res['inference_time']:.2f}s")

    if args.play:
        print("[PLAYING] Playing audio...")
        tts.play_audio(res['output_path'])


if __name__ == "__main__":
    main()
