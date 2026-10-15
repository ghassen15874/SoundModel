# 🎙️ Arabic Natural Human Speech TTS Studio
### Multi-Model Platform for Conversational Arabic, Explainer Videos, Reels & Voice Cloning

An integrated AI studio designed to benchmark, test, and generate natural, human-sounding Arabic speech across multiple state-of-the-art text-to-speech architectures. Engineered specifically for content creation, short explainer videos (TikTok, Reels, Shorts), and natural voiceover without robotic artifacts.

---

## 🎧 Supported TTS Engines & Models

| # | Model | Architecture | Sampling Rate | Runtime Mode | Description & Voice Profile |
|---|---|---|---|---|---|
| 1 | **🔥 SILMA TTS v1** | Diffusion / Flow Matching | 24,000 Hz | Local AI | State-of-the-art Arabic diffusion model delivering ultra-realistic, natural human intonation. |
| 2 | **💎 Arabic-F5-TTS-v2** | F5-TTS Diffusion | 24,000 Hz | Local Voice Clone | Advanced second-generation F5-TTS model tailored for Arabic voice cloning and conversational explainer pacing. |
| 3 | **🎙️ Studio Neural HD** | Azure Neural | 24,000 Hz | High-Speed Cloud | Crystal-clear natural human voices covering multiple dialects (Saudi, Egyptian, Moroccan, Emirati, Tunisian, Algerian, Jordanian). |
| 4 | **⚡ Piper Neural ONNX** | VITS ONNX | 22,050 Hz | Local Offline | Lightweight, fast local neural voice ("Kareem") that operates completely without internet. |
| 5 | **🌐 Google Neural** | Google Neural MSA | 24,000 Hz | High-Speed Cloud | Balanced, clear Modern Standard Arabic (MSA) pronunciation. |
| 6 | **🤖 Facebook MMS-TTS** | Meta VITS | 16,000 Hz | Local Offline | Meta's open-source Arabic VITS model cached locally for comparison and baseline testing. |

---

## 🚀 Running the Interactive Web Studio

1. **Activate the Virtual Environment & Launch Server:**
   ```bash
   source venv/bin/activate
   python3 server.py
   ```

2. **Open in your Browser:**
   👉 **http://localhost:7860**

### Web Studio Features:
- **Instant Engine Switching:** Switch between all 6 engines with a single click to compare outputs on the same input text.
- **Live Model Download Badges:** Real-time percentage badges showing background checkpoint progress (`Downloading X%` ➔ `Ready to use ✓`).
- **Interactive Audio Player:** Built-in audio controls with an animated audio visualizer and instant `.mp3`/`.wav` download button.
- **Explainer Presets:** Pre-configured viral hook presets for quick content testing.

---

## 💻 Command Line Interface (CLI)

Generate audio directly from the terminal using `infer.py`:

```bash
# 1. Generate with SILMA TTS v1 (Natural Diffusion)
./venv/bin/python3 infer.py -e silma -t "السلام عليكم، هذا شرح سريع لكيفية استخدام الذكاء الاصطناعي" -p

# 2. Generate with Arabic-F5-TTS-v2 (Natural Voice Cloning)
./venv/bin/python3 infer.py -e arabic_f5 -t "مرحبا بكم في هذا الفيديو التعليمي الجديد" -p

# 3. Generate with Studio Neural HD (e.g. Natural Explainer Voice at 1.10x tempo)
./venv/bin/python3 infer.py -e neural -v ar-MA-JamalNeural -t "سلام عليكم! فهاد الفيديو غادي نوريكم بلان واعر بزاف" -s 1.10 -p

# 4. Generate with Piper ONNX (Offline local neural speech)
./venv/bin/python3 infer.py -e piper -t "هذا نموذج بايبر العصبي المحلي بصوت كريم" -p

# 5. Generate with Google Neural MSA
./venv/bin/python3 infer.py -e google -t "هذه تجربة توليد الصوت العربي عبر محرك جوجل" -p

# 6. Generate with Facebook MMS-TTS
./venv/bin/python3 infer.py -e mms -t "هذا نموذج فيسبوك إم إم إِس" -p

# List all available voices and dialect IDs:
./venv/bin/python3 infer.py --list-voices
```

---

## 📡 Automated Model Downloader & Monitoring

The repository includes a dedicated multi-threaded parallel chunk downloader with automatic resume:

- **Run/Resume Downloader Script:**
  ```bash
  ./venv/bin/python3 scripts/fast_hf_download.py
  ```

- **Watch Download Progress via Terminal:**
  ```bash
  watch -n 1 "python3 -c \"import json, os; d=json.load(open('models/silma/model.pt.parts.json')) if os.path.exists('models/silma/model.pt.parts.json') else []; print('SILMA Progress:', len(d)*4, 'MB / 2482 MB (', round(len(d)/621*100, 1), '%)')\""
  ```
  Or inspect files on disk:
  ```bash
  watch -n 1 "ls -lh models/silma/ models/arabic_f5/"
  ```

---

## 📁 Repository Structure

```text
SoundModel/
├── infer.py                     # Unified inference engine for all 6 TTS models
├── server.py                    # Flask server & REST API endpoints
├── templates/
│   └── index.html               # Responsive Web Studio frontend
├── static/
│   └── audio/                   # Generated audio outputs directory
├── models/
│   ├── silma/                   # SILMA TTS weights (model.pt), config, & vocab
│   ├── arabic_f5/               # Arabic-F5-TTS-v2 weights, YAML config, & vocab
│   ├── piper/                   # Piper ONNX model (ar_JO-kareem-medium.onnx)
│   └── f5_ref/                  # Reference audio wave for voice cloning
├── scripts/
│   ├── fast_hf_download.py      # Resumable multi-threaded chunk downloader
│   ├── download_both_models.py  # Alternative fallback downloader
│   └── monitor.py               # Live terminal status monitor
├── .gitignore                   # Excludes virtual environments & large checkpoints
└── README.md                    # Project documentation (English)
```

---

## 📄 License
This project utilizes open-source models licensed under their respective Hugging Face and Apache/MIT terms.
