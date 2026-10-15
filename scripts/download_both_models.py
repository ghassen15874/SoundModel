import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from scripts.fast_hf_download import download_file_parallel
except ImportError:
    from fast_hf_download import download_file_parallel

MODELS_TO_DOWNLOAD = [
    {
        "repo_id": "silma-ai/silma-tts",
        "filename": "model.pt",
        "local_dir": "models/silma"
    },
    {
        "repo_id": "IbrahimSalah/Arabic-F5-TTS-v2",
        "filename": "model_547500_8_18.pt",
        "local_dir": "models/arabic_f5"
    }
]

def main():
    for item in MODELS_TO_DOWNLOAD:
        download_file_parallel(item["repo_id"], item["filename"], item["local_dir"])
    print("\n[+++] ALL REQUESTED MODELS DOWNLOADED SUCCESSFULLY [+++]")

if __name__ == "__main__":
    main()
