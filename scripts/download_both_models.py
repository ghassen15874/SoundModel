import os
import sys
import time

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "60"

from huggingface_hub import hf_hub_download

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

def download_file(repo_id, filename, local_dir, max_attempts=30):
    dest_path = os.path.join(local_dir, filename)
    print(f"\n==========================================")
    print(f"[*] Target: {repo_id} -> {filename}")
    print(f"[*] Destination: {dest_path}")
    print(f"==========================================")
    
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 100 * 1024 * 1024:
        print(f"[+] Already downloaded: {dest_path} ({os.path.getsize(dest_path)/(1024*1024):.2f} MB)")
        return dest_path
        
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"[Attempt {attempt}/{max_attempts}] Downloading {filename}...")
            result = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir,
                resume_download=True
            )
            print(f"[+] Successfully finished: {result}")
            return result
        except Exception as e:
            print(f"[!] Interruption during attempt {attempt}: {e}")
            time.sleep(3)
            
    raise RuntimeError(f"Failed to download {filename} after {max_attempts} attempts.")

def main():
    for item in MODELS_TO_DOWNLOAD:
        download_file(item["repo_id"], item["filename"], item["local_dir"])
    print("\n[+++] ALL REQUESTED MODELS DOWNLOADED SUCCESSFULLY [+++]")

if __name__ == "__main__":
    main()
