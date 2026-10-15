import os
import sys
import time
import json
import threading
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

CHUNK_SIZE = 4 * 1024 * 1024  # 4 MB
MAX_WORKERS = 6

def get_cdn_info(hf_url):
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(hf_url, headers=headers, allow_redirects=False, timeout=20)
    if r.status_code in (301, 302, 307, 308):
        cdn_url = r.headers["Location"]
    elif r.status_code == 200:
        cdn_url = hf_url
    else:
        raise RuntimeError(f"Unexpected status code {r.status_code} for {hf_url}")
    
    # Get total size
    r_head = requests.head(cdn_url, headers=headers, timeout=20)
    if "Content-Length" in r_head.headers:
        total_size = int(r_head.headers["Content-Length"])
    else:
        # Fallback to GET with range 0-0
        r_range = requests.get(cdn_url, headers={"Range": "bytes=0-0"}, timeout=20)
        cr = r_range.headers.get("Content-Range", "")
        total_size = int(cr.split("/")[-1])
        
    return cdn_url, total_size

def download_file_parallel(repo_id, filename, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, filename)
    tmp_path = dest_path + ".tmp"
    state_path = dest_path + ".parts.json"
    
    hf_url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    print(f"\n==========================================")
    print(f"[*] Downloading {repo_id}/{filename}")
    print(f"[*] Target File: {dest_path}")
    print(f"==========================================")
    
    cdn_url, total_size = get_cdn_info(hf_url)
    total_mb = total_size / (1024 * 1024)
    print(f"[*] Total Size: {total_mb:.2f} MB ({total_size} bytes)")
    
    if os.path.exists(dest_path) and os.path.getsize(dest_path) == total_size:
        print(f"[+] File already completely downloaded! Skipping.")
        return dest_path
        
    # Initialize tmp file
    if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) != total_size:
        print(f"[*] Pre-allocating sparse file: {tmp_path}")
        with open(tmp_path, "wb") as f:
            f.truncate(total_size)
            
    # Calculate chunk ranges
    num_chunks = (total_size + CHUNK_SIZE - 1) // CHUNK_SIZE
    chunks = []
    for i in range(num_chunks):
        start = i * CHUNK_SIZE
        end = min(total_size - 1, (i + 1) * CHUNK_SIZE - 1)
        chunks.append((i, start, end))
        
    # Load completed state
    completed_chunks = set()
    if os.path.exists(state_path):
        try:
            with open(state_path, "r") as f:
                completed_chunks = set(json.load(f))
            print(f"[*] Resuming: {len(completed_chunks)}/{num_chunks} chunks already done.")
        except Exception:
            completed_chunks = set()
            
    remaining_chunks = [c for c in chunks if c[0] not in completed_chunks]
    if not remaining_chunks:
        print(f"[+] All chunks already marked complete.")
    else:
        print(f"[*] Starting download for remaining {len(remaining_chunks)} chunks with {MAX_WORKERS} threads...")
        
    state_lock = threading.Lock()
    file_lock = threading.Lock()
    active_cdn = {"url": cdn_url, "last_refresh": time.time()}
    
    bytes_downloaded = sum((c[2] - c[1] + 1) for c in chunks if c[0] in completed_chunks)
    t_start = time.time()
    last_print = time.time()
    
    def refresh_cdn_url():
        with state_lock:
            if time.time() - active_cdn["last_refresh"] > 60:
                print(f"[!] Refreshing CDN signed URL...")
                new_url, _ = get_cdn_info(hf_url)
                active_cdn["url"] = new_url
                active_cdn["last_refresh"] = time.time()
            return active_cdn["url"]
            
    def fetch_chunk(chunk_info):
        nonlocal bytes_downloaded, last_print
        chunk_idx, start, end = chunk_info
        chunk_len = end - start + 1
        
        for attempt in range(15):
            url = active_cdn["url"]
            try:
                headers = {"Range": f"bytes={start}-{end}", "User-Agent": "Mozilla/5.0"}
                resp = requests.get(url, headers=headers, timeout=25)
                if resp.status_code in (401, 403):
                    url = refresh_cdn_url()
                    continue
                if resp.status_code not in (200, 206):
                    time.sleep(1)
                    continue
                    
                data = resp.content
                if len(data) != chunk_len:
                    time.sleep(1)
                    continue
                    
                with file_lock:
                    with open(tmp_path, "r+b") as f:
                        f.seek(start)
                        f.write(data)
                        
                with state_lock:
                    completed_chunks.add(chunk_idx)
                    bytes_downloaded += chunk_len
                    now = time.time()
                    if now - last_print >= 5:
                        pct = (bytes_downloaded / total_size) * 100
                        speed = (bytes_downloaded - sum((c[2]-c[1]+1) for c in chunks if c[0] in completed_chunks and c not in remaining_chunks)) / max(1, now - t_start)
                        speed_kb = speed / 1024
                        print(f"[{pct:5.1f}%] {bytes_downloaded/(1024*1024):.1f}/{total_mb:.1f} MB | {len(completed_chunks)}/{num_chunks} chunks | {speed_kb:.1f} KB/s", flush=True)
                        last_print = now
                        with open(state_path, "w") as sf:
                            json.dump(list(completed_chunks), sf)
                return True
            except Exception as e:
                time.sleep(1.5)
                
        raise RuntimeError(f"Chunk {chunk_idx} failed after 15 attempts.")

    if remaining_chunks:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_chunk = {executor.submit(fetch_chunk, c): c for c in remaining_chunks}
            for future in as_completed(future_to_chunk):
                future.result()
                
    # Final state save and atomic move
    with open(state_path, "w") as sf:
        json.dump(list(completed_chunks), sf)
        
    if os.path.getsize(tmp_path) == total_size:
        os.replace(tmp_path, dest_path)
        if os.path.exists(state_path):
            os.remove(state_path)
        print(f"\n[+++] SUCCESSFULLY COMPLETED: {dest_path} ({total_mb:.2f} MB) [+++]\n")
        return dest_path
    else:
        raise RuntimeError("Final file size does not match expected size!")

def main():
    models = [
        ("silma-ai/silma-tts", "model.pt", "models/silma"),
        ("IbrahimSalah/Arabic-F5-TTS-v2", "model_547500_8_18.pt", "models/arabic_f5")
    ]
    for repo_id, filename, dest_dir in models:
        download_file_parallel(repo_id, filename, dest_dir)
    print("\n[********************************************]")
    print("[*] BOTH SILMA AND ARABIC-F5 DOWNLOAD COMPLETE [*]")
    print("[********************************************]\n")

if __name__ == "__main__":
    main()
