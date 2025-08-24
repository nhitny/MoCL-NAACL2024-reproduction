import os
import requests
from tqdm import tqdm

# Hugging Face mirror base URL
base_url = "https://hf-mirror.com/google-bert/bert-base-uncased/resolve/main/"
save_dir = "/workspace/nhitny/mcs/MoCL-NAACL-2024/pretrained/bert-base-uncased1"
os.makedirs(save_dir, exist_ok=True)

file_list = [
    "config.json",
    "pytorch_model.bin",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "README.md",
    "special_tokens_map.json",
]

for fname in file_list:
    url = base_url + fname
    dest = os.path.join(save_dir, fname)
    print(f"⬇️  Downloading: {url}")

    try:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("Content-Length", 0))
            block_size = 1024  # 1 Kibibyte
            t = tqdm(total=total_size, unit="iB", unit_scale=True, desc=fname)

            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        t.update(len(chunk))
            t.close()

            if total_size != 0 and t.n != total_size:
                print(f"⚠️ WARNING: Downloaded size mismatch for {fname}")
            else:
                print(f"✅ Saved: {dest}")
    except Exception as e:
        print(f"❌ Failed: {fname} ({e})")
