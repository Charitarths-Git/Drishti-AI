"""
download_egoschema.py
=====================
Downloads the massive 100GB+ EgoSchema video benchmark dataset to your
custom cache folder using resumable chunks.

Usage:
    python download_egoschema.py
"""

import os
from huggingface_hub import login, snapshot_download

# Force HuggingFace to use our custom massive-storage directory
CACHE_DIR = "/usershome/cs671_user3/models/datasets"
os.environ["HF_HOME"] = "/usershome/cs671_user3/models"

def main():
    print("\n" + "="*65)
    print("  EGOSCHEMA BENCHMARK DOWNLOADER (~106 GB)")
    print("="*65)
    print(f"  Target directory: {CACHE_DIR}")
    print("  Safe to press Ctrl+C to pause; running this again will resume progress.")
    print("-" * 65 + "\n")
    
    # 1. Login
    token = input("Paste your HuggingFace token (hf_...) and press Enter: ").strip()
    if token:
        login(token=token)
    else:
        print("Note: Proceeding without token. (EgoSchema is public, so this should work).")

    # 2. Download
    print("\nStarting download...")
    try:
        snapshot_download(
            repo_id="lmms-lab/egoschema",
            repo_type="dataset",
            cache_dir=CACHE_DIR,
            resume_download=True,
            max_workers=2, 
        )
        print("\n" + "═" * 65)
        print("  ✅ DOWNLOAD COMPLETE")
        print("  You can now run 'python run_teacher_eval.py' for egoschema!")
        print("═" * 65)
    except KeyboardInterrupt:
        print("\n\n  ⏸️ Download paused. Run the script again to resume.")
    except Exception as e:
        print(f"\n  ❌ Error: {e}")

if __name__ == "__main__":
    main()
