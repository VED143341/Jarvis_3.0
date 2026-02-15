from huggingface_hub import hf_hub_download
import os

print("Creating models directory...")
os.makedirs("models/7B", exist_ok=True)

print("Downloading model... This may take several minutes (2-3 GB)...")

try:
    model_path = hf_hub_download(
        repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
        filename="Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        local_dir="models/7B",
        local_dir_use_symlinks=False
    ) # pyright: ignore[reportCallIssue]
    print(f"\nSuccess! Model downloaded to: {model_path}")
except Exception as e:
    print(f"Error: {e}")