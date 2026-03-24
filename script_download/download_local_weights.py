import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def has_model_files(local_dir: Path) -> bool:
    if not local_dir.exists():
        return False
    markers = ["config.json", "tokenizer.json", "model.safetensors", "pytorch_model.bin"]
    return any((local_dir / m).exists() for m in markers)


def download_model(repo_id: str, local_dir: str, hf_token: str | None = None):
    target = Path(local_dir)
    target.mkdir(parents=True, exist_ok=True)
    print(f"[DOWNLOAD] {repo_id} -> {target}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target),
        token=hf_token,
    )
    print(f"[DONE] {repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Download local LLM + embedding weights into project weight/ folder (no Ollama)."
    )
    parser.add_argument(
        "--llm_repo",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Hugging Face repo for causal LLM",
    )
    parser.add_argument(
        "--llm_dir",
        default="weight/llm/qwen2.5-0.5b-instruct",
        help="Local directory to save LLM weights",
    )
    parser.add_argument(
        "--embed_repo",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Hugging Face repo for embedding model",
    )
    parser.add_argument(
        "--embed_dir",
        default="weight/embeddings/paraphrase-multilingual-minilm-l12-v2",
        help="Local directory to save embedding weights",
    )
    parser.add_argument(
        "--only",
        choices=["all", "llm", "embed"],
        default="all",
        help="Download only one component or both",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if model files already exist",
    )
    parser.add_argument(
        "--hf_token",
        default=None,
        help="Optional Hugging Face token for gated/private models",
    )
    args = parser.parse_args()

    llm_dir = Path(args.llm_dir)
    embed_dir = Path(args.embed_dir)

    if args.only in ("all", "llm"):
        if args.force or not has_model_files(llm_dir):
            download_model(args.llm_repo, args.llm_dir, args.hf_token)
        else:
            print(f"[SKIP] LLM already exists at {llm_dir}")

    if args.only in ("all", "embed"):
        if args.force or not has_model_files(embed_dir):
            download_model(args.embed_repo, args.embed_dir, args.hf_token)
        else:
            print(f"[SKIP] Embedding model already exists at {embed_dir}")

    print("\nAll local weights are ready.")
    print("Now you can run app without Ollama.")


if __name__ == "__main__":
    main()
