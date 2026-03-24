import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def download_model(repo_id: str, local_dir: str):
    target = Path(local_dir)
    target.mkdir(parents=True, exist_ok=True)
    print(f"[DOWNLOAD] {repo_id} -> {target}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"[DONE] {repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Download local LLM + embedding weights into project weight/ folder (no Ollama)."
    )
    parser.add_argument(
        "--llm_repo",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Hugging Face repo for causal LLM",
    )
    parser.add_argument(
        "--llm_dir",
        default="weight/llm/qwen2.5-1.5b-instruct",
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
    args = parser.parse_args()

    download_model(args.llm_repo, args.llm_dir)
    download_model(args.embed_repo, args.embed_dir)

    print("\nAll local weights are ready.")
    print("Now you can run app without Ollama.")


if __name__ == "__main__":
    main()
