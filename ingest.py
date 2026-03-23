"""
ingest.py — Index tài liệu vào ChromaDB
========================================
Dùng script này khi muốn index tài liệu riêng, không cần chạy chatbot.

    python ingest.py                    # index thư mục documents/
    python ingest.py --dir /path/to/docs
    python ingest.py --rebuild          # xóa DB cũ, index lại
"""

import argparse
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress, SpinnerColumn, BarColumn,
    TextColumn, TimeElapsedColumn,
)

console = Console()

# ── Cấu hình ─────────────────────────────────────────────────────────────────
EMBED_MODEL  = "nomic-embed-text"       # ollama pull nomic-embed-text
CHROMA_DIR   = "./chroma_db"
OLLAMA_URL   = "http://localhost:11434"
CHUNK_SIZE   = 500
CHUNK_OVERLAP = 50


def ingest(docs_dir: str, rebuild: bool = False):
    from langchain_ollama import OllamaEmbeddings
    from langchain_chroma import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import (
        PyPDFLoader, TextLoader, DirectoryLoader,
    )

    docs_path = Path(docs_dir)
    if not docs_path.exists():
        console.print(f"[red]Thư mục không tồn tại: {docs_dir}[/red]")
        return

    # Load tài liệu
    console.print(f"\n[cyan]Đang load tài liệu từ[/cyan] [bold]{docs_dir}[/bold]")
    all_docs = []
    stats = {}

    for glob, cls in [("**/*.pdf", PyPDFLoader), ("**/*.txt", TextLoader), ("**/*.md", TextLoader)]:
        loader = DirectoryLoader(docs_dir, glob=glob, loader_cls=cls, silent_errors=True)
        try:
            docs = loader.load()
            stats[glob] = len(docs)
            all_docs.extend(docs)
        except Exception:
            stats[glob] = 0

    for pat, count in stats.items():
        icon = "✓" if count > 0 else "–"
        color = "green" if count > 0 else "dim"
        console.print(f"  [{color}]{icon}[/{color}] {pat}: {count} trang")

    if not all_docs:
        console.print("[yellow]⚠ Không tìm thấy tài liệu nào[/yellow]")
        return

    # Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", ",", " "],
    )
    chunks = splitter.split_documents(all_docs)
    console.print(f"\n[cyan]Tổng cộng:[/cyan] {len(all_docs)} trang → [bold]{len(chunks)} chunks[/bold]")

    # Embedding + Chroma
    console.print(f"\n[cyan]Embedding model:[/cyan] [bold]{EMBED_MODEL}[/bold] (Ollama local)")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)

    if rebuild and Path(CHROMA_DIR).exists():
        import shutil
        shutil.rmtree(CHROMA_DIR)
        console.print(f"[yellow]Đã xóa DB cũ: {CHROMA_DIR}[/yellow]")

    start = time.time()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} chunks"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        # Xử lý từng batch 50 chunks để hiện progress
        task = progress.add_task("Đang tạo embeddings...", total=len(chunks))
        batch_size = 50
        vectorstore = None

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            if vectorstore is None:
                vectorstore = Chroma.from_documents(
                    documents=batch,
                    embedding=embeddings,
                    persist_directory=CHROMA_DIR,
                )
            else:
                vectorstore.add_documents(batch)
            progress.advance(task, len(batch))

    elapsed = time.time() - start
    console.print(Panel(
        f"[green]✓ Đã index [bold]{len(chunks)} chunks[/bold] trong {elapsed:.1f}s[/green]\n"
        f"Lưu tại: [cyan]{CHROMA_DIR}[/cyan]",
        title="Hoàn tất",
        border_style="green",
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index tài liệu vào ChromaDB")
    parser.add_argument("--dir", default="./documents", help="Thư mục tài liệu")
    parser.add_argument("--rebuild", action="store_true", help="Xóa DB cũ, index lại")
    args = parser.parse_args()
    ingest(args.dir, args.rebuild)
