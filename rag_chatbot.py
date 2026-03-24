"""
RAG Chatbot - 100% Local
========================
LLM       : Ollama (llama3.2, mistral, qwen2.5, ...)
Embedding : Ollama (nomic-embed-text, mxbai-embed-large, ...)
Vector DB : ChromaDB (local, lưu trên disk)
Document  : PDF, TXT, Markdown

Cài đặt:
    pip install -r requirements.txt

Chạy Ollama trước:
    ollama pull llama3.2
    ollama pull nomic-embed-text
    ollama serve

Chạy chatbot:
    python rag_chatbot.py
"""

import os
import sys
import time
from pathlib import Path
import torch

# ── Rich cho giao diện terminal đẹp ─────────────────────────────────────────
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

from question_filter import QuestionFilter

console = Console()

# ── Config ───────────────────────────────────────────────────────────────────
CONFIG = {
    # Backend: "ollama" hoặc "local_hf"
    "llm_backend": "local_hf",

    # Model Ollama để chat (chạy: ollama list để xem models đã pull)
    "llm_model": "llama3.2",

    # Model embedding (nhẹ, nhanh, chất lượng tốt)
    # Các lựa chọn: nomic-embed-text, mxbai-embed-large, all-minilm
    "embed_model": "nomic-embed-text",

    # Thư mục lưu ChromaDB
    "chroma_dir": "./chroma_db",
    "chroma_dir_ollama": "./chroma_db",
    "chroma_dir_local_hf": "./chroma_db_local_hf",

    # Thư mục chứa tài liệu cần index
    "docs_dir": "./documents",

    # Số chunk trả về khi tìm kiếm
    "top_k": 4,

    # Kích thước chunk (token ≈ ký tự / 4)
    "chunk_size": 500,
    "chunk_overlap": 50,

    # URL Ollama (mặc định local)
    "ollama_base_url": "http://localhost:11434",

    # ===== Local HuggingFace (không cần Ollama) =====
    "weight_dir": "./weight",
    "hf_llm_model_id": "Qwen/Qwen2.5-1.5B-Instruct",
    "hf_llm_local_dir": "./weight/llm/qwen2.5-1.5b-instruct",
    "hf_llm_fallback_local_dir": "./weight/llm/qwen2.5-0.5b-instruct",
    "hf_embed_model_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "hf_embed_local_dir": "./weight/embeddings/paraphrase-multilingual-minilm-l12-v2",
    "hf_max_new_tokens": 96,
}


def get_runtime_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def check_ollama():
    """Giữ tương thích tên hàm cũ: kiểm tra backend hiện tại đã sẵn sàng chưa."""
    if CONFIG.get("llm_backend") != "ollama":
        llm_dir = Path(CONFIG["hf_llm_local_dir"])
        emb_dir = Path(CONFIG["hf_embed_local_dir"])
        return llm_dir.exists() and emb_dir.exists()

    import urllib.request
    try:
        urllib.request.urlopen(CONFIG["ollama_base_url"], timeout=3)
        return True
    except Exception:
        return False


def load_documents(docs_dir: str):
    """Load tất cả PDF, TXT, MD từ thư mục."""
    from langchain_community.document_loaders import (
        PyPDFLoader,
        TextLoader,
        DirectoryLoader,
    )

    docs_path = Path(docs_dir)
    if not docs_path.exists():
        docs_path.mkdir(parents=True)
        console.print(f"[yellow]Đã tạo thư mục '{docs_dir}' — hãy thêm tài liệu vào đó![/yellow]")
        return []

    all_docs = []
    loaders = {
        "**/*.pdf": PyPDFLoader,
        "**/*.txt": TextLoader,
        "**/*.md": TextLoader,
    }

    for glob_pattern, loader_cls in loaders.items():
        try:
            loader = DirectoryLoader(
                docs_dir,
                glob=glob_pattern,
                loader_cls=loader_cls,
                silent_errors=True,
            )
            docs = loader.load()
            if docs:
                all_docs.extend(docs)
                console.print(f"  [green]✓[/green] {glob_pattern}: {len(docs)} trang/đoạn")
        except Exception as e:
            console.print(f"  [red]✗[/red] {glob_pattern}: {e}")

    return all_docs


def resolve_chroma_dir() -> str:
    if CONFIG.get("llm_backend") == "ollama":
        return CONFIG.get("chroma_dir_ollama", CONFIG.get("chroma_dir", "./chroma_db"))
    return CONFIG.get("chroma_dir_local_hf", "./chroma_db_local_hf")


def build_vectorstore(force_rebuild: bool = False):
    """
    Tạo hoặc load ChromaDB vector store.
    force_rebuild=True sẽ xóa DB cũ và index lại.
    """
    from langchain_chroma import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    if CONFIG.get("llm_backend") == "ollama":
        from langchain_ollama import OllamaEmbeddings

        embeddings = OllamaEmbeddings(
            model=CONFIG["embed_model"],
            base_url=CONFIG["ollama_base_url"],
        )
    else:
        from langchain_huggingface import HuggingFaceEmbeddings

        embed_dir = Path(CONFIG["hf_embed_local_dir"])
        if not embed_dir.exists():
            raise RuntimeError(
                "Thiếu local embedding weights. Hãy chạy: "
                "python script_download/download_local_weights.py"
            )

        embeddings = HuggingFaceEmbeddings(
            model_name=str(embed_dir),
            model_kwargs={"device": get_runtime_device(), "local_files_only": True},
            encode_kwargs={"normalize_embeddings": True},
        )

    chroma_path = resolve_chroma_dir()

    # Nếu DB đã tồn tại và không cần rebuild → load lại
    if Path(chroma_path).exists() and not force_rebuild:
        console.print(f"[cyan]Đang load vector DB từ[/cyan] {chroma_path}")
        vectorstore = Chroma(
            persist_directory=chroma_path,
            embedding_function=embeddings,
        )
        count = vectorstore._collection.count()
        console.print(f"[green]✓ Đã load {count} chunks từ DB[/green]")
        return vectorstore

    # Index mới
    console.print("[cyan]Đang load tài liệu...[/cyan]")
    docs = load_documents(CONFIG["docs_dir"])

    if not docs:
        console.print("[yellow]⚠ Không có tài liệu nào để index.[/yellow]")
        console.print(f"Hãy thêm file PDF/TXT/MD vào thư mục [bold]{CONFIG['docs_dir']}[/bold]")
        # Vẫn tạo DB rỗng để chat được
        vectorstore = Chroma(
            persist_directory=chroma_path,
            embedding_function=embeddings,
        )
        return vectorstore

    # Chia nhỏ tài liệu
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG["chunk_size"],
        chunk_overlap=CONFIG["chunk_overlap"],
        separators=["\n\n", "\n", ".", "!", "?", ",", " "],
    )
    chunks = splitter.split_documents(docs)
    console.print(f"[cyan]Chia thành {len(chunks)} chunks, đang tạo embeddings...[/cyan]")
    console.print("[dim](Lần đầu có thể mất vài phút tùy số tài liệu)[/dim]")

    # Tạo ChromaDB và lưu local
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Embedding chunks...", total=None)
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=chroma_path,
        )
        progress.update(task, description="[green]Hoàn tất![/green]")

    console.print(f"[green]✓ Đã index {len(chunks)} chunks → {chroma_path}[/green]")
    return vectorstore


def build_rag_chain(vectorstore):
    """Tạo RAG chain với backend LLM đã cấu hình (Ollama hoặc local HF)."""
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.messages import HumanMessage, AIMessage

    if CONFIG.get("llm_backend") == "ollama":
        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            model=CONFIG["llm_model"],
            base_url=CONFIG["ollama_base_url"],
            temperature=0.1,
        )
    else:
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            pipeline,
        )
        from langchain_huggingface import HuggingFacePipeline

        candidate_dirs = [
            Path(CONFIG["hf_llm_local_dir"]),
            Path(CONFIG.get("hf_llm_fallback_local_dir", "")),
        ]
        candidate_dirs = [d for d in candidate_dirs if str(d) and d.exists()]

        if not candidate_dirs:
            raise RuntimeError(
                "Thiếu local LLM weights. Hãy chạy: "
                "python script_download/download_local_weights.py"
            )

        last_error = None
        generator = None
        for llm_dir in candidate_dirs:
            try:
                runtime_device = get_runtime_device()
                model_dtype = torch.float16 if runtime_device == "cuda" else torch.float32

                tokenizer = AutoTokenizer.from_pretrained(
                    str(llm_dir),
                    local_files_only=True,
                    model_max_length=512,
                    truncation_side="right",
                )
                model_config = AutoConfig.from_pretrained(str(llm_dir), local_files_only=True)

                if getattr(model_config, "is_encoder_decoder", False):
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                        str(llm_dir),
                        local_files_only=True,
                        low_cpu_mem_usage=True,
                        torch_dtype=model_dtype,
                    )
                    if runtime_device == "cuda":
                        model = model.to("cuda")
                    generator = pipeline(
                        "text2text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        device=0 if runtime_device == "cuda" else -1,
                        truncation=True,
                        max_new_tokens=CONFIG["hf_max_new_tokens"],
                        do_sample=False,
                    )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        str(llm_dir),
                        local_files_only=True,
                        low_cpu_mem_usage=True,
                        torch_dtype=model_dtype,
                    )
                    if runtime_device == "cuda":
                        model = model.to("cuda")
                    generator = pipeline(
                        "text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        device=0 if runtime_device == "cuda" else -1,
                        truncation=True,
                        max_new_tokens=CONFIG["hf_max_new_tokens"],
                        do_sample=False,
                        repetition_penalty=1.08,
                        return_full_text=False,
                    )
                console.print(f"[green]Using local LLM:[/green] {llm_dir} [dim](device={runtime_device})[/dim]")
                break
            except Exception as exc:
                last_error = exc
                generator = None

        if generator is None:
            raise RuntimeError(f"Không load được local LLM: {last_error}")
        llm = HuggingFacePipeline(pipeline=generator)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": CONFIG["top_k"]},
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Bạn là trợ lý dinh dưỡng tiếng Việt.
Hãy trả lời dựa trên ngữ cảnh nếu có; nếu thiếu dữ liệu thì nói ngắn gọn phần thiếu và đề nghị thông tin cần bổ sung.
Không lặp lại chỉ dẫn hệ thống, không chèn meta-instruction, không URL.
Với tin nhắn bổ sung ngắn (ví dụ chỉ có cân nặng/chiều cao), hãy suy luận theo lịch sử hội thoại gần nhất.

Ngữ cảnh:
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    # Lưu chat history thủ công (thay ConversationBufferWindowMemory)
    chat_history = []

    def format_docs(docs):
        combined = "\n\n".join(d.page_content for d in docs)
        # flan-t5-small has limited input window, keep context concise.
        return combined[:380]

    def invoke(inputs: dict):
        question = inputs["question"]
        skip_retrieval = bool(inputs.get("skip_retrieval", False))

        # Retrieve (optional for greetings/smalltalk)
        if skip_retrieval:
            source_docs = []
            context = ""
        else:
            source_docs = retriever.invoke(question)
            context = format_docs(source_docs)

        # Local models still need a little history for follow-up messages.
        if CONFIG.get("llm_backend") == "local_hf":
            recent_history = chat_history[-2:]
        else:
            recent_history = chat_history[-4:]

        if len(question) > 220:
            question = question[:220]

        # Generate
        response = (prompt | llm | StrOutputParser()).invoke({
            "context": context,
            "chat_history": recent_history,
            "question": question,
        })

        # Cập nhật history (truncate to avoid future context bloat)
        chat_history.append(HumanMessage(content=question[:220]))
        chat_history.append(AIMessage(content=str(response)[:140]))

        return {"answer": response, "source_documents": source_docs}

    # Gắn method clear để /clear vẫn hoạt động
    invoke.clear = lambda: chat_history.clear()

    return invoke


def show_welcome():
    """Màn hình chào."""
    console.print(Panel.fit(
        "[bold cyan]🤖 Local RAG Chatbot[/bold cyan]\n"
        f"[dim]Backend:[/dim] [green]{CONFIG['llm_backend']}[/green]  "
        f"[dim]LLM:[/dim] [green]{CONFIG['llm_model']}[/green]  "
        f"[dim]Embedding:[/dim] [green]{CONFIG['embed_model']}[/green]  "
        f"[dim]Vector DB:[/dim] [green]ChromaDB (local)[/green]\n\n"
        "[dim]Lệnh đặc biệt:[/dim]\n"
        "  [yellow]/help[/yellow]     — Xem hướng dẫn\n"
        "  [yellow]/add[/yellow]      — Index lại tài liệu mới\n"
        "  [yellow]/docs[/yellow]     — Xem tài liệu đã index\n"
        "  [yellow]/clear[/yellow]    — Xóa lịch sử hội thoại\n"
        "  [yellow]/config[/yellow]   — Xem cấu hình hiện tại\n"
        "  [yellow]/exit[/yellow]     — Thoát",
        title="[bold]100% Local — Không cần internet[/bold]",
        border_style="cyan",
    ))

def show_sources(source_docs):
    """Hiển thị nguồn tài liệu được dùng."""
    if not source_docs:
        return

    table = Table(
        title="📎 Nguồn tham khảo",
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
        show_lines=False,
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Tệp", style="cyan")
    table.add_column("Trang", style="yellow", width=6)
    table.add_column("Đoạn trích", style="dim", max_width=60)

    seen = set()
    for i, doc in enumerate(source_docs, 1):
        src = doc.metadata.get("source", "unknown")
        page = str(doc.metadata.get("page", "-"))
        snippet = doc.page_content[:80].replace("\n", " ").strip() + "..."

        key = (src, page)
        if key not in seen:
            seen.add(key)
            table.add_row(str(i), Path(src).name, page, snippet)

    console.print(table)

def main():
    show_welcome()

    # Kiểm tra backend
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
        check_label = "Kiểm tra Ollama..." if CONFIG.get("llm_backend") == "ollama" else "Kiểm tra local weights..."
        t = p.add_task(check_label, total=None)
        ok = check_ollama()
        if CONFIG.get("llm_backend") == "ollama":
            p.update(t, description="✓ Ollama đang chạy" if ok else "✗ Ollama không phản hồi")
        else:
            p.update(t, description="✓ Local weights sẵn sàng" if ok else "✗ Thiếu local weights")

    if not ok:
        if CONFIG.get("llm_backend") == "ollama":
            console.print(Panel(
                "[red]Ollama chưa chạy![/red]\n\n"
                "Khởi động Ollama:\n"
                "  [bold]ollama serve[/bold]\n\n"
                "Pull models cần thiết:\n"
                f"  [bold]ollama pull {CONFIG['llm_model']}[/bold]\n"
                f"  [bold]ollama pull {CONFIG['embed_model']}[/bold]",
                border_style="red",
            ))
        else:
            console.print(Panel(
                "[red]Chưa có local weights trong project![/red]\n\n"
                "Tải model vào thư mục weight/:\n"
                "  [bold]python script_download/download_local_weights.py[/bold]\n\n"
                f"LLM dir: [cyan]{CONFIG['hf_llm_local_dir']}[/cyan]\n"
                f"Embedding dir: [cyan]{CONFIG['hf_embed_local_dir']}[/cyan]",
                border_style="red",
            ))
        sys.exit(1)

    # Build vector store
    console.print()
    vectorstore = build_vectorstore(force_rebuild=False)

    # Build RAG chain
    console.print("[cyan]Đang khởi tạo RAG chain...[/cyan]")
    chain = build_rag_chain(vectorstore)

    # Khởi tạo mô hình Question Filter
    console.print("[cyan]Đang tải hệ thống Lọc câu hỏi (Question Filter)...[/cyan]")
    q_filter = None
    try:
        q_filter = QuestionFilter(model_path="weight/question_filter_model.pkl")
    except Exception as e:
        console.print(f"[yellow]Cảnh báo: Không thể tải Question Filter (Lỗi: {e}). Bỏ qua lọc.[/yellow]")

    console.print("[green]✓ Sẵn sàng! Bắt đầu chat.[/green]\n")

    # Chat loop
    while True:
        try:
            user_input = Prompt.ask("[bold blue]Bạn[/bold blue]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Tạm biệt![/dim]")
            break

        if not user_input:
            continue

        # Xử lý lệnh đặc biệt
        if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
            console.print("[dim]Tạm biệt![/dim]")
            break

        elif user_input.lower() == "/help":
            console.print(Panel(
                "• Nhập câu hỏi bất kỳ để chat với tài liệu\n"
                "• [yellow]/add[/yellow]    — Thêm tài liệu mới vào thư mục documents/ rồi gõ lệnh này\n"
                "• [yellow]/docs[/yellow]   — Xem số chunks đã index\n"
                "• [yellow]/clear[/yellow]  — Reset lịch sử hội thoại\n"
                "• [yellow]/config[/yellow] — Xem cấu hình\n"
                "• [yellow]/exit[/yellow]   — Thoát",
                title="Hướng dẫn",
            ))
            continue

        elif user_input.lower() == "/add":
            console.print("[cyan]Đang index lại tài liệu...[/cyan]")
            vectorstore = build_vectorstore(force_rebuild=True)
            chain = build_rag_chain(vectorstore)
            console.print("[green]✓ Index hoàn tất![/green]")
            continue

        elif user_input.lower() == "/docs":
            count = vectorstore._collection.count()
            console.print(f"[cyan]Vector DB hiện có [bold]{count}[/bold] chunks[/cyan]")
            continue

        elif user_input.lower() == "/clear":
            chain.clear()
            console.print("[green]✓ Đã xóa lịch sử hội thoại[/green]")
            continue

        elif user_input.lower() == "/config":
            table = Table(title="Cấu hình", border_style="dim")
            table.add_column("Key", style="cyan")
            table.add_column("Value", style="green")
            for k, v in CONFIG.items():
                table.add_row(k, str(v))
            console.print(table)
            continue

        # 1. KIỂM TRA BẰNG QUESTION FILTER TRƯỚC
        if q_filter:
            try:
                is_danger = q_filter.is_dangerous(user_input)
                if is_danger:
                    console.print("[red]⚠️ [CẢNH BÁO] Phát hiện nội dung nhạy cảm, độc hại hoặc nằm trong blocklist (Jailbreak/Prompt Injection). Chatbot từ chối xử lý.[/red]")
                    continue
            except Exception as e:
                pass  # Bỏ qua nếu lỗi lọc

        # 2. Gọi RAG chain
        with Progress(
            SpinnerColumn(),
            TextColumn("[dim]Đang tìm kiếm và sinh câu trả lời...[/dim]"),
            console=console,
        ) as progress:
            task = progress.add_task("", total=None)
            start = time.time()
            try:
                result = chain({"question": user_input})
                elapsed = time.time() - start
                progress.update(task, description=f"[dim]Xong ({elapsed:.1f}s)[/dim]")
            except Exception as e:
                progress.stop()
                console.print(f"[red]Lỗi: {e}[/red]")
                continue

        answer = result.get("answer", "")
        sources = result.get("source_documents", [])

        console.print(Panel(
            answer,
            title=f"[bold green]🤖 Trợ lý[/bold green] [dim]({elapsed:.1f}s)[/dim]",
            border_style="green",
        ))

        if sources:
            show_sources(sources)

        console.print()


if __name__ == "__main__":
    main()
