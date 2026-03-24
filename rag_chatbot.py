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
    # Model Ollama để chat (chạy: ollama list để xem models đã pull)
    "llm_model": "llama3.2",

    # Model embedding (nhẹ, nhanh, chất lượng tốt)
    # Các lựa chọn: nomic-embed-text, mxbai-embed-large, all-minilm
    "embed_model": "nomic-embed-text",

    # Thư mục lưu ChromaDB
    "chroma_dir": "./chroma_db",

    # Thư mục chứa tài liệu cần index
    "docs_dir": "./documents",

    # Số chunk trả về khi tìm kiếm
    "top_k": 4,

    # Kích thước chunk (token ≈ ký tự / 4)
    "chunk_size": 500,
    "chunk_overlap": 50,

    # URL Ollama (mặc định local)
    "ollama_base_url": "http://localhost:11434",
}


def check_ollama():
    """Kiểm tra Ollama đang chạy không."""
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


def build_vectorstore(force_rebuild: bool = False):
    """
    Tạo hoặc load ChromaDB vector store.
    force_rebuild=True sẽ xóa DB cũ và index lại.
    """
    from langchain_ollama import OllamaEmbeddings
    from langchain_chroma import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # Embedding model local qua Ollama
    embeddings = OllamaEmbeddings(
        model=CONFIG["embed_model"],
        base_url=CONFIG["ollama_base_url"],
    )

    chroma_path = CONFIG["chroma_dir"]

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
    """Tạo RAG chain với Ollama LLM (LCEL style)."""
    from langchain_ollama import ChatOllama
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.messages import HumanMessage, AIMessage

    llm = ChatOllama(
        model=CONFIG["llm_model"],
        base_url=CONFIG["ollama_base_url"],
        temperature=0.1,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": CONFIG["top_k"]},
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Bạn là trợ lý AI hữu ích. Hãy trả lời câu hỏi dựa trên ngữ cảnh được cung cấp.
Nếu ngữ cảnh không đủ thông tin, hãy nói rõ và trả lời dựa trên kiến thức của bạn.
Trả lời ngắn gọn, rõ ràng và chính xác. Ưu tiên dùng tiếng Việt nếu câu hỏi bằng tiếng Việt.

Ngữ cảnh:
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    # Lưu chat history thủ công (thay ConversationBufferWindowMemory)
    chat_history = []

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    def invoke(inputs: dict):
        question = inputs["question"]

        # Retrieve
        source_docs = retriever.invoke(question)
        context = format_docs(source_docs)

        # Giới hạn 5 lượt gần nhất (k=5 → 10 messages)
        recent_history = chat_history[-10:]

        # Generate
        response = (prompt | llm | StrOutputParser()).invoke({
            "context": context,
            "chat_history": recent_history,
            "question": question,
        })

        # Cập nhật history
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=response))

        return {"answer": response, "source_documents": source_docs}

    # Gắn method clear để /clear vẫn hoạt động
    invoke.clear = lambda: chat_history.clear()

    return invoke


def show_welcome():
    """Màn hình chào."""
    console.print(Panel.fit(
        "[bold cyan]🤖 Local RAG Chatbot[/bold cyan]\n"
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

    # Kiểm tra Ollama
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
        t = p.add_task("Kiểm tra Ollama...", total=None)
        ok = check_ollama()
        p.update(t, description="✓ Ollama đang chạy" if ok else "✗ Ollama không phản hồi")

    if not ok:
        console.print(Panel(
            "[red]Ollama chưa chạy![/red]\n\n"
            "Khởi động Ollama:\n"
            "  [bold]ollama serve[/bold]\n\n"
            "Pull models cần thiết:\n"
            f"  [bold]ollama pull {CONFIG['llm_model']}[/bold]\n"
            f"  [bold]ollama pull {CONFIG['embed_model']}[/bold]",
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
        q_filter = QuestionFilter(model_path="models/question_filter_model.pkl")
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
