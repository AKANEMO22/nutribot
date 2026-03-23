# 🤖 Local RAG Chatbot

**100% chạy local** — không cần internet, không cần API key.

| Thành phần | Công nghệ |
|---|---|
| LLM | Ollama (`llama3.2`, `mistral`, `qwen2.5`...) |
| Embedding | Ollama (`nomic-embed-text`) |
| Vector DB | ChromaDB (lưu trên disk) |
| Framework | LangChain |

---

## Cài đặt

### 1. Cài Ollama

```bash
# macOS / Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows: tải tại https://ollama.ai/download
```

### 2. Pull models

```bash
# LLM để chat (chọn 1 trong các model sau)
ollama pull llama3.2          # 2GB — nhanh, tốt cho tiếng Anh/Việt
ollama pull qwen2.5           # 4.7GB — tốt cho tiếng Việt
ollama pull mistral           # 4.1GB — cân bằng tốc độ/chất lượng

# Embedding model (bắt buộc)
ollama pull nomic-embed-text  # 274MB — nhẹ, nhanh, chất lượng tốt
```

### 3. Cài Python packages

```bash
pip install -r requirements.txt
```

### 4. Thêm tài liệu

```bash
mkdir documents
# Copy PDF, TXT, MD vào thư mục documents/
cp your_file.pdf documents/
```

---

## Chạy

```bash
# Khởi động Ollama (nếu chưa chạy)
ollama serve

# Index tài liệu (lần đầu)
python ingest.py

# Bắt đầu chat
python rag_chatbot.py
```

---

## Lệnh trong chatbot

| Lệnh | Mô tả |
|---|---|
| `/add` | Index lại tài liệu sau khi thêm file mới |
| `/docs` | Xem số chunks đã index |
| `/clear` | Xóa lịch sử hội thoại |
| `/config` | Xem cấu hình hiện tại |
| `/exit` | Thoát |

---

## Đổi model

Mở `rag_chatbot.py`, sửa phần `CONFIG`:

```python
CONFIG = {
    "llm_model":   "qwen2.5",          # đổi LLM
    "embed_model": "mxbai-embed-large", # đổi embedding
    "top_k": 4,                         # số chunks truy xuất
    "chunk_size": 500,                  # kích thước chunk
}
```

### So sánh embedding models

| Model | Kích thước | Tốc độ | Chất lượng |
|---|---|---|---|
| `nomic-embed-text` | 274MB | ⚡⚡⚡ | ★★★★ |
| `mxbai-embed-large` | 670MB | ⚡⚡ | ★★★★★ |
| `all-minilm` | 46MB | ⚡⚡⚡⚡ | ★★★ |

---

## Cấu trúc thư mục

```
local_rag_chatbot/
├── rag_chatbot.py      # Chatbot chính
├── ingest.py           # Index tài liệu riêng
├── requirements.txt    # Python dependencies
├── documents/          # Đặt PDF/TXT/MD vào đây
└── chroma_db/          # Vector DB (tự tạo)
```

---

## Troubleshooting

**Lỗi: Ollama không phản hồi**
```bash
ollama serve  # chạy lại
```

**Lỗi: Model not found**
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

**Muốn xóa DB và index lại**
```bash
python ingest.py --rebuild
# hoặc gõ /add trong chatbot
```

**Muốn dùng thư mục tài liệu khác**
```bash
python ingest.py --dir /path/to/my/docs
```
