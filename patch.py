import io

with open('rag_chatbot.py', 'r', encoding='utf-8') as f:
    content = f.read()

target = """        # Gọi RAG chain
        with Progress("""

new_code = """        # 1. KIỂM TRA BẰNG QUESTION FILTER TRƯỚC
        if q_filter:
            try:
                is_danger = q_filter.is_dangerous(user_input)
                if is_danger:
                    console.print("[red]⚠️ [CẢNH BÁO] Phát hiện nội dung nhạy cảm, độc hại hoặc nằm trong blocklist (Jailbreak/Prompt Injection). Chatbot từ chối xử lý.[/red]")
                    continue
            except Exception as e:
                pass  # Bỏ qua nếu lỗi lọc

        # 2. Gọi RAG chain
        with Progress("""

if target in content:
    content = content.replace(target, new_code)
    with open('rag_chatbot.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("XONG! Đã nạp thành công.")
else:
    print("KHÔNG TÌM THẤY TARGET.")
