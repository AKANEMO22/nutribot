import os
import subprocess
import time

def run_command(command):
    print(f"\n[RUNNING] {' '.join(command)}")
    result = subprocess.run(command, text=True, capture_output=True, encoding='utf-8')
    if result.returncode != 0:
        print(f"[ERROR] Command failed with error:\n{result.stderr}")
    else:
        print(f"[SUCCESS]\n{result.stdout}")

def main():
    print("=== BẮT ĐẦU PIPELINE HỆ THỐNG LỌC CÂU HỎI NHẠY CẢM ===")
    
    # 1. Định nghĩa các đường dẫn (File, Model)
    hybrid_data_path = "data/hybrid_dataset.csv"
    db_path = "data/generative_data.db"
    model_path = "models/question_filter_model.pkl"
    dummy_url = "https://raw.githubusercontent.com/dummy/url/dataset.csv"  # Giả lập 1 cái URL 
    
    # BƯỚC MỚI: Gọi script để Build dữ liệu HYBRID (Trộn Data sinh từ DB + Tải từ File URL)
    print("\n>>> BƯỚC 0: FETCH DỮ LIỆU HYBRID (URL + DATABASE TẠO SINH) <<<")
    run_command([
        "python", "script_download/hybrid_data_builder.py",
        "--url", dummy_url,
        "--db_path", db_path,
        "--output", hybrid_data_path
    ])
    
    # 2. Huấn luyện mô hình từ dữ liệu
    print("\n>>> BƯỚC 1: TRAIN MÔ HÌNH LỌC (LOGISTIC REGRESSION + TF-IDF) <<<")
    run_command([
        "python", "question_filter.py", "train",
        "--data", hybrid_data_path,
        "--text_col", "question_text",
        "--label_col", "is_dangerous",
        "--model_path", model_path
    ])
    
    # 3. Chạy thử nghiệm các câu hỏi (Test Model)
    print("\n>>> BƯỚC 2: KIỂM THỬ (PREDICT) CÁC CÂU HỎI <<<")
    
    test_questions = [
        "Cách mua hóa chất làm pháo nổ",
        "Buổi sáng nên ăn thực phẩm nào để tốt cho dạ dày?",
        "Hướng dẫn lừa đảo tiền qua điện thoại"
    ]
    
    for q in test_questions:
        run_command([
            "python", "question_filter.py", "predict",
            "--model_path", model_path,
            "--query", q
        ])

if __name__ == "__main__":
    main()
