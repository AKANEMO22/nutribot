import os
import subprocess
import time
import argparse

def run_command(command):
    print(f"\n[RUNNING] {' '.join(command)}")
    result = subprocess.run(command, text=True, capture_output=True, encoding='utf-8')
    if result.returncode != 0:
        print(f"[ERROR] Command failed with error:\n{result.stderr}")
    else:
        print(f"[SUCCESS]\n{result.stdout}")

def main(source_dir=None, source_files=None, source_url=None, min_db_rows=120):
    print("=== BẮT ĐẦU PIPELINE HỆ THỐNG LỌC CÂU HỎI NHẠY CẢM ===")
    
    # 1. Định nghĩa các đường dẫn (File, Model)
    hybrid_data_path = "data/hybrid_dataset.csv"
    db_path = "data/generative_data.db"
    model_path = "weight/question_filter_model.pkl"
    
    # BƯỚC MỚI: Gọi script để Build dữ liệu HYBRID (Trộn Data sinh từ DB + Tải từ File URL)
    print("\n>>> BƯỚC 0: FETCH DỮ LIỆU HYBRID (URL + DATABASE TẠO SINH) <<<")
    build_cmd = [
        "python", "script_download/hybrid_data_builder.py",
        "--db_path", db_path,
        "--output", hybrid_data_path,
        "--min_db_rows", str(min_db_rows),
    ]

    for source_file in (source_files or []):
        build_cmd.extend(["--source_file", source_file])

    if source_dir:
        build_cmd.extend(["--source_dir", source_dir])

    if source_url:
        build_cmd.extend(["--url", source_url])

    run_command(build_cmd)
    
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
    parser = argparse.ArgumentParser(description="End-to-end pipeline for question filtering.")
    parser.add_argument("--source_dir", default="data", help="Directory containing source CSV/JSONL dataset files")
    parser.add_argument(
        "--source_file",
        action="append",
        default=["data/dummy_dataset.csv"],
        help="Source file path (can be repeated), supports CSV/JSONL",
    )
    parser.add_argument("--source_url", default=None, help="Optional remote CSV URL")
    parser.add_argument("--min_db_rows", type=int, default=120, help="Minimum synthetic rows to keep in DB")
    args = parser.parse_args()

    main(
        source_dir=args.source_dir,
        source_files=args.source_file,
        source_url=args.source_url,
        min_db_rows=args.min_db_rows,
    )
