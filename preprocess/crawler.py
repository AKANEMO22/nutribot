import requests
from bs4 import BeautifulSoup

def crawl_single_page(url: str, output_file: str = "output.txt"):
    res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    res.encoding = "utf-8"  # hoặc "utf-8-sig" nếu bị lỗi ký tự
    soup = BeautifulSoup(res.text, "html.parser")

    # Xóa rác trước
    for tag in soup.select("script, style, nav, header, footer, .ads, .sidebar, .comment"):
        tag.decompose()

    # Lấy vùng nội dung chính  ← chỉ cần sửa dòng này
    content = soup.select_one("article")  # hoặc div bao ngoài — inspect thêm 1 cấp

    # Lấy tất cả đoạn văn bản
    paragraphs = content.select("p")
    full_text = "\n\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))

    print(full_text)
    return full_text

text = crawl_single_page("https://www.healthline.com/nutrition/foods-that-make-you-taller")

f = open(r"C:\Users\catchtherainbow\Desktop\nutrious consultant\Data\refer1.txt", "w", encoding="utf-8")

f.write(text)