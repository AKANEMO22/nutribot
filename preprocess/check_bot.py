from urllib.robotparser import RobotFileParser
from urllib.parse import urljoin

def can_crawl(base_url: str, path: str) -> bool:
    rp = RobotFileParser()
    rp.set_url(urljoin(base_url, "/robots.txt"))
    rp.read()
    return rp.can_fetch("*", urljoin(base_url, path))

# Kiểm tra trước khi crawl
if can_crawl("https://www.healthline.com/", "/blog/"):
    print("OK, được crawl")
else:
    print("Không được phép")