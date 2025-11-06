import os
import requests
from urllib.parse import urljoin

BASE_URL = "https://wolnelektury.pl"
DATA_DIR = "data/base/train"
AUTHORS = ["henryk-sienkiewicz", "fiodor-dostojewski", "michail-bulhakow"]

def download_author_texts(author):
    api_url = f"{BASE_URL}/api/authors/{author}/books/"
    books = requests.get(api_url).json()

    dir = f"{DATA_DIR}"
    os.makedirs(dir, exist_ok=True)

    for book in books:
        try:
            book_details = requests.get(urljoin(BASE_URL, book["href"])).json()
            text_response = requests.get(urljoin(BASE_URL, book_details["txt"]))
            text_response.raise_for_status()

            filename = f"{book['title']}.txt"
            filepath = os.path.join(dir, filename)

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(text_response.text)

        except Exception as e:
            print(f"Error processing {book.get('title', 'unknown')}: {e}")

if __name__ == "__main__":
    for author in AUTHORS:
        download_author_texts(author)