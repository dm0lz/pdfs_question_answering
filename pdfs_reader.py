import io
from typing import List, Optional
from dataclasses import dataclass
import PyPDF2
import requests


@dataclass
class PDFSReader:
    def download_pdf(self, url: str) -> Optional[io.BytesIO]:
        try:
            headers = {
                "Accept": "*/*",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return io.BytesIO(response.content)
        except Exception as e:
            print(f"Error downloading PDF from {url}: {e}")
            return None

    def read_pdfs_from_urls(self, urls: List[str]) -> List[str]:
        pdf_texts = []
        for url in urls:
            pdf_file = self.download_pdf(url)
            if pdf_file:
                try:
                    reader = PyPDF2.PdfReader(pdf_file)
                    text = "".join(page.extract_text() for page in reader.pages)
                    pdf_texts.append(text)
                except Exception as e:
                    print(f"Error reading PDF from {url}: {e}")
        return pdf_texts
