import pdfplumber
import docx
from typing import Optional


class ResumeParser:
    @staticmethod
    def parse_pdf(file_obj) -> str:
        """
        file_obj: file-like object from Flask (request.files['...'])
        """
        text = ""
        with pdfplumber.open(file_obj) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text

    @staticmethod
    def parse_docx(file_obj) -> str:
        """
        file_obj: file-like object
        """
        document = docx.Document(file_obj)
        return "\n".join(p.text for p in document.paragraphs)

    @staticmethod
    def parse(file_obj, filename: str) -> Optional[str]:
        """
        Decide parser based on extension.
        """
        filename = filename.lower()
        if filename.endswith(".pdf"):
            return ResumeParser.parse_pdf(file_obj)
        elif filename.endswith(".docx"):
            return ResumeParser.parse_docx(file_obj)
        else:
            return None
