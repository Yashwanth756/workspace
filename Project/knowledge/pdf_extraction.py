"""Gets the text out of the documents."""
import os

import pypdf


def extract_text_from_directory(docs_directory):
    """Read every PDF in the folder, in file-name order.

    Each entry holds document_name, the file name, and document_text, the text
    of all its pages joined by line breaks. A file that is not a PDF is
    skipped. A document with a page that gives up no text is refused, naming
    the file, and so is a folder that holds no PDF.
    """
    pdf_names = sorted(name for name in os.listdir(docs_directory)
                       if name.lower().endswith(".pdf")
                       and os.path.isfile(os.path.join(docs_directory, name)))
    if not pdf_names:
        raise ValueError(f"The folder {docs_directory} holds no PDF document")
    documents = []
    for document_name in pdf_names:
        reader = pypdf.PdfReader(os.path.join(docs_directory, document_name))
        page_texts = []
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if not page_text.strip():
                raise ValueError(f"{document_name}: page {page_number} gives up no text")
            page_texts.append(page_text)
        documents.append({"document_name": document_name, "document_text": "\n".join(page_texts)})
    return documents
    