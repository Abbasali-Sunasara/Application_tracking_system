import fitz  # PyMuPDF

def extract_text_from_pdf(file_stream):
    text = ""
    try:
        with fitz.open(stream=file_stream.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text() + "\n"
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None
    return text