from pypdf import PdfReader


root = "resume_samples/"

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

text_from_resume = extract_text_from_pdf(f"{root}1Amy.pdf")
print(text_from_resume)