from pypdf import PdfReader
import os

root = "resume_samples/"

files = os.listdir(root)

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

resumes = []
for f in files:
    if f.endswith('.pdf'):
        text_from_resume = extract_text_from_pdf(f"{root}{f}")
        resumes.append(text_from_resume)

        
