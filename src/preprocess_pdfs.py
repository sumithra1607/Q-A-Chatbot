
import os
import fitz  # PyMuPDF
import re
import json

# Function to clean text (remove extra spaces/newlines)
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Function to chunk text by words (not characters)
def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []

    start = 0
    chunk_id = 1
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append({
            "chunk_id": chunk_id,
            "content": " ".join(chunk_words)
        })
        start += chunk_size - overlap
        chunk_id += 1
    return chunks

# Process all PDFs in a folder
def process_pdfs(pdf_folder, output_folder="C:/Users/sumit/Downloads/Q&A/data/processed_chunks"):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            doc = fitz.open(pdf_path)

            all_chunks = []

            for page_num, page in enumerate(doc, start=1):
                text = page.get_text()
                text = clean_text(text)
                page_chunks = chunk_text(text, chunk_size=200, overlap=50)

                # add page info to each chunk
                for chunk in page_chunks:
                    chunk["page"] = page_num
                    chunk["file"] = filename
                    all_chunks.append(chunk)

            # Save chunks to JSON
            output_file = os.path.join(output_folder, f"{filename}.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(all_chunks, f, indent=2, ensure_ascii=False)

            print(f"Processed {filename} → {len(all_chunks)} chunks saved in {output_file}")

# Example usage
process_pdfs("C:/Users/sumit/Downloads/Q&A/data/raw_pdfs")  # replace "pdfs" with your folder containing 11 PDFs
