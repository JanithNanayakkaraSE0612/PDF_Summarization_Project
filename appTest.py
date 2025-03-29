import streamlit as st
import pdfplumber
from transformers import pipeline
import time

def extract_text_from_pdf(file):
    """Extracts text from an uploaded PDF file."""
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page_number, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                else:
                    st.warning(f"Skipping page {page_number + 1}: No extractable text found.")
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None

    if not text.strip():
        st.error("The uploaded PDF contains no extractable text.")
        return None

    return text

def summarize_text(text):
    """Summarizes the extracted text using a transformer model."""
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    max_length = 500
    min_length = 100

    chunk_size = 1024  #  model capacity
    text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    summary = []
    for chunk in text_chunks:
        try:
            summarized_chunk = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
            summary.append(summarized_chunk[0]['summary_text'])
        except Exception as e:
            st.error(f"Summarization error: {e}")
    
    return " ".join(summary)

def show_progress(duration=3):
    """Simulates progress while processing."""
    progress_bar = st.progress(0)
    for percent in range(100):
        time.sleep(duration / 100)
        progress_bar.progress(percent + 1)
    progress_bar.empty()

def main():
    st.set_page_config(page_title="PDF Summarizer Version 1", layout="centered")
    st.title("📄 PDF Summarizing Web App")
    st.write("Summarize your PDF files using AI-powered NLP models.")
    
    st.divider()
    
    pdf = st.file_uploader("Upload your PDF Document", type="pdf", accept_multiple_files=False)

    if pdf:
        if st.button("Generate Summary"):
            st.subheader("Extracting Text...")
            show_progress(2)
            extracted_text = extract_text_from_pdf(pdf)

            if extracted_text:
                st.subheader("Summarizing...")
                show_progress(3)
                summary = summarize_text(extracted_text)
                
                st.subheader("Summary:")
                st.write(summary)
                
             
                st.download_button(
                    label="Download Summary as Text File",
                    data=summary,
                    file_name="summary.txt",
                    mime="text/plain"
                )

if __name__ == '__main__':
    main()
