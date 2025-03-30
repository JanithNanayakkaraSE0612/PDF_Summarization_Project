import time
import openai
import streamlit as st
import pdfplumber
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains import LLMChain  
from openai import OpenAIError, RateLimitError
import os
from dotenv import load_dotenv
from PIL import Image
import pytesseract


load_dotenv()

def extract_text_from_pdf(file):
    """Extracts text from a PDF using pdfplumber and OCR as fallback."""
    st.write("Extracting text from the PDF...")
    text = ""

    with pdfplumber.open(file) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
            else:
                st.warning(f"Text extraction failed on page {page_number}, using OCR instead.")
                img = page.to_image().original  #  original image
                text += pytesseract.image_to_string(img)  # Perform OCR
    
    st.write("Text extraction completed.")
    return text

def generate_summary(text):
    """Generates a summary of the extracted text using LangChain and OpenAI."""
    st.write("Generating summary...")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API key is missing! Set it in the environment variables.")
        return None
    
    llm = OpenAI(openai_api_key=api_key)
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Summarize the following text:\n{text}"
    )
    chain = LLMChain(prompt=prompt, llm=llm)
    
    retries = 5
    wait_time = 10  
    
    for attempt in range(retries):
        try:
            summary = chain.run({"text": text})
            st.success("Summary generated successfully.")
            return summary
        except RateLimitError:
            if attempt < retries - 1:
                st.warning(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                wait_time *= 2  # Exponential backoff
            else:
                st.error("Rate limit retry attempts exhausted. Please try again later.")
                return None
        except OpenAIError as e:
            st.error(f"OpenAI API error: {e}")
            return None
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            return None

def main():
    """Streamlit app to upload and summarize PDF files."""
    st.set_page_config(page_title="PDF Summarization App")
    st.title("PDF Summarization App")
    st.write("Upload a PDF file and generate a summary in seconds.")
    st.divider()
    
    pdf = st.file_uploader("Upload a PDF file", type="pdf")
    
    if pdf:
        st.write("PDF uploaded successfully.")
        if st.button("Generate Summary"):
            st.write("Processing PDF...")
            text = extract_text_from_pdf(pdf)
            if text.strip():
                summary = generate_summary(text)
                if summary:
                    st.subheader("Summary:")
                    st.write(summary)
                else:
                    st.error("Failed to generate summary.")
            else:
                st.error("No extractable text found in PDF.")

if __name__ == "__main__":
    main()
