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

#  extract text from PDF
def summarize_pdf(file):
    st.write("Extracting text from the PDF using pdfplumber...")
    text = ""

    with pdfplumber.open(file) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
            else:
                st.warning(f"Text extraction failed on page {page_number}, using OCR instead.")
                img = page.to_image().original  # Get original image
                text += pytesseract.image_to_string(img)  # OCR extraction

    st.write("Text extraction completed.")
    return text

#  generate summary using LangChain and OpenAI
def generate_summary(text):
    st.write("Generating summary using LangChain and OpenAI...")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API key is not set! Please check your environment variables.")
        return None
    
    # Initialize the OpenAI model
    llm = OpenAI(openai_api_key=api_key)
    
    # Create a prompt template for summarization
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Summarize the following text: {text}",
    )
    
    # Create the LLMChain with the prompt and LLM
    chain = LLMChain(prompt=prompt, llm=llm)  # Use LLMChain instead of RunnableSequence
    
    retries = 200
    wait_time = 120
    
    for attempt in range(retries):
        try:
            # Generate summary using the chain
            summary = chain.run({"text": text})  # Use .run() method with LLMChain
            st.write("Summary generation successful.")
            return summary
        except RateLimitError:
            if attempt < retries - 1:
                st.warning(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                wait_time *= 4
            else:
                st.error("Exceeded retry limit. Please try again later.")
                return None
        except OpenAIError as e:
            st.error(f"API error occurred: {e}")
            return None
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            return None

# Streamlit app setup
def main():
    st.set_page_config(page_title="PDF Summarizing App")
    st.title("PDF Summarizing App")
    st.write("Summarize your PDF files in just a few seconds.")
    st.divider()

    pdf = st.file_uploader("Upload your PDF Document", type="pdf")

    if pdf:
        st.write("PDF file uploaded.")
        submit = st.button("Generate Summary")
        
        if submit:
            st.write("Processing the PDF...")
            text = summarize_pdf(pdf)
            if text.strip():  # Ensure text is not empty
                summary = generate_summary(text)
                if summary:
                    st.subheader("Summary:")
                    st.write(summary)
                else:
                    st.error("Failed to generate summary.")
            else:
                st.error("No text found in PDF.")

if __name__ == "__main__":
    main()
