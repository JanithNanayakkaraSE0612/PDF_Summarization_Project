import time
import openai
import streamlit as st
import pdfplumber
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv 
from dotenv import load_dotenv
from PIL import Image
import pytesseract

# Load environment variables from .env file
load_dotenv()

# Function to extract text from the PDF file using pdfplumber
def summarize_pdf(file):
    st.write("Extracting text from the PDF using pdfplumber...")
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
            else:
                # If the page doesn't contain extractable text, use OCR
                st.warning(f"Text extraction failed on page {pdf.pages.index(page) + 1}, using OCR instead.")
                img = page.to_image()
                text += pytesseract.image_to_string(img)  # OCR extraction
    st.write("Text extraction completed.")
    return text

# Function to generate a summary using LangChain and OpenAI
def generate_summary(text):
    try:
        st.write("Generating summary using LangChain and OpenAI...")
        # Retrieve your API key from environment variables for security
        openai.api_key = os.getenv("OPENAI_API_KEY")  # Make sure to set this in your environment
        if not openai.api_key:
            st.error("OpenAI API key is not set!")
            return None

        llm = OpenAI(openai_api_key=openai.api_key)
        
        # Customize your prompt
        prompt = PromptTemplate(
            input_variables=["text"],
            template="Summarize the following text: {text}",
        )

        # Create the chain using the LLM and the prompt
        chain = LLMChain(llm=llm, prompt=prompt)
        
        retries = 5
        wait_time = 30  # Initial wait time in seconds
        
        for attempt in range(retries):
            try:
                # Generate summary
                summary = chain.invoke(text)
                st.write("Summary generation successful.")
                return summary
            except openai.RateLimitError:  # Handling RateLimitError directly from openai module
                if attempt < retries - 1:
                    st.warning(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    wait_time *= 5  # Exponentially increase the wait time
                else:
                    st.error("Exceeded retry limit. Please try again later.")
                    return None
            except openai.OpenAIError as e:  # Handling OpenAIError directly from openai module
                st.error(f"API error occurred: {e}")
                return None
            except Exception as e:  # Catching other general errors
                st.error(f"An error occurred: {e}")
                return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Streamlit app setup
def main():
    st.set_page_config(page_title="PDF Summarizing App")

    st.title("PDF Summarizing App")
    st.write("Summarize your PDF files in just a few seconds.")
    st.divider()

    pdf = st.file_uploader('Upload your PDF Document', type='pdf')

    if pdf:
        st.write("PDF file uploaded.")
        submit = st.button("Generate Summary")
        
        if submit:
            # Extract text from PDF
            st.write("Processing the PDF...")
            text = summarize_pdf(pdf)
            
            # Generate summary using LangChain and OpenAI
            summary = generate_summary(text)
            
            if summary:
                st.subheader("Summary:")
                st.write(summary)
            else:
                st.error("Failed to generate summary.")

if __name__ == '__main__':
    main()
