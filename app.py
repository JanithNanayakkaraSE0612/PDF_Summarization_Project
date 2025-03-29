import streamlit as st
import pdfplumber

#Summerise Function
def summarize_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ""
        for page_number, page in enumerate(pdf.pages):
            try:
              
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            except Exception as e:
             
                st.warning(f"Skipping page {page_number + 1} due to error: {e}")
        
    summarized_text = text[:5000] + "..."  
    return summarized_text
# Central Function
def main():
    st.set_page_config(page_title="PDF Summarizer")
    
    st.title("PDF Summarizing Web App")
    st.write("Summarize your PDF files in just a few seconds.")
    st.divider()
    
    pdf = st.file_uploader('Upload your PDF Document', type='pdf')
    
    if pdf:
        submit = st.button("Generate Summary")
        
        if submit:
            summary = summarize_pdf(pdf)
            st.subheader("Summary:")
            st.write(summary)

if __name__ == '__main__':
    main()
