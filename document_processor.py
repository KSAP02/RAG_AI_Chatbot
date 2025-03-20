import io
import fitz
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_text_from_file(uploaded_file):
    print("Processing Document...")
        
    if uploaded_file.name.endswith(".pdf"): # The uploaded file is a BytesIO object not a file path
        
        documents = []
        # Read the uploaded file as bytes stream
        pdf_stream = io.BytesIO(uploaded_file.getvalue())

        doc = fitz.open("pdf", pdf_stream)
        
        text = "\n".join([page.get_text() for page in doc])
        
        documents.append(Document(page_content=text, metadata={"source": uploaded_file.name}))

    return documents

def chunk_text(document):
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    # print(document.page_content)
    chunks = text_splitter.split_documents(document)
    
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [{'source': f'doc_{i}'} for i in range(len(chunks))]
    
    return texts, metadatas
    