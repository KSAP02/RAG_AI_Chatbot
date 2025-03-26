import os
import faiss
import requests
import numpy as np
from langchain.schema import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

LLM_MODEL = "google/flan-t5-base"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class ChatbotBackend:
    def __init__(self,
                 hf_api_token: str,
                 model_id = LLM_MODEL,
                 embedding_model = EMBEDDING_MODEL):
        
        self.hf_api_token = hf_api_token
        self.model_id = model_id
        self.embedding_model = embedding_model
        
        # Set up API endpoint for the LLM
        self.llm_api_url = f"https://api-inference.huggingface.co/models/{LLM_MODEL}"
        
        # Set up API endpoint for the embedding model
        self.embedding_api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{EMBEDDING_MODEL}"
        
        # Headers for API requests
        self.headers = {
            "Authorization": f"Bearer {hf_api_token}",
            }
        
    def get_embeddings(self, chunks):
        response = requests.post(self.embedding_api_url, headers=self.headers, json={"inputs":chunks}) 
        
        if response.status_code == 200:
            embeddings = response.json()
        else:
            raise Exception(f"Error fetching embeddings: {response.json()}")
        
        return embeddings
    
    def create_vector_store(self, chunks, metadatas):
        
        # print(chunks)
        embeddings = self.get_embeddings(chunks)
        
        dimension = len(embeddings[0])
        faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(np.array(embeddings, dtype=np.float32))
        
        documents_dict = {str(i): Document(page_content=chunks[i], metadata=metadatas[i]) for i in range(len(chunks))}
        
        docstore = InMemoryDocstore(documents_dict)
        
        index_to_docstore_id = {i: str(i) for i in range(len(chunks))}
        
        # Wrap FAISS index with Langchain's FAISS and store vector data of the pdf and store in self var
        self.vector_store = FAISS(
            index=faiss_index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
            embedding_function=lambda x: self.get_embeddings(x)
        )
        
    def retrieve_context(self, query, chunks):
        query_embedding = self.get_embeddings([query])[0]
        
        distances, indices = self.vector_store.index.search(np.array([query_embedding], dtype=np.float32), k=3)
        
        # indices[0] is used and not just indices as an array because indices is a 2d array
        retrieved_texts = [chunks[i] for i in indices[0] if i < len(chunks)]
        
        return "\n\n".join(retrieved_texts) # Return relevant strings as a one complete string
    
    def answer_prompt(self, prompt):
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_length": 512,
                "min_length": 30,
                "temperature": 0.5,
                "num_return_sequences": 1
            }
        }
        
        response = requests.post(self.llm_api_url, headers=self.headers, json=payload)
        
        if response.status_code == 200:
            # print(type((response.json())[0]))
            try:
                response_json = response.json()
                return (response_json[0])
            
            except Exception as e:
                
                print(f"Error processing response :{e}")
                print(f"Raw Response: {response.text}")
                return f"Error: {str(e)}"
        else:
            error_msg = f"Error ({response.status_code}): {response.text}"
            # print(error_msg)
            return error_msg
    
    def get_answer(self, query, chunks, use_context=True):
        # "context" contains relevant text data to the query from the vector database
        context = self.retrieve_context(query, chunks)
        
        # Prompt Engineering
        prompt = f"""Based on the following information, please answer the question throughly.
        INFORMATION:
        {context}
        
        QUESTION:
        {query}
        
        """
        # print(prompt)
        response = self.answer_prompt(prompt)
        # print(f"{response}")
        return response['generated_text']