#Document Loaders
#Text Splitting
#Embeddings
#Vector Store
#Environment Variables
#Retrieval QA Chain
import os
import sys
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

import gradio as gr
from dotenv import load_dotenv

print("Loading libraries...")

# Load dotenv first
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

print("Libraries loaded successfully!")

print("Initializing RAG system...")

# Read the document manually
with open('data/data.txt', 'r', encoding='utf8') as f:
    content = f.read()

# Simple text splitting by chunks
chunk_size = 500
chunk_overlap = 150
docs = []
start = 0
while start < len(content):
    end = start + chunk_size
    chunk = content[start:end]
    docs.append(Document(page_content=chunk, metadata={"source": "data.txt"}))
    start = end - chunk_overlap

print(f"Number of documents after splitting: {len(docs)}")

#Embeddings - OpenAI Embeddings
embeddings = OpenAIEmbeddings()

#Vector Store - FAISS
vectorstore = FAISS.from_documents(docs, embedding=embeddings)

#Create retriever
retriever = vectorstore.as_retriever()

#Create prompt template
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

#Create the chain using LCEL
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("RAG system initialized!")

#Function to handle user queries
def answer_question(user_query):
    if not user_query.strip():
        return "Please enter a question."
    try:
        response = chain.invoke(user_query)
        return response
    except Exception as e:
        return f"Error: {str(e)}"

#Create Gradio interface
demo = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(
        label="Enter your question",
        placeholder="Ask a question about the document...",
        lines=2
    ),
    outputs=gr.Textbox(label="Answer", lines=5),
    title="RAG Question Answering System",
    description="Ask questions about the document and get AI-powered answers based on the content.",
    examples=[
        ["What is RAG?"],
        ["Leave & Time Off?"],
        ["Tell me about the policies"]
    ]
)

#Launch the Gradio app
if __name__ == "__main__":
    print(f"Number of documents after splitting: {len(docs)}")
    demo.launch(share=True)
