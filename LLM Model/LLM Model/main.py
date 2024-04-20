from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from huggingface_hub import notebook_login
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from langchain import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
import os
import sys

document=[]
for file in os.listdir("docs"):
  if file.endswith(".pdf"):
    pdf_path="./docs/"+file
    loader=PyPDFLoader(pdf_path)
    document.extend(loader.load())
  elif file.endswith('.docx') or file.endswith('.doc'):
    doc_path="./docs/"+file
    loader=Docx2txtLoader(doc_path)
    document.extend(loader.load())
  elif file.endswith('.txt'):
    text_path="./docs/"+file
    loader=TextLoader(text_path)
    document.extend(loader.load())

document_splitter=CharacterTextSplitter(separator='\n', chunk_size=500, chunk_overlap=100)

document_chunks=document_splitter.split_documents(document)

len(document_chunks)

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')


openai_api_key = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings()
embeddings

vectordb=Chroma.from_documents(document_chunks,embedding=embeddings, persist_directory='./data')

vectordb.persist()
llm=ChatOpenAI(temperature=0.7, model_name='gpt-3.5-turbo')


memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True)


#Create our Q/A Chain
pdf_qa=ConversationalRetrievalChain.from_llm(llm=llm,
                                             retriever=vectordb.as_retriever(search_kwargs={'k':6}),
                                             verbose=False, memory=memory)
result=pdf_qa({"question":"How to contact macro"})

print(result['answer'])
