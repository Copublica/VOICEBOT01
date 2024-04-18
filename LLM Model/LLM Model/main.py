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
from flask import Flask, request, jsonify


# document=[]
# for file in os.listdir("docs"):
#   if file.endswith(".pdf"):
#     pdf_path="./docs/"+file
#     loader=PyPDFLoader(pdf_path)
#     document.extend(loader.load())
#   elif file.endswith('.docx') or file.endswith('.doc'):
#     doc_path="./docs/"+file
#     loader=Docx2txtLoader(doc_path)
#     document.extend(loader.load())
#   elif file.endswith('.txt'):
#     text_path="./docs/"+file
#     loader=TextLoader(text_path)
#     document.extend(loader.load())

# document_splitter=CharacterTextSplitter(separator='\n', chunk_size=500, chunk_overlap=100)

# document_chunks=document_splitter.split_documents(document)

# len(document_chunks)

# embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
# os.environ["OPENAI_API_KEY"]="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"

# embeddings = OpenAIEmbeddings()
# embeddings

# vectordb=Chroma.from_documents(document_chunks,embedding=embeddings, persist_directory='./data')

# vectordb.persist()
# llm=ChatOpenAI(temperature=0.7, model_name='gpt-3.5-turbo')


# memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True)


# #Create our Q/A Chain
# pdf_qa=ConversationalRetrievalChain.from_llm(llm=llm,
#                                              retriever=vectordb.as_retriever(search_kwargs={'k':6}),
#                                              verbose=False, memory=memory)
# result=pdf_qa({"question":"Who are human"})

# print(result['answer'])





# -------------------------------------------------------------------------------
app = Flask(__name__)


document_directory = "./docs"

os.environ["OPENAI_API_KEY"] = "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"

# Load and split documents
def load_and_split_documents(directory):
    document = []
    for file in os.listdir(directory):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(directory, file)
            loader = PyPDFLoader(pdf_path)
            document.extend(loader.load())
        elif file.endswith('.docx') or file.endswith('.doc'):
            doc_path = os.path.join(directory, file)
            loader = Docx2txtLoader(doc_path)
            document.extend(loader.load())
        elif file.endswith('.txt'):
            text_path = os.path.join(directory, file)
            loader = TextLoader(text_path)
            document.extend(loader.load())
    document_splitter = CharacterTextSplitter(separator='\n', chunk_size=500, chunk_overlap=100)
    document_chunks = document_splitter.split_documents(document)
    return document_chunks


def initialize_embeddings_and_vectordb():
    embeddings_huggingface = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    embeddings_openai = OpenAIEmbeddings()
    
  
    document_chunks = load_and_split_documents(document_directory)

    vectordb_huggingface = Chroma.from_documents(document_chunks, embedding=embeddings_huggingface, persist_directory='./data_huggingface')
    vectordb_huggingface.persist()


    vectordb_openai = Chroma.from_documents(document_chunks, embedding=embeddings_openai, persist_directory='./data_openai')
    vectordb_openai.persist()

    return vectordb_huggingface, vectordb_openai


def initialize_conversational_chain(vectordb):
    llm = ChatOpenAI(temperature=0.7, model_name='gpt-3.5-turbo')
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    pdf_qa = ConversationalRetrievalChain.from_llm(llm=llm,
                                                   retriever=vectordb.as_retriever(search_kwargs={'k': 6}),
                                                   verbose=False, memory=memory)
    return pdf_qa

@app.route('/qa', methods=['POST'])
def question_answer():
    data = request.get_json()
    question = data.get('question', '')
    embedding_method = data.get('embedding', 'huggingface').lower()


    vectordb_huggingface, vectordb_openai = initialize_embeddings_and_vectordb()


    if embedding_method == 'huggingface':
        vectordb = vectordb_huggingface
    elif embedding_method == 'openai':
        vectordb = vectordb_openai
    else:
        return jsonify({'error': 'Invalid embedding method specified'})

    pdf_qa = initialize_conversational_chain(vectordb)

    result = pdf_qa({"question": question})
    answer = result['answer']

    return jsonify({'answer': answer})

if __name__ == '__main__':
     app.run(host='0.0.0.0', port=5002, debug=True)
