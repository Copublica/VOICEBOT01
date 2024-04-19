

from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

document_directory = "./docs"

os.environ["OPENAI_API_KEY"] = "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"

# Global var for embedding and vec-db
embeddings_openai = None
vectordb_openai = None
document_chunks = None

def load_and_split_documents(directory):
    global document_chunks
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

def initialize():
    global embeddings_openai, vectordb_openai
    embeddings_openai = OpenAIEmbeddings()
    load_and_split_documents(document_directory)
    vectordb_openai = Chroma.from_documents(document_chunks, embedding=embeddings_openai, persist_directory='./data_openai')
    vectordb_openai.persist()

def initialize_conversational_chain():
    llm = ChatOpenAI(temperature=0.7, model_name='gpt-3.5-turbo')
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    pdf_qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectordb_openai.as_retriever(search_kwargs={'k': 6}), verbose=False, memory=memory)
    return pdf_qa

# Initialize embeddings and vector database before first request
initialize()

@app.route('/qa', methods=['POST'])
def question_answer():
    data = request.get_json()
    question = data.get('question', '')
    
    pdf_qa = initialize_conversational_chain()
    
    result = pdf_qa({"question": question})
    answer = result['answer']
    
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
