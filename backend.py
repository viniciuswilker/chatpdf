from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
import streamlit as st

from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

folder_files = Path(__file__).parent / 'files'
model_name = 'llama3-8b-8192'

def importa_documentos():
    documentos = []
    for arquivo in folder_files.glob("*.pdf"):
        loader = PyPDFLoader(arquivo)
        documentos_arquivo = loader.load()
        documentos.extend(documentos_arquivo)
    return documentos

def split_documentos(documentos):
    recur_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    documentos = recur_splitter.split_documents(documentos)

    for i, doc in enumerate(documentos):
        doc.metadata['source'] = doc.metadata['source'].split("/")[-1]
        doc.metadata["doc_id"] = i

    return documentos

def cria_vector_store(documentos):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(
        documents=documentos,
        embedding=embedding_model
    )
    return vector_store

def cria_chain_conversa():
    documentos = importa_documentos()
    documentos = split_documentos(documentos)
    vector_store = cria_vector_store(documentos)

    chat = ChatGroq(model=model_name)
    memory = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history",
        output_key="answer"
    )
    retriever = vector_store.as_retriever()

    # Prompt forçando idioma português
    prompt_pt = PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template="""
Você é um assistente de IA que responde perguntas com base no conteúdo de documentos fornecidos.

Sempre responda em português, de forma clara e precisa. Use um tom educado e profissional.

Histórico do chat:
{chat_history}

Contexto dos documentos:
{context}

Pergunta:
{question}
"""
    )

    chat_chain = ConversationalRetrievalChain.from_llm(
        llm=chat,
        memory=memory,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": prompt_pt},
        return_source_documents=True,
        verbose=True
    )

    st.session_state['chain'] = chat_chain
    return chat_chain
