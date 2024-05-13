import streamlit as st
import sqlite3
import openai
import os
import getpass
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter



st.write("# AKAI CHAT")
col1,buff, col2 = st.columns([7,0.5,4])

# Connect to SQLite database (or create if not exists)
def get_connection():
    return sqlite3.connect('history.db')

#delete history
def del_history():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM articles")
    conn.commit()
    col2.write("successfully deleted")
    conn.close()

def get_analysis(document):
    load_dotenv()
    # Access the OpenAI API key
    openai_api_key = os.getenv('OPENAI_API_KEY')
    os.environ['OPENAI_API_KEY'] = openai_api_key

    loader = UnstructuredURLLoader(urls=document)
    data = loader.load()
    #print(pages[0])
    # Load the document, split it into chunks, embed each chunk and load it into the vector store.
    
    chunk_size = 500
    chunk_overlap = 30
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

# Split
    splits = text_splitter.split_documents(data)
    
    db = FAISS.from_documents(splits, OpenAIEmbeddings())


    return db

def get_response(query,db):   
    #print(docs[0].page_content)

    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
    Question: {input}
    <context>
    {context}
    </context>

    """)
    llm=ChatOpenAI()

    document_chain = create_stuff_documents_chain(llm, prompt)

    retriever = db.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({"input":query})
    #print(response["answer"])
    return  response["answer"]

with col1:
    # File uploader widget
    uploaded_file = st.text_input("Upload a file")

    my_button = st.button("SUBMIT")
    prompt = st.chat_input('ask your question',key='store')
    toggle_button = st.checkbox("Add to history")

    # Check if file is uploaded
    if 'analysis' not in st.session_state:
        st.session_state.analysis = None
        st.session_state.answer=None

    if my_button:
        st.session_state.analysis = get_analysis(uploaded_file)
        st.write('Content  analysed successfully')

    if prompt:
        if st.session_state.analysis is not None:
            st.session_state.answer = get_response(prompt, st.session_state.analysis)
            st.write(st.session_state.answer)
            if toggle_button:
                conn = get_connection()
                cursor = conn.cursor()
                cursor.execute("INSERT INTO articles (title, content) VALUES (?, ?)", (prompt,st.session_state.answer))
                conn.commit()
                conn.close()
        else:
            st.write("Please upload a file and analyze it before asking a question.")

with col2:
    st.write('### History')
    his_del_button = st.button('CLEAR HISTORY', on_click=del_history)

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT title, content FROM articles")
    my_var = cursor.fetchall()
    column2_values = [row[1] for row in my_var]

    for ans in reversed(column2_values):
        with col2:
            st.write('response :',ans)
    conn.close()




