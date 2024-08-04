__import__('pysqlite3') 
import sys 
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import requests

def get_vectorstore_from_url(url, api_key):
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to retrieve URL: {e}")
        return None

    loader = WebBaseLoader(url)
    document = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings(api_key=api_key))

    return vector_store

def get_context_retriever_chain(vector_store, api_key):
    llm = ChatOpenAI(api_key=api_key)
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain
    
def get_conversational_rag_chain(retriever_chain, api_key): 
    llm = ChatOpenAI(api_key=api_key)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input, api_key):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store, api_key)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain, api_key)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    
    return response['answer']

st.set_page_config(page_title="Chat with websites", page_icon="🤖")
st.title("Chat with websites")

with st.sidebar:
    st.header("Settings")
    openai_api_key = st.text_input("Enter your OPENAI KEY", type="password")
    website_url = st.text_input("Website URL")

if not openai_api_key or not website_url:
    st.info("Please enter both your OpenAI key and a website URL.")
else:
    if "current_url" not in st.session_state:
        st.session_state.current_url = ""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [AIMessage(content="Hello, I am a bot. How can I help you?")]

    if website_url and website_url != st.session_state.current_url:
        vector_store = get_vectorstore_from_url(website_url, openai_api_key)
        if vector_store:
            st.session_state.vector_store = vector_store
            st.session_state.current_url = website_url
            st.session_state.chat_history = [AIMessage(content=f"Currently giving you answers from {website_url}")]

    user_query = st.chat_input("Type your message here...")
    if user_query:
        response = get_response(user_query, openai_api_key)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
