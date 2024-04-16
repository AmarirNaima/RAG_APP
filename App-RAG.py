#Helo
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings,HuggingFaceInstructEmbeddings

from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from langchain_community.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

import torch 
from typing import List
from langchain.chains import RetrievalQA
from langchain.llms.ollama import Ollama
from App_temp import css,user_template,bot_template


# Fonction pour charger le modèle de langage (Loading the model
#llm = Ollama(model="orca-mini", temperature=0)


def get_pdf_text(pdf_docs):
    text =""
    #for each pdf pages we loop and extract the text
    for pdf in pdf_docs:
        pdf_reader =PyPDF2.PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
            
    return text

            
    
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100, add_start_index=True
    )
    chunks =text_splitter.split_text(text)
    
    return chunks
    

# function for loading the embedding model
def load_embedding_model(model_path, normalize_embedding=True):
    return HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={'device':'cpu'}, # here we will run the model with CPU only
        encode_kwargs = {
            'normalize_embeddings': normalize_embedding # keep True to compute cosine similarity
        }
    )

# Function for creating embeddings using FAISS
def create_embeddings(chunks, embedding_model, storing_path="vectorstore"):
    
   
    # Creating the embeddings using FAISS
    vectorstore = FAISS.from_texts(chunks, embedding_model)
    
    # Saving the model in current directory
    vectorstore.save_local(storing_path)
    
    # returning the vectorstore
    return vectorstore



def get_converation_chain(vectorstore):
    #the language model 
    llm = Ollama(model="orca-mini", temperature=0)# Chargement du modèle de langage
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # retriever=vectorstore.as_retriever(search_kwargs={'k': 5})
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    ### Testing  if it works
    retrieved_docs = retriever.invoke("What is the file about ?")
    print(len(retrieved_docs))

    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm= llm,
        
        retriever=retriever,
        memory=memory 
    )
    
    return  conversation_chain


def handle_userinput(user_question):
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Chat with your PDFs" ,page_icon=":books:")
    ## Activer les css sur la page 
    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    
    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    
    if user_question:
        handle_userinput(user_question)
        
        
    #st.write(user_template.replace("{{MSG}}", "Hello ROBOT"), unsafe_allow_html=True)
    #st.write(bot_template.replace("{{MSG}}", "Hello HUMAIN"), unsafe_allow_html=True)

    
    
    with st.sidebar:
        st.subheader( "Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        
        if st.button("Process"):
            with st.spinner("Processing"):
            
                # get the pdf files 
                raw_text = get_pdf_text(pdf_docs)
                ## For a QUICK test  ---> st.write(raw_text)
                
                # get the chunks
                text_chunks = get_text_chunks(raw_text)
                ## For a QUICK test  ---> st.write(text_chunks) 
                
                # Loading the Embedding Model to use this you have to import 2 packages:instructorEmbedding and sentence transform
                #embed = HuggingFaceInstructEmbeddings(model_name ="hkunlp/instructor-xl")
                # Loading the Embedding Model
                embed = load_embedding_model(model_path="all-MiniLM-L6-v2")
                
                #creating vectorstore
                vectorstore = create_embeddings(text_chunks, embed)
                st.write(vectorstore)
                # create conversation !!!!
                # take the history of the convesation and retourn the next element of the conversation
                #conversation = get_converation_chain(vectorstore)
                # to make the variable attache to the session (streamlit do the entire update if somthing changes)
                st.session_state.conversation=get_converation_chain(vectorstore)
                
                st.success("done")
    #--------------------------- Display -----------------------------#
    # we want to use the variable conversation in all our App
    #conversation will be available outside of the scoop 
    # st.session_state.conversation  # we have to initialise it first --look up # so we can use it all over our app
    


if __name__ == '__main__':
    main()
    