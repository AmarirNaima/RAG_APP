from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import  TextSplitter, RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()


PDF_FillePath ="Etude_de_Cas.pdf"

##  Chargement et parsing  du document PDF    
loader = PyMuPDFLoader(PDF_FillePath)

Document = loader.load()

##  Decoupage en chunks
#  ici on utilise une simple division de la page par défaut (1000 caractères)
splitter = RecursiveCharacterTextSplitter(chunk_size =1000 , chunk_overlap = 50)

doc_chunks = splitter.split_documents(Document)

##   Embedding des documents dans un vecteur
embedding = OpenAIEmbeddings()

## Creating vectorstor
vectorstore = chroma.from_documents(Document =doc_chunks , embedding=embedding )

## Documents Retrieval
doc_retriever = vectorstore.as_retriever(top_k =20) 




docs =  doc_retriever("C'est quoi la Problematique de notre client") # Recherche d'un texte ("l'homme") dans le corpus
for doc in docs:
    print(f"Page  {doc.metadata['page']} : doc.page_content[:70]...")