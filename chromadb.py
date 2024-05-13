import os
import getpass
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
#from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain





# Load environment variables from .env file
load_dotenv()

    # Access the OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY')


os.environ['OPENAI_API_KEY'] = openai_api_key



loader = PyPDFLoader('agripdf.pdf')
pages = loader.load_and_split()
#print(pages[0])
# Load the document, split it into chunks, embed each chunk and load it into the vector store.
raw_documents = pages
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
print(documents[0])
db = FAISS.from_documents(documents, OpenAIEmbeddings())


query = input()
docs = db.similarity_search(query)
print(docs[0].page_content)




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
print(response["answer"])

    

# LangSmith offers several features that can help with testing:...