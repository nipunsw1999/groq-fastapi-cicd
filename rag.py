from dotenv import load_dotenv
import os 
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import  FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.tools.retriever import create_retriever_tool 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_openai_tools_agent
from langchain import hub
from langchain.agents import AgentExecutor

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

def url_rag(url,question):
    loader = WebBaseLoader(url)
    docs=loader.load()
    documents = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200).split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorDB = FAISS.from_documents(documents, embeddings) 
    retriever = vectorDB.as_retriever()
    retriever_tool = create_retriever_tool(
    retriever,
    "url_search",
    "url search website",
    )
    
    tools = [retriever_tool]
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    prompt = hub.pull("hwchase17/openai-functions-agent")
    agent = create_openai_tools_agent(llm,tools,prompt)
    agent_executer = AgentExecutor(agent=agent,tools=tools,verbose=True)
    return agent_executer.invoke({"input":question})
    