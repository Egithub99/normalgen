from llama_index.core.llms import ChatMessage
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.core.agent import ReActAgent

# bring in our LLAMA_CLOUD_API_KEY
from dotenv import load_dotenv
load_dotenv()
# bring in deps
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.lmstudio import LMStudio
import nest_asyncio
nest_asyncio.apply()
import os.path

# bge-base embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

Settings.llm = LMStudio(
    model_name="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
    base_url="http://localhost:1234/v1",
    temperature=0,
    request_timeout=360,
)

# check if storage already exists
PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    # load the documents and create the index
    documents3 = LlamaParse().load_data("./data4/table_9_7.pdf")
    index = VectorStoreIndex.from_documents(documents3)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine()


from llama_index.core.tools import QueryEngineTool, ToolMetadata
# Define custom tools or use QueryEngineTool with the created QueryEngine
query_engine_tool = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="my_query_engine",
        description="Provides detailed information from the indexed documents.",
    ),
)

# Create a ReActAgent instance with the tools and LLM
agent = ReActAgent.from_tools([query_engine_tool], llm=Settings.llm, verbose=True)

# Query the agent
response = agent.chat("Name some construction types from the table.")
print(str(response))