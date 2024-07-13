# Import necessary libraries and modules
import os
from dotenv import load_dotenv
import nest_asyncio
import autogen
from autogen import UserProxyAgent, GroupChat, GroupChatManager
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.lmstudio import LMStudio

# Apply nested asyncio
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Define embedding model settings
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# Define LLM settings
Settings.llm = LMStudio(
    model_name="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
    base_url="http://localhost:1234/v1",
    temperature=0,
    request_timeout=360,
)

# Define LLM configuration
llm_config = {
    "config_list": [
        {
            "model": "TheBloke/Mistral-7B-Instruct-v0.1-GGUF/mistral-7b-instruct-v0.1.Q5_0.gguf",
            "base_url": "http://localhost:1234/v1",
            "api_key": "lm-studio",
        },
    ],
    "cache_seed": None,  # Disable caching.
}

# Define the directory for persistent storage
PERSIST_DIR = "./storage_test2"

# Check if storage already exists and load or create the index
if not os.path.exists(PERSIST_DIR):
    # Load the documents and create the index
    documents = LlamaParse().load_data("./data5/8_developing-a-concept.pdf")
    index = VectorStoreIndex.from_documents(documents)
    # Store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # Load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

# Create a query engine from the index
query_engine = index.as_query_engine()

# Define a new agent for connecting to the vector database
vector_db_agent = autogen.AssistantAgent(
    name="VectorDBAgent",
    system_message="Responsible for querying the vector database.",
    llm_config=llm_config,
)

# Function for the VectorDBAgent to query the vector database
def query_vector_database(query):
    # Query the vector database using the query engine
    return query_engine.query(query)

# Integrate the vector database querying functionality into the VectorDBAgent
vector_db_agent.query_database = query_vector_database

# Function to allow the UserProxyAgent to ask the VectorDBAgent a question and get a response
def ask_vector_db_agent(question):
    # UserProxyAgent sends the question to VectorDBAgent
    response = vector_db_agent.query_database(question)
    # Return the response to the user
    return response
