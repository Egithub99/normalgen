# Import necessary libraries and modules
import os
from dotenv import load_dotenv
import nest_asyncio
import autogen
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

# Function to query the vector database
def query_vector_database(query: str) -> str:
    # Query the vector database using the query engine
    return query_engine.query(query)

# Simple interface to ask questions to the VectorDB
def ask_vector_db(question: str) -> str:
    response = query_vector_database(question)
    print(response)
    return response  # Return the response as a string


# Example query to the VectorDB
ask_vector_db("Name six key steps in the conceptual design.")


from autogen import ConversableAgent

# Define the assistant agent that suggests tool calls
assistant = ConversableAgent(
    name="Assistant",
    system_message="You are a helpful AI assistant. "
    "You can help with reading documents. "
    "Return 'TERMINATE' when the task is done.",
    llm_config=llm_config,
)

# The user proxy agent is used for interacting with the assistant agent and executes tool calls
user_proxy = ConversableAgent(
    name="User",
    llm_config=False,
    is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
    human_input_mode="NEVER",
)


# Register the tool signature with the assistant agent
assistant.register_for_llm(name="RAG_test", description="Interacting with the database")(ask_vector_db)

# Register the tool function with the user proxy agent
user_proxy.register_for_execution(name="RAG_test")(ask_vector_db)


# Start the sequence of chats with extended max_turns for feedback incorporation
chat_results = user_proxy.initiate_chats(
    [
        {
            "recipient": assistant,
            "message": "Name six key steps in the conceptual design.",
            "max_turns": 1,
            "summary_method": "last_msg",
        },
    ]
)