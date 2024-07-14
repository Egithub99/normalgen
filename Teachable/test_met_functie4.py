# Import necessary libraries and modules
import os
from dotenv import load_dotenv
import nest_asyncio
import autogen
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.lmstudio import LMStudio
from typing_extensions import Annotated
from typing import Annotated
from autogen import register_function

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

from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent

class NewRetrieverUserProxyAgent(RetrieveUserProxyAgent):
# Combined function to query the vector database and print the response
    def query_and_ask_vector_db(question: Annotated[str, "The question to ask the VectorDB"]) -> Annotated[str, "The response from the VectorDB"]:
        # Query the vector database using the query engine
        response = query_engine.query(question)
        print(response)
        return response  # Return the response as a string\
    

# NewRetrieverUserProxyAgent.query_and_ask_vector_db("What are six steps in the conceptual design phase?")

# Use QdrantRetrieveUserProxyAgent
qdrantragagent = NewRetrieverUserProxyAgent(
    name="ragproxyagent",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=2,
    retrieve_config={
        "task": "qa",
    },
    code_execution_config=False,  # Do not execute the code
)

# 1. Create a RetrieveAssistantAgent instance named "assistant"
assistant = RetrieveAssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config=llm_config,
)


# Reset the assistant before starting a new conversation
assistant.reset()

qa_problem = "What are six steps in the conceptual design phase?"
chat_result = qdrantragagent.initiate_chat(
    assistant, 
    message=qdrantragagent.message_generator, 
    problem=qa_problem)

print(chat_result)





# # # Example query to the VectorDB (this line can be removed if the agent is expected to handle it)
# # query_and_ask_vector_db("Name six key steps in the conceptual design.")


# # 2. Create the RetrieveUserProxyAgent instance named "ragproxyagent"

# # 1. Create a RetrieveAssistantAgent instance named "assistant"
# assistant = RetrieveAssistantAgent(
#     name="assistant",
#     system_message="You are a helpful assistant.",
#     llm_config=llm_config,
# )

# ragproxyagent = RetrieveUserProxyAgent(
#     name="ragproxyagent",
#     human_input_mode="NEVER",
#     # max_consecutive_auto_reply=5,
#     retrieve_config={
#         "task": "qa",
#         "docs_path": "./storage_test2",
#     },
#     code_execution_config=False,  # Do not execute the code
# )

# # Reset the assistant before starting a new conversation
# assistant.reset()


# qa_problem = "Name six key steps in the conceptual design."
# chat_result = ragproxyagent.initiate_chat(
#     assistant, 
#     message=ragproxyagent.message_generator, 
#     problem=qa_problem)

# print(chat_result)