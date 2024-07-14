# Import necessary libraries
from autogen import UserProxyAgent, AssistantAgent
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.lmstudio import LMStudio
import nest_asyncio
import os

# Apply nest_asyncio
nest_asyncio.apply()

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Define the LlamaIndex settings
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
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

# Check if storage already exists and load the index
PERSIST_DIR = "./storage_test_nieuw"
if not os.path.exists(PERSIST_DIR):
    documents = LlamaParse().load_data("./data5/8_developing-a-concept.pdf")
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine()

# Define the User Proxy Agent
user_proxy = UserProxyAgent(
    name="User_proxy",
    system_message="A human admin.",
    code_execution_config=False,
    human_input_mode="TERMINATE"
)

# Define the Assistant Agent that interacts with the vector database
class QueryAssistantAgent(AssistantAgent):
    def __init__(self, name, llm_config, query_engine):
        super().__init__(name, llm_config)
        self.query_engine = query_engine

    def handle_message(self, message):
        query = message['content']
        print(f"Received query: {query}")  # Debug statement
        response = self.query_engine.query(query)
        print(f"Query response: {response}")  # Debug statement
        return str(response)

assistant = QueryAssistantAgent(name="QueryAssistant", 
                                llm_config=llm_config, 
                                query_engine=query_engine)

# Define the user query
user_query = "Name six key steps in the conceptual design."

# Initiate the chat
chat_results = user_proxy.initiate_chats(
    [
        {
            "recipient": assistant,
            "message": user_query,
            "max_turns": 1,
            "summary_method": "last_msg",
        },
    ]
)

# # Print the chat results to see the output
# print(chat_results)

# # Function to print the response
# def handle_user_query_initiate_chats(chat_results):
#     response = chat_results[0]['response']
#     print(f"Final response: {response}")

# # Example usage
# handle_user_query_initiate_chats(chat_results)
