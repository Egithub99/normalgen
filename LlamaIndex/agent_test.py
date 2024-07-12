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
    index3 = VectorStoreIndex.from_documents(documents3)
    # store it for later
    index3.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index3 = load_index_from_storage(storage_context)

query_engine3 = index3.as_query_engine()

# response3 = query_engine3.query(
#     "Can you name the Construction types from the table?"
# )
# print(response3)


response4 = query_engine3.query(
    "What a construction type option in steel?"
)
print(response4)

import autogen
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

# Define the main task
main_task = "Develop a structural conceptual design of a 2-story parking garage with dimensions 20 by 20 meters."

# Define tasks
tasks = [
    f"{main_task} Consult the Material Agent for material selection and the Load-Bearing Agent for load-bearing system selection."
]


# Define subtasks using f-strings to incorporate the main task context
subtasks = [
    f"Please recommend the best material for constructing a building with dimensions similar to the one described in the main task: {main_task}.",
    f"Based on the chosen material, please recommend the best load-bearing system for a building similar to the one described in the main task: {main_task}."
]

# Define the Lead Engineer
lead_engineer = autogen.ConversableAgent(
    name="Lead_engineer",
    system_message=f"You are the lead engineer responsible for: {tasks[0]}",
    llm_config=llm_config,
    human_input_mode="ALWAYS",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
)

# Define the Material Agent
material_agent = autogen.AssistantAgent(
    name="Material_Agent",
    system_message="You are an expert in materials for construction projects. Provide the best material option based on the requirements provided to you.",
    llm_config=llm_config,
)

# Define the Load-Bearing Agent
load_bearing_agent = autogen.AssistantAgent(
    name="Load_Bearing_Agent",
    system_message=f"You are an expert in load-bearing systems. Based on the chosen material, provide the best load-bearing system option based on {response4}",
    llm_config=llm_config,
)

# Define the User Proxy Agent
user_proxy = autogen.UserProxyAgent(
    name="User_Proxy",
    system_message="A human admin overseeing the process.",
    code_execution_config=False,
    human_input_mode="ALWAYS"  # Allow human input at all stages
)

# # Define the Writer Agent
# writer = autogen.AssistantAgent(
#     name="Writer",
#     llm_config=llm_config,
#     system_message="You are a professional writer. Reply 'TERMINATE' in the end when everything is done.",
# )


# Start the sequence of chats with extended max_turns for feedback incorporation
chat_results = lead_engineer.initiate_chats(
    [
        # {
        #     "recipient": material_agent,
        #     "message": subtasks[0],
        #     "max_turns": 1,
        #     "summary_method": "last_msg",
        # },
        {
            "recipient": load_bearing_agent,
            "message": subtasks[1],
            "max_turns": 1,
            "summary_method": "last_msg",
        },
    ]
)
