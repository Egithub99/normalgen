
import json
import os

import chromadb

import autogen
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent

# Accepted file formats for that can be stored in
# a vector database instance
from autogen.retrieve_utils import TEXT_FORMATS

llm_config = {
    "config_list": [
        {
            "model": "TheBloke/Mistral-7B-Instruct-v0.1-GGUF/mistral-7b-instruct-v0.1.Q5_0.gguf",
            # "model": "mistral",
            "base_url": "http://localhost:1234/v1",
            "api_key": "lm-studio",
        },
    ],
    "cache_seed": None,  # Disable caching.
}

model_name = llm_config["config_list"][0]["model"]
print(model_name)

print("Accepted file formats for `docs_path`:")
print(TEXT_FORMATS)

# 1. Create a RetrieveAssistantAgent instance named "assistant"
assistant = RetrieveAssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config=llm_config,
)

# 2. Create the RetrieveUserProxyAgent instance named "ragproxyagent"
ragproxyagent = RetrieveUserProxyAgent(
    name="ragproxyagent",
    human_input_mode="NEVER",
    # max_consecutive_auto_reply=3,
    retrieve_config={
        "task": "qa",
        "docs_path": "1_introduction.pdf",
        "custom_text_types": ["pdf"],  # Specify the file type to be processed
        "chunk_token_size": 2000,  # Reduce chunk size
        # "model": model_name,  # Pass the model name instead of the whole config
        "vector_db": "chroma",
        "overwrite": True,
        "must_break_at_empty_line": False,

    },
    code_execution_config=False,  # Do not execute the code
)

# Reset the assistant before starting a new conversation
assistant.reset()

qa_problem = "What is this chapter about?"
chat_result = ragproxyagent.initiate_chat(
    assistant, 
    message=ragproxyagent.message_generator, 
    problem=qa_problem)

print(chat_result)
