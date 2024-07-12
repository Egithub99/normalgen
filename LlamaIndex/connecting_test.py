from llama_index.core import Settings
import autogen
from autogen.agentchat.contrib.llamaindex_conversable_agent import LLamaIndexConversableAgent
from llama_index.llms.lmstudio import LMStudio
from agent_test import agent


Settings.llm = LMStudio(
    model_name="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
    base_url="http://localhost:1234/v1",
    temperature=0,
    request_timeout=360,
)

table_reader = LLamaIndexConversableAgent(
    "table_reader",
    llama_index_agent=agent,
    system_message="""Reads documents and answer questions""",
)

