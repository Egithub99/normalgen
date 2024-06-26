import os
import autogen
from autogen import ConversableAgent

llm_config = {
    "config_list": [
        {
            "model": "lmstudio-ai/gemma-2b-it-GGUF/gemma-2b-it-q8_0.gguf",
            "base_url": "http://localhost:1234/v1",
            "api_key": "lm-studio",
        },
    ],
    "cache_seed": None,  # Disable caching.
}

# Define each agent
lead_engineer = ConversableAgent(
    name="Lead Engineer",
    system_message="Lead Engineer: Ensures the engineering integrity and supervises all technical activities.",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

planner = ConversableAgent(
    name="Planner",
    system_message="""Planner: Coordinates project timelines and ensures that all tasks are scheduled efficiently.
                        Decomposes the process of designing a parking garage into this one step: Choosing a Load bearing system. 
                        Only name the steps provided, no elaboration.  
                        """,
    llm_config=llm_config,
    human_input_mode="NEVER",
)

load_bearing_agent = ConversableAgent(
    name="Load Bearing Agent",
    system_message="""Load Bearing Agent. Has extensive knowledge on load-bearing systems in structural engineering""",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# Start a sequence of two-agent chats.
# Each element in the list is a dictionary that specifies the arguments
# for the initiate_chat method.
chat_results = lead_engineer.initiate_chats(
    [
        {
            "recipient": planner,
            "message": "Let's start planning the project.",
            "max_turns": 1,
            "summary_method": "last_msg",
            "discription": "Planner, tasked with scheduling and coordinating project activities.",
        },
        {
            "recipient": load_bearing_agent,
            "message": "Choose a load-bearing system.",
            "max_turns": 2,
            "summary_method": "last_msg",
        },
    ]
)

