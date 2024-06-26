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
    description="I am the Lead Engineer responsible for overseeing all engineering aspects of the project."
)

planner = ConversableAgent(
    name="Planner",
    system_message="Planner: Coordinates project timelines and ensures that all tasks are scheduled efficiently.",
    llm_config=llm_config,
    human_input_mode="NEVER",
    description="I am the Planner, tasked with scheduling and coordinating project activities."
)

load_bearing_agent = ConversableAgent(
    name="Load Bearing Agent",
    system_message="Load Bearing Agent: Ensures that all load-bearing structures are designed to be safe and effective.",
    llm_config=llm_config,
    human_input_mode="NEVER",
    description="I am the Load Bearing Agent, focusing on the structural integrity and load distribution of the project."
)

# Start a sequence of two-agent chats.
# Each element in the list is a dictionary that specifies the arguments
# for the initiate_chat method.
chat_results = lead_engineer.initiate_chats(
    [
        {
            "recipient": planner,
            "message": "Let's start planning the project.",
            "max_turns": 2,
            "summary_method": "last_msg",
        },
        {
            "recipient": load_bearing_agent,
            "message": "Here's the initial plan from the planner.",
            "max_turns": 2,
            "summary_method": "last_msg",
        },
    ]
)

