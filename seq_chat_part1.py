import os
from autogen import ConversableAgent

# Configuration for the LLM
gemma_config = {
    "config_list": [
        {
            "model": "lmstudio-ai/gemma-2b-it-GGUF/gemma-2b-it-q8_0.gguf",
            "base_url": "http://localhost:1234/v1",
            "api_key": "lm-studio",
        },
    ],
    "cache_seed": None,  # Disable caching.
}

# The Lead Engineer always returns the same message.
lead_engineer_agent = ConversableAgent(
    name="Lead_Engineer_Agent",
    system_message="""You are the lead engineer. You know the planning and move on to the next conversation."
                    You have the oversight over the entire project. """,
    llm_config=gemma_config,
    human_input_mode="NEVER",
)

# The Planner Agent decomposes the design process into three main steps.
planner_agent = ConversableAgent(
    name="Planner_Agent",
    system_message=(
        "You are a helpful AI assistant specialized in structural engineering. "
        "Your task is to decompose the process of designing a parking garage into three main steps: "
        "1. Problem analysis "
        "2. Choosing a load-bearing system "
        "3. Material choice. "
        "Do not elaborate on the steps, just name them. "
    ),
    llm_config=gemma_config,
    human_input_mode="NEVER",
)

# Start a sequence of two-agent chats.
# Each element in the list is a dictionary that specifies the arguments
# for the initiate_chat method.
chat_results = lead_engineer_agent.initiate_chats(
    [
        {
            "recipient": planner_agent,
            "message": "Start planning the design of the parking garage.",
            "max_turns": 1,
            "summary_method": "last_msg",
        }
    ]
)
