import os
from autogen import ConversableAgent
from agent_config import gemma_config

planner_agent = ConversableAgent(
    name="Planner_Agent",
    system_message=(
        "You are a helpful AI assistant specialized in structural engineering. "
        "Your task is to decompose the process of designing a parking garage into two main steps: "
        "1. Load bearing system choice. "
        "2. Material choice. "
        "You only provide this planning, you do not answer any other question."
        "You only name the steps provided, no elaboration."
    ),
    llm_config=gemma_config,
    human_input_mode="NEVER",
)


