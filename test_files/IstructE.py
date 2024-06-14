import pprint

import autogen

from autogen import AssistantAgent

import os
from datetime import datetime
from typing import Callable, Dict, Literal, Optional, Union

from typing_extensions import Annotated

from autogen import (
    Agent,
    AssistantAgent,
    ConversableAgent,
    GroupChat,
    GroupChatManager,
    UserProxyAgent,
    config_list_from_json,
    register_function,
)
from autogen.agentchat.contrib import agent_builder
from autogen.cache import Cache
from autogen.coding import DockerCommandLineCodeExecutor, LocalCommandLineCodeExecutor


# Configuration list for the language model
config_list = {
    "config_list": [
        {
            "model": "lmstudio-ai/gemma-2b-it-GGUF/gemma-2b-it-q8_0.gguf",
            "base_url": "http://localhost:1234/v1",
            "api_key": "lm-studio",
        },
    ],
    "cache_seed": None,  # Disable caching.
}

assistant = autogen.AssistantAgent(name='assistant', llm_config=config_list)

# Create planner agent
planner = AssistantAgent(
    name="planner",
    llm_config={
        "config_list": config_list,
        "cache_seed": None,  # Disable legacy cache.
    },
    system_message=(
        "You are a helpful AI assistant specialized in structural engineering. "
        "Your task is to decompose the process of designing a parking garage into three main steps: "
        "1. Problem analysis "
        "2. Choosing a load-bearing system "
        "3. Material choice. "
        "If the plan is not good, suggest a better plan. "
        "If the execution is wrong, analyze the error and suggest a fix."
    ),
)

# Create a planner user agent to interact with the planner
planner_user = UserProxyAgent(
    name="planner_user",
    human_input_mode="NEVER",
    code_execution_config=False,
)

# Function for asking the planner
def task_planner(question: Annotated[str, "Question to ask the planner."]) -> str:
    with Cache.disk(cache_seed=4) as cache:
        planner_user.initiate_chat(planner, message=question, max_turns=1, cache=cache)
    # Return the last message received from the planner
    return planner_user.last_message()["content"]

# Create assistant agent
assistant = AssistantAgent(
    name="assistant",
    system_message=(
        "You are a helpful AI assistant specialized in structural engineering. "
        "You can use the task planner to decompose the design process of a parking garage into three main steps: "
        "1. Problem analysis "
        "2. Choosing a load-bearing system "
        "3. Material choice. "
        "Make sure you follow through the sub-tasks. "
        "When needed, write Python code in markdown blocks, and I will execute them. "
        "Give the user a final solution at the end. "
        "Return TERMINATE only if the sub-tasks are completed."
    ),
    llm_config={
        "config_list": config_list,
        "cache_seed": None,  # Disable legacy cache.
    },
)

# Setting up code executor
os.makedirs("planning", exist_ok=True)
code_executor = LocalCommandLineCodeExecutor(work_dir="planning")

# Create user proxy agent to interact with the assistant
user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="ALWAYS",
    is_termination_msg=lambda x: "content" in x
    and x["content"] is not None
    and x["content"].rstrip().endswith("TERMINATE"),
    code_execution_config={"executor": code_executor},
)

# Register the function to the agent pair
register_function(
    task_planner,
    caller=assistant,
    executor=user_proxy,
    name="task_planner",
    description="A task planner that can help you with decomposing the design process of a parking garage into sub-tasks.",
)

# Use Cache.disk to cache LLM responses
task = "Design a parking garage for structural engineering"
with Cache.disk(cache_seed=1) as cache:
    # The assistant receives a message from the user, which contains the task description
    user_proxy.initiate_chat(
        assistant,
        message=task,
        cache=cache,
    )
