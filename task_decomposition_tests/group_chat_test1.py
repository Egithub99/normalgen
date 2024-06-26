from autogen import UserProxyAgent, AssistantAgent, GroupChat, GroupChatManager, Cache
import os

# New configuration using gemma
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


# Define agents
user_proxy = UserProxyAgent(
    name="Admin",
    system_message="A human admin. Task: Design a simple parking garage.",
    code_execution_config={"last_n_messages": 2, "work_dir": "groupchat", "use_docker": False},
    human_input_mode="TERMINATE"
)

planner = AssistantAgent(
    name="Planner",
    system_message="""Helpful AI planner agent. 
                        Your task is to decompose the process of designing a parking garage into this step: Choosing a Load bearing system choice. 
                        You only provide this planning, you do not answer any other question.
                        You only name the steps provided, no elaboration. 
                        This plan should only include selecting a load-bearing system. 
                        but the actual selection will be done by the load-bearing agent.
                        """,
    llm_config=llm_config,
    description="Provides a plan to decompose the conceptual design."
)


engineer = AssistantAgent(
    name="Engineer",
    system_message="""Engineer. You keep the overview of the design and consult the load_bearing agent for advice on the load-bearing system""",
    llm_config=llm_config,
)

writer = AssistantAgent(
    name="Writer",
    system_message="""Writer. Please write a plan for the task in markdown format. 
                    The plan should include the selected load-bearing system and any relevant details provided by the engineer.""",
    llm_config=llm_config,
)

# Create necessary directory
os.makedirs("groupchat", exist_ok=True)

# Define the group chat
groupchat = GroupChat(
    agents=[user_proxy, planner, engineer, writer],
    messages=[],
    max_round=20,
    speaker_selection_method="auto",
)

# Setup GroupChat Manager
manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# Use Cache.disk to cache LLM responses. Change cache_seed for different responses.
with Cache.disk(cache_seed=41) as cache:
    chat_history = user_proxy.initiate_chat(
        manager,
        message="Design a simple parking garage.",
        cache=cache,
    )
