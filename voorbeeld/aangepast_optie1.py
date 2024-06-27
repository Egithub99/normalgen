import os
import tempfile
from autogen import ConversableAgent, GroupChatManager, GroupChat, UserProxyAgent

temp_dir = tempfile.gettempdir()

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

# Load Bearing Agent
load_bearing_agent = ConversableAgent(
    name="Load_Bearing_Agent",
    system_message="You are an expert in load bearing systems for buildings.",
    llm_config=llm_config,
    human_input_mode="NEVER",
    code_execution_config={"use_docker": False, "work_dir": temp_dir},
)

# Material Agent
material_agent = ConversableAgent(
    name="Material_Agent",
    system_message="You are an expert in building materials.",
    llm_config=llm_config,
    human_input_mode="NEVER",
    code_execution_config={"use_docker": False, "work_dir": temp_dir},
)

# User Proxy Agent
user_proxy = UserProxyAgent(
    name="User_proxy",
    system_message="A human admin.",
    code_execution_config={"last_n_messages": 2, "use_docker": False, "work_dir": temp_dir},
    human_input_mode="TERMINATE"
)

# Group Chat Manager
group_chat_manager = GroupChatManager(
    groupchat=GroupChat(
        agents=[user_proxy, load_bearing_agent, material_agent],
        messages=[],
        max_round=5
    ),
    llm_config=llm_config
)

# Nested Chats
nested_chats = [
    {
        "recipient": load_bearing_agent,
        "message": "Choose a load bearing system for the building.",
        "summary_method": "reflection_with_llm",
    },
    {
        "recipient": material_agent,
        "message": "Choose a material to construct the building.",
        "summary_method": "reflection_with_llm",
    },
    {
        "recipient": group_chat_manager,
        "message": "Terminate",
        "summary_method": "last_msg",
    },
]

# Registering nested chats for the user proxy agent
user_proxy.register_nested_chats(
    nested_chats,
    trigger=lambda sender: sender not in [load_bearing_agent, material_agent, group_chat_manager],
)

# Initiate the process overview message
initial_message = """Situation:

The conceptual design of a simple parking garage.

Process overview:

Step 1: choose a load bearing system.

Step 2: choose a material to construct the building in.

Step 3: Conclude the process with sending 'Terminate'
"""

# Start the conversation
user_proxy.generate_reply(messages=[{"role": "user", "content": initial_message}])
