import autogen

# Configuration for LLM
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
load_bearing_agent = autogen.AssistantAgent(
    "LoadBearingAgent",
    llm_config=llm_config,
    system_message="You are a structural engineering expert in choosing load-bearing systems for buildings. "
                   "You have information on various load-bearing systems such as Post and Beam, Reinforced Concrete, "
                   "Steel Frame, and Masonry. You choose the most suitable system."
)

material_agent = autogen.AssistantAgent(
    "MaterialAgent",
    llm_config=llm_config,
    system_message="You are a structural engineering expert in selecting construction materials for buildings. "
                   "You have information on various materials such as Wood, Steel, Concrete, Glass, and Aluminum. "
                   "You need to choose the most suitable material."
)

# Define the user proxy agent with the initiation message
user_proxy = autogen.UserProxyAgent(
    name="UserProxy",
    system_message="""
                    The client.
""",
    code_execution_config={"last_n_messages": 2, "use_docker": False, "work_dir": "groupchat"},
    human_input_mode="NEVER",
    is_termination_msg=lambda x: x.get("content", "").find("Terminate") >= 0,
)

# Define the group chat with both agents
groupchat = autogen.GroupChat(
    agents=[load_bearing_agent, material_agent],
    messages=[],
    speaker_selection_method="round_robin",
    allow_repeat_speaker=False,
    max_round=6,
)

# Define the group chat manager
group_chat_manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config,
)

# Nested chat queue with clearer coordination by the chat_manager
nested_chat_queue = [
    {"recipient": load_bearing_agent, "message": "Choose a load bearing system based on typical building requirements.", "summary_method": "last_msg", "max_turns": 1},
    {"recipient": group_chat_manager, "message": "Notify material agent to select material based on the chosen load bearing system.", "summary_method": "last_msg", "max_turns": 1},
    {"recipient": material_agent, "message": "Choose a material to construct the building based on the load bearing system provided by the load bearing agent.", "summary_method": "last_msg", "max_turns": 1},
    {"recipient": group_chat_manager, "message": "Terminate", "summary_method": "last_msg"}
]

# Register the nested chats
user_proxy.register_nested_chats(
    nested_chat_queue,
    trigger=user_proxy,
)

# Initiate the chat with the user proxy agent
user_proxy.initiate_chat(
    recipient=group_chat_manager,
    message="""Task to solve: The conceptual design of a simple parking garage.
    
    Process overview to solve task:

Step 1: The load_bearing_agent chooses a load bearing system based on typical building requirements.

Step 2: The group_chat_manager notifies the material_agent to select a material based on the load bearing system.

Step 3: The material_agent chooses a material to construct the building based on the load bearing system.

Step 4: Conclude the process with sending 'done'.
""",
    max_turns=1,
    summary_method="last_msg",
)
