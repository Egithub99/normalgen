import autogen

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

tasks = [
    """Develop a conceptual design for a simple parking garage."""
]



# Inner Agents
load_bearing_agent = autogen.AssistantAgent(
    "load_bearing_agent",
    llm_config=llm_config,
    system_message="""
    You are a structural engineering expert with extensive knowledge about load-bearing systems for buildings.
    Your task is to assist in selecting the most appropriate load-bearing system.
    Be very brief when answering.

    """,
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
)

material_agent = autogen.AssistantAgent(
    "material_agent",
    llm_config=llm_config,
    system_message="""
    You are a structural engineering expert with extensive knowledge about materials like steel, timber and concrete for buildings.
    Your task is to assist in selecting the most appropriate material.
    Be very brief when answering.
    """,
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
)

# Group Chat for Inner Agents
inner_groupchat = autogen.GroupChat(
    agents=[load_bearing_agent, material_agent],
    messages=[],
    speaker_selection_method="round_robin",
    allow_repeat_speaker=True,
    max_round=3,
)

inner_manager = autogen.GroupChatManager(
    groupchat=inner_groupchat,
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    llm_config=llm_config,
    code_execution_config={
        "work_dir": "design",
        "use_docker": False,
    },
)

# Outer Agent
# load_bearing_agent = autogen.AssistantAgent(
#     name="Load_bearing_agent",
#     llm_config=llm_config,
#     system_message="""
#     You are a structural engineering expert with extensive knowledge about load-bearing systems for buildings.
#     Your task is to assist in selecting the most appropriate load-bearing system for the conceptual design.
#     Be very brief when answering.

#     """,
# )

assistant_1 = autogen.AssistantAgent(
    name="Assistant_1",
    system_message="""
                    You are the lead-engineer in this design process. 
                    Adhere to the following steps:
                    The load bearing agent will decide on the load-bearing system.
                    The material agent will decide on the appropriate material choice.
                    """,
    llm_config=llm_config,
)

user = autogen.UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    code_execution_config={
        "last_n_messages": 1,
        "work_dir": "tasks",
        "use_docker": False,
    },
)

# # # Message function for interaction
# def load_bearing_message(recipient, messages, sender, config):
#     return f"Choose a load-bearing system. \n\n {recipient.chat_messages_for_summary(sender)[-1]['content']}"

# Nested Chat Queue connecting inner agents to load_bearing_agent
nested_chat_queue = [
    {"recipient": inner_manager, "summary_method": "reflection_with_llm"},
    # {"recipient": load_bearing_agent, "message": "Choose a load-bearing system.", "summary_method": "last_msg", "max_turns": 1},
]
assistant_1.register_nested_chats(
    nested_chat_queue,
    trigger=user,
)

# Initiate chat with the main task and step 1 explicitly included
res = user.initiate_chats(
    [
        {
            "recipient": assistant_1, 
            "message": tasks[0], 
            "max_turns": 1, 
            "summary_method": "last_msg",
        },
        # {
        #     "recipient": load_bearing_agent,
        #     "max_turns": 1,
        #     "message": "Choose a load-bearing system.",
        #     "summary_method": "last_msg",
        # },
    ]
)
