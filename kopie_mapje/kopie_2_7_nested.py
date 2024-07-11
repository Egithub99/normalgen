import autogen



llm_config = {
    "config_list": [
        {
            "model": "TheBloke/Mistral-7B-Instruct-v0.1-GGUF/mistral-7b-instruct-v0.1.Q5_0.gguf",
            "base_url": "http://localhost:1234/v1",
            "api_key": "lm-studio",
        },
    ],
    "cache_seed": None,  # Disable caching.
}

tasks = [
    """Develop a conceptual design for a simple 2 story parking garage."""
]



# Inner Agents


material_agent = autogen.AssistantAgent(
    "material_agent",
    llm_config=llm_config,
    system_message="""
    You are a structural engineering expert with extensive knowledge about materials for buildings.
    Only consider the material choice for the load-bearing system. 
    Do not consider the foundation or roof.
    You can only choose between steel and timber for now.
    Your objective is to select the most appropriate material based on the task.
    Be very brief when answering.
    """,
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
)

load_bearing_agent = autogen.AssistantAgent(
    "load_bearing_agent",
    llm_config=llm_config,
    system_message="""
    You are a structural engineering expert with extensive knowledge about load-bearing systems for buildings.
    Your select the most appropriate load-bearing system.
    Select an appropriate load-bearing system based on the answer of the material agent.
    Only consider the above ground structure, do not make recommendations for the foundation or the roof.
    Be very brief when answering.

    """,
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
)

# Group Chat for Inner Agents
inner_groupchat = autogen.GroupChat(
    agents=[material_agent, load_bearing_agent],
    messages=[],
    speaker_selection_method="round_robin",
    allow_repeat_speaker=False,
    max_round=8,
)

inner_manager = autogen.GroupChatManager(
    groupchat=inner_groupchat,
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    llm_config=llm_config,
    code_execution_config=False
)


assistant_1 = autogen.AssistantAgent(
    name="Assistant_1",
    system_message="""
                    You are the lead-engineer in this design process. 
                    Adhere to the following steps:
                    The material agent will decide on the appropriate material choice.
                    The load bearing agent will decide on the load-bearing system.
                    
                    """,
    llm_config=llm_config,
)

# writer = autogen.AssistantAgent(
#     name="Writer",
#     llm_config=llm_config,
#     system_message="""
#     You are a professional writer.
#     You transform complex concepts into a well structured text.
#     """,
# )


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


# def writing_message(recipient, messages, sender, config):
#     return f"Polish the content to make a nicely formatted document. \n\n {recipient.chat_messages_for_summary(sender)[-1]['content']}"


# Nested Chat Queue connecting inner agents to load_bearing_agent
nested_chat_queue = [
    {"recipient": inner_manager, "summary_method": "reflection_with_llm"},
    # {"recipient": writer, "message": writing_message, "summary_method": "last_msg", "max_turns": 1},
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
        }
    ]
)