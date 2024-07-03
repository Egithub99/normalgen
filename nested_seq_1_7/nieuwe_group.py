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
    """Develop a structural design concept for a simple 2 story parking garage."""
]

# New Problem Analysis Agent
problem_analysis_agent = autogen.ConversableAgent(
    "problem_analysis_agent",
    llm_config=llm_config,
    system_message="""
    You are an expert in analyzing building requirements.
    Your task is to interact with the user to gather the necessary details about the building requirements.
    Start by asking: 'How many square meters should the building be?'
    Continue the conversation to understand all requirements.
    """,
    human_input_mode="ALWAYS"
)

user = autogen.UserProxyAgent(
    name="User",
    human_input_mode="ALWAYS",
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    code_execution_config={
        "last_n_messages": 1,
        "work_dir": "tasks",
        "use_docker": False,
    },
)

# Group Chat for Inner Agents
analysis = autogen.GroupChat(
    agents=[problem_analysis_agent, user],
    messages=[],
    speaker_selection_method="round_robin",
    allow_repeat_speaker=False,
    max_round=3,
    send_introductions=False
)

inner_manager = autogen.GroupChatManager(
    groupchat=analysis,
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    llm_config=llm_config,
    code_execution_config=False,
    human_input_mode="ALWAYS"
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

# Nested Chat Queue connecting inner agents to load_bearing_agent
nested_chat_queue = [
    {"recipient": inner_manager, "summary_method": "reflection_with_llm"},
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
