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
    """Develop a conceptual design for a simple parking garage. The process includes 2 steps:
    Step 1: The Load bearing agent chooses a load-bearing system based on typical building requirements.
    Step 2: The Material agent chooses a material to construct the building based on the selected load-bearing system.
    """,
]

inner_assistant = autogen.AssistantAgent(
    "Inner-assistant",
    llm_config=llm_config,
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
)

# inner_lead_engineer = autogen.UserProxyAgent(
#     "Inner-lead-engineer",
#     human_input_mode="NEVER",
#     code_execution_config=False,
#     default_auto_reply="",
#     is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
# )

inner_lead_engineer = autogen.AssistantAgent(
    "Inner-lead-engineer",
    llm_config=llm_config,
    default_auto_reply="",
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
)


groupchat = autogen.GroupChat(
    agents=[inner_assistant, inner_lead_engineer],
    messages=[],
    speaker_selection_method="round_robin",
    allow_repeat_speaker=False,
    max_round=8,
)

manager = autogen.GroupChatManager(
    groupchat=groupchat,
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    llm_config=llm_config,
    code_execution_config={
        "work_dir": "design",
        "use_docker": False,
    },
)

assistant_1 = autogen.AssistantAgent(
    name="Assistant_1",
    llm_config=llm_config,
)

load_bearing_agent = autogen.AssistantAgent(
    name="Load_bearing_agent",
    llm_config=llm_config,
    system_message="""
    You are a structural engineering expert with extensive knowledge about load-bearing systems for buildings.
    Your task is to assist in selecting the most appropriate load-bearing system for the conceptual design.
    You should focus on discussions and providing insights without writing or executing any code.
    """,
)

material_agent = autogen.AssistantAgent(
    name="Material_agent",
    llm_config=llm_config,
    system_message="""
    You are a structural engineering expert with extensive knowledge about materials for building.
    Your task is to assist in selecting the most appropriate materials for the conceptual design.
    You should focus on discussions and providing insights without writing or executing any code.
    """,
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

def load_bearing_message(recipient, messages, sender, config):
    return f"Step 1: Choose a load-bearing system based on typical building requirements. \n\n {recipient.chat_messages_for_summary(sender)[-1]['content']}"

def material_message(recipient, messages, sender, config):
    return f"Step 2: Choose a material to construct the building based on the selected load-bearing system. \n\n {recipient.chat_messages_for_summary(sender)[-1]['content']}"

nested_chat_queue = [
    {"recipient": manager, "summary_method": "reflection_with_llm"},
    {"recipient": load_bearing_agent, "message": load_bearing_message, "summary_method": "last_msg", "max_turns": 1},
    {"recipient": material_agent, "message": material_message, "summary_method": "last_msg", "max_turns": 1},
]
assistant_1.register_nested_chats(
    nested_chat_queue,
    trigger=user,
)

res = user.initiate_chats(
    [
        {
            "recipient": assistant_1, 
            "message": tasks[0], 
            "max_turns": 1, 
            "summary_method": "last_msg",
            "steps": [
                "Step 1: The Load_bearing_agent chooses a load-bearing system based on typical building requirements.",
                "Step 2: The Material_agent chooses a material to construct the building based on the selected load-bearing system."
            ]
        },
    ]
)
