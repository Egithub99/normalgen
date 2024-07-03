import autogen
from load_bearing_agent import load_bearing_agent

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

# User Agent
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

# Problem Analysis Agent
problem_analysis_agent = autogen.ConversableAgent(
    name="problem_analysis_agent",
    llm_config=llm_config,
    system_message="""
    You are a problem analysis expert. Your goal is to gather detailed requirements for the design task.
    Please ask the following questions one by one and wait for the user to respond:
    1. How many stories should the building have?
    2. Do you prefer a certain material (e.g., steel or timber)?
    3. How many square meters should the building be?
    """,
    human_input_mode="ALWAYS",
    max_consecutive_auto_reply=1
)

material_agent = autogen.ConversableAgent(
    "material_agent",
    llm_config=llm_config,
    system_message="""
    You are a structural engineering expert with extensive knowledge about materials for buildings.
    Only consider the material choice for the load-bearing system. 
    Do not consider the foundation or roof.
    You can only choose between steel and timber for now.
    Your objective is to select the most appropriate material based on the task.
    """,
    # is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    max_consecutive_auto_reply=1,
    human_input_mode="NEVER"
)




# Analysis Group Chat
analysis = autogen.GroupChat(
    agents=[problem_analysis_agent, user],
    messages=[],
    speaker_selection_method="round_robin",
    allow_repeat_speaker=True,
    max_round=3,
    send_introductions=False
)

# Analysis Group Chat Manager
analysis_manager = autogen.GroupChatManager(
    groupchat=analysis,
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    llm_config=llm_config,
    code_execution_config=False,
    human_input_mode="ALWAYS"
)

# Synthesis Group Chat
synthesis = autogen.GroupChat(
    agents=[material_agent, load_bearing_agent],
    messages=[],
    speaker_selection_method="round_robin",
    allow_repeat_speaker=False,
    max_round=3,
    send_introductions=False
)

# Synthesis Group Chat Manager
synthesis_manager = autogen.GroupChatManager(
    groupchat=synthesis,
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    llm_config=llm_config,
    code_execution_config=False,
    human_input_mode="NEVER"
)

# Assistant Agent
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



# Writer Agent
writer = autogen.AssistantAgent(
    name="Writer",
    llm_config=llm_config,
    system_message="""
    You are a professional writer who provides a summary after a meeting.
    """,
)



# Nested Chat Queue connecting inner agents to load_bearing_agent
nested_chat_queue = [
    {"recipient": synthesis_manager, "summary_method": "reflection_with_llm"}, 
]

# Register nested chats for the assistant
assistant_1.register_nested_chats(
    nested_chat_queue,
    trigger=user,
)



# Start the analysis phase
analysis_res = analysis_manager.initiate_chats(
    [
        {
            "recipient": analysis_manager, 
            "message": tasks[0], 
            "max_turns": 3, 
            "summary_method": "last_msg",
        }
    ]
)

# Use the results of the analysis phase to start the synthesis phase
synthesis_res = user.initiate_chats(
    [
        {
            "recipient": assistant_1, 
            "message": analysis_res[0]["content"], 
            "max_turns": 1, 
            "summary_method": "last_msg",
        }
    ]
)
