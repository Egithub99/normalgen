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
    """Develop a structural design concept for a simple parking garage.
        The parking garage will be placed on the TU Delft campus and should hold about 200 cars.
        The users are employees and students from the TU Delft.
        The parking garage can be 2-3 stories high.
    """
]

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

# # Group Chat for Inner Agents
# synthesis = autogen.GroupChat(
#     agents=[material_agent, load_bearing_agent],
#     messages=[],
#     speaker_selection_method="round_robin",
#     allow_repeat_speaker=False,
#     max_round=3,
#     send_introductions=False
# )

# inner_manager = autogen.GroupChatManager(
#     groupchat=synthesis,
#     is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
#     llm_config=llm_config,
#     code_execution_config=False,
#     human_input_mode="NEVER"
# )

assistant_1 = autogen.AssistantAgent(
    name="Assistant_1",
    system_message="""
                    You are the lead-engineer in this design process.
                    """,
    llm_config=llm_config,
    human_input_mode="NEVER"
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


# # Nested Chat Queue connecting inner agents to load_bearing_agent and summary to user
# nested_chat_queue = [
#     {"recipient": material_agent, "summary_method": "last_msg"},
#     {"recipient": load_bearing_agent, "summary_method": "last_msg"},
# ]

# assistant_1.register_nested_chats(
#     nested_chat_queue,
#     trigger=user,
# )

# Initiate chat with the main task and step 1 explicitly included
# res = user.initiate_chats(
#     [
#         {
#             "recipient": assistant_1, 
#             "message": f"""For the following task: {tasks[0]}.

#                         The process overview for this task is:
#                         Step 1: The material agent will choose a suitable material.

#                         Step 2: The load-bearing agent will choose a suitable load-bearing system which is in line with the material choice.

#                         """, 
#             "max_turns": 1, 
#             "summary_method": "last_msg",
#         }
#     ]
# )

# Initiate chat with the main task and step 1 explicitly included
res = user.initiate_chats(
    [
        {
            "recipient": assistant_1, 
            "message": f"""For the following task: {tasks[0]}.

                        The process overview for this task is:
                        Step 1: The material agent will choose a suitable material.

                        Step 2: The load-bearing agent will choose a suitable load-bearing system which is in line with the material choice.

                        """, 
            "max_turns": 1, 
            "summary_method": "last_msg",
        }
    ]
)