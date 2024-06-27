import autogen
from typing import List, Dict, Any

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

# process_overview_message = """
#     The process overview is as follows:
    
#     Step 1: The load-bearing agent should choose a load-bearing system.
# """

tasks = [
    "Develop a conceptual design for a simple parking garage."
]

load_bearing_agent = autogen.AssistantAgent(
    name="Load_bearing_agent",
    llm_config=llm_config,
    system_message="""
    You are a structural engineering expert with extensive knowledge about load-bearing systems for buildings.
    Your task is to assist in selecting the most appropriate load-bearing system for the conceptual design.
    You should focus on discussions and providing insights without writing or executing any code.
    """,
)

assistant_1 = autogen.AssistantAgent(
    name="Assistant_1",
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

def load_bearing_message(recipient: autogen.AssistantAgent, messages: List[Dict[str, Any]], sender: autogen.AssistantAgent, config: Dict[str, Any]) -> str:
    return f"Choose a load-bearing system based on typical building requirements. \n\n {recipient.chat_messages_for_summary(sender)[-1]['content']}"

nested_chat_queue = [
    {"recipient": load_bearing_agent, "message": load_bearing_message, "summary_method": "last_msg", "max_turns": 1},
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
        },
    ]
)


# res = user.initiate_chats(
#     [
#         {
#             "recipient": assistant_1, 
#             "message": tasks[0], 
#             "max_turns": 1, 
#             "summary_method": "last_msg",
#         },
#         {
#             "recipient": load_bearing_agent,
#             "message": "Step 1: The Load_bearing_agent chooses a load-bearing system based on typical building requirements.",
#             "max_turns": 1,
#             "summary_method": "last_msg",
#         },
#     ]
# )
