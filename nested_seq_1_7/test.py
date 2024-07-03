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
        The parking garage should can be 2-3 stories high.
        """
]


# Inner Agents
material_agent = autogen.ConversableAgent(
    "material_agent",
    llm_config=llm_config,
    system_message="""
    You are a structural engineering expert with extensive knowledge about materials for buildings.
    Your task is to select the most appropriate material for the load-bearing system of the parking garage. 
    Consider factors such as cost, durability, ease of construction, maintenance requirements, and environmental impact.
    You can only choose between steel and timber for now.
    Provide a clear rationale for your choice.
    """,
    max_consecutive_auto_reply=1,
    human_input_mode="NEVER"
)



# Group Chat for Inner Agents
synthesis = autogen.GroupChat(
    agents=[material_agent, load_bearing_agent],
    messages=[],
    speaker_selection_method="round_robin",
    allow_repeat_speaker=False,
    max_round=3,
    send_introductions=False
)

inner_manager = autogen.GroupChatManager(
    groupchat=synthesis,
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    llm_config=llm_config,
    code_execution_config=False,
    human_input_mode="ALWAYS"
)

assistant_1 = autogen.AssistantAgent(
    name="Assistant_1",
    llm_config=llm_config,
    system_message="""
    You are the lead engineer overseeing the design process of the parking garage.
    Coordinate with the material and load-bearing agents to ensure a cohesive and efficient design.
    Your responsibilities include:
    - Reviewing and integrating input from the material and load-bearing agents.
    - Ensuring the design meets all specified requirements and constraints.
    - Providing guidance and making final decisions where necessary.
    """,
    human_input_mode="NEVER"
)



writer = autogen.AssistantAgent(
    name="Writer",
    llm_config=llm_config,
    system_message="""
    You are a professional writer tasked with providing clear and concise summaries of meetings.
    Summarize the discussions, decisions made, and action points from each meeting.
    Ensure the summary is easy to read and understand for all stakeholders.
    """,
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


def writing_message(recipient, messages, sender, config):
    return f"Polish the content so it is easy to read. \n\n {recipient.chat_messages_for_summary(sender)[-1]['content']}"


# Nested Chat Queue connecting inner agents to load_bearing_agent
nested_chat_queue = [
    {"recipient": inner_manager, "summary_method": "reflection_with_llm"}, 
    {"recipient": writer, "message": writing_message, "summary_method": "last_msg", "max_turns": 1},
    # {"recipient": reviewer, "message": "Review the content provided.", "summary_method": "last_msg", "max_turns": 1},
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



