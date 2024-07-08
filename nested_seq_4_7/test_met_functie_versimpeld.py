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

# Define tasks
tasks = [
    "Develop a structural conceptual design of a 2-story parking garage with dimensions 20 by 20 meters. Consult the Material Agent for material selection and the Load-Bearing Agent for load-bearing system selection.",

]

subtasks = [
    "Please recommend the best material for constructing a 2-story parking garage with dimensions 20 by 20 meters.",
    "Based on the chosen material, please recommend the best load-bearing system for the parking garage."
]

lead_engineer = autogen.ConversableAgent(
    name="Lead_engineer",
        system_message=f"""You are the lead engineer responsible for: {tasks[0]}"""
                      ,
    llm_config=llm_config,
    human_input_mode="ALWAYS",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
)


# Define the Material Agent
material_agent = autogen.AssistantAgent(
    name="Material_Agent",
    system_message="""You are an expert in materials for construction projects. 
                      Provide the best material option based on the requirements provided to you.
                    """,
    llm_config=llm_config,
)

# Define the Load-Bearing Agent
load_bearing_agent = autogen.AssistantAgent(
    name="Load_Bearing_Agent",
    system_message="""You are an expert in load-bearing systems. 
                      Based on the chosen material, provide the best load-bearing system option according to the requirements provided to you.
                    """,
    llm_config=llm_config,
)

# Define the User Proxy Agent
user_proxy = autogen.UserProxyAgent(
    name="User_Proxy",
    system_message="A human admin overseeing the process.",
    code_execution_config=False,
    human_input_mode="ALWAYS"  # Allow human input at all stages
)

writer = autogen.AssistantAgent(
    name="writer",
    llm_config=llm_config,
    system_message="""
        You are a professional writer.
        Reply "TERMINATE" in the end when everything is done.
        """,
)





#De max_turns can be set to 2 to allow for human feedback.
#However if no human feedback is allowed then the chat continues, while actually it needs to stop.
# Start the sequence of chats with extended max_turns for feedback incorporation
chat_results = lead_engineer.initiate_chats(
    [
        {
            "recipient": material_agent,
            "message": subtasks[0],
            "max_turns": 1,  
            "summary_method": "last_msg",
        },
        {
            "recipient": load_bearing_agent,
            "message": subtasks[1],
            "max_turns": 1,
            "summary_method": "last_msg",
        },
                {
            "recipient": writer,
            "message": "Summarize the conversation",
            "max_turns": 1,
            "summary_method": "reflection_with_llm",
        },
    ]
)