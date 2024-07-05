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

# Set up the group chat
groupchat = autogen.GroupChat(
    agents=[user_proxy, material_agent, load_bearing_agent],
    messages=[],
    max_round=12
)

# Create the GroupChatManager with the system message of the Lead Engineer
manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config,
    system_message="""You are the manager responsible for developing structural conceptual designs. 
                      At this stage, you should only consult the Material Agent for material selection and the Load-Bearing Agent for load-bearing system selection.
                    """,
    human_input_mode="ALWAYS"
)

# Define tasks
tasks = [
    "Develop a structural conceptual design of a 2-story parking garage with dimensions 20 by 20 meters. Consult the Material Agent for material selection and the Load-Bearing Agent for load-bearing system selection.",
    "Please recommend the best material for constructing a 2-story parking garage with dimensions 20 by 20 meters.",
    "Based on the chosen material, please recommend the best load-bearing system for the parking garage."
]

# Manager initiates the task
manager.groupchat.messages.append({
    "sender": "manager",
    "recipient": "User_Proxy",
    "content": tasks[0],
    "turn": 0
})

# Start the sequence of chats with extended max_turns for feedback incorporation
chat_results = manager.initiate_chats(
    [
        {
            "recipient": material_agent,
            "message": tasks[1],
            "max_turns": 1,  
            "summary_method": "last_msg",
        },
        {
            "recipient": load_bearing_agent,
            "message": f"Based on the material recommendation, please recommend the best load-bearing system for the parking garage.",
            "max_turns": 1,
            "summary_method": "last_msg",
        },
        # {
        #     "recipient": user_proxy,
        #     "message": "Review the load-bearing system recommendation.",
        #     "max_turns": 1,
        #     "summary_method": "last_msg",
        # },
        {
            "recipient": user_proxy,
            "message": "Finalize the structural conceptual design with the selected material and load-bearing system.",
            "max_turns": 1,
            "summary_method": "reflection_with_llm",
        }
    ]
)

# # Extract the final results
# material_recommendation = manager.groupchat.messages[1]["content"]
# load_bearing_recommendation = manager.groupchat.messages[2]["content"]

# print("Material Recommendation:", material_recommendation)
# print("Load-Bearing Recommendation:", load_bearing_recommendation)
