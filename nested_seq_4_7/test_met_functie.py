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

# Define the Lead Engineer agent
lead_engineer = autogen.AssistantAgent(
    name="Lead_Engineer",
    system_message="""
                    You are the lead engineer responsible for developing structural conceptual designs. 
                    Focus only on the structural system.
                    Consult the Material Agent and Load-Bearing Agent for their expertise and make the final decisions.""",
    llm_config=llm_config,
)

# Define the Material Agent
material_agent = autogen.AssistantAgent(
    name="Material_Agent",
    system_message="You are an expert in materials for construction projects. Provide the best material options based on the requirements provided to you.",
    llm_config=llm_config,
)

# Define the Load-Bearing Agent
load_bearing_agent = autogen.AssistantAgent(
    name="Load_Bearing_Agent",
    system_message="You are an expert in load-bearing systems. Based on the chosen material, provide the best load-bearing system options according to the requirements provided to you.",
    llm_config=llm_config,
)

# Define the User Proxy Agent
user_proxy = autogen.UserProxyAgent(
    name="User_Proxy",
    system_message="A human admin overseeing the process.",
    code_execution_config=False,
    human_input_mode="TERMINATE"
)

# Set up the group chat
groupchat = autogen.GroupChat(
    agents=[user_proxy, lead_engineer, material_agent, load_bearing_agent],
    messages=[],
    max_round=12
)

# Create the GroupChatManager
manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config
)

# Define tasks
tasks = [
    "Develop a structural conceptual design of a 2-story parking garage with dimensions 20 by 20 meters.",
    "Please recommend the best material for constructing a 2-story parking garage with dimensions 20 by 20 meters.",
    "Based on the chosen material, please recommend the best load-bearing system for the parking garage."
]

# Function to simulate the workflow
def simulate_workflow():
    # Step 1: Lead Engineer initiates the task
    res = manager.initiate_chats(
        [
            {
                "recipient": lead_engineer,
                "message": tasks[0],
                "max_turns": 1,
                "summary_method": "last_msg",
            }
        ]
    )

    # Step 2: Lead Engineer asks Material Agent for material recommendation
    res = manager.initiate_chats(
        [
            {
                "recipient": material_agent,
                "message": tasks[1],
                "max_turns": 1,
                "summary_method": "last_msg",
            }
        ]
    )
    # material_recommendation = manager.groupchat.messages[-1]["content"]  # Extract the last message content

    # Step 3: Lead Engineer reviews the material recommendation
    res = manager.initiate_chats(
        [
            {
                "recipient": lead_engineer,
                "message": f"Review the material recommendation.",
                "max_turns": 1,
                "summary_method": "last_msg",
            }
        ]
    )

    # Step 4: Lead Engineer asks Load-Bearing Agent for load-bearing system based on the material
    res = manager.initiate_chats(
        [
            {
                "recipient": load_bearing_agent,
                "message": f"Based on the material, please recommend the best load-bearing system for the parking garage.",
                "max_turns": 1,
                "summary_method": "last_msg",
            }
        ]
    )
    # load_bearing_recommendation = manager.groupchat.messages[-1]["content"]  # Extract the last message content

    # Step 5: Lead Engineer reviews the load-bearing system recommendation
    res = manager.initiate_chats(
        [
            {
                "recipient": lead_engineer,
                "message": f"Review the load-bearing system recommendation.",
                "max_turns": 1,
                "summary_method": "last_msg",
            }
        ]
    )

    # Final step: Lead Engineer finalizes the structural conceptual design
    res = manager.initiate_chats(
        [
            {
                "recipient": user_proxy,
                "message": f"Finalize the structural conceptual design with the following choices: Material, Load-Bearing System.",
                "max_turns": 1,
                "summary_method": "last_msg",
            }
        ]
    )

# Run the simulation
simulate_workflow()

