from inladen_data import query_engine3
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


# Define the main task
main_task = "Develop a structural conceptual design of a 2-story parking garage with dimensions 20 by 20 meters."

# Define tasks
tasks = [
    f"{main_task} Consult the Material Agent for material selection and the Load-Bearing Agent for load-bearing system selection."
]


# Define subtasks using f-strings to incorporate the main task context
subtasks = [
    f"Please recommend the best material for constructing a building with dimensions similar to the one described in the main task: {main_task}.",
    f"Based on the chosen material, please recommend the best load-bearing system for a building similar to the one described in the main task: {main_task}."
]

response = query_engine3.query(
    f"{subtasks[1]}"
)
print(response)

# Define the Lead Engineer
lead_engineer = autogen.ConversableAgent(
    name="Lead_engineer",
    system_message=f"You are the lead engineer responsible for: {tasks[0]}",
    llm_config=llm_config,
    human_input_mode="ALWAYS",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
)

# Define the Load-Bearing Agent
load_bearing_agent = autogen.AssistantAgent(
    name="Load_Bearing_Agent",
    system_message=f"""You are an expert in load-bearing systems. Based on the chosen material, provide the best load-bearing system option.
                        This is the knowledge to use for the answer: {response}""",
    llm_config=llm_config,
)


# Start the sequence of chats with extended max_turns for feedback incorporation
chat_results = lead_engineer.initiate_chats(
    [
        {
            "recipient": load_bearing_agent,
            "message": subtasks[1],
            "max_turns": 1,
            "summary_method": "last_msg",
        },
    ]
)
