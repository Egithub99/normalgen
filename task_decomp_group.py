from autogen import UserProxyAgent, AssistantAgent, GroupChat, GroupChatManager, Cache

# Configuration for the agents
# llm_config = {"config_list": config_list_gpt4, "cache_seed": 42}
llm_config = {
    "config_list": [
        {
            "model": "lmstudio-ai/gemma-2b-it-GGUF/gemma-2b-it-q8_0.gguf",
            "base_url": "http://localhost:1234/v1",
            "api_key": "lm-studio",
        },
    ],
}

# User Proxy Agent
user_proxy = UserProxyAgent(
    name="Admin",
    system_message="A human admin overseeing the task of conceptual design of a simple parking garage in structural engineering. Provide the task, and send instructions to planner, engineer, and writer for refinement.",
    code_execution_config=False,
)

# Planner Agent
planner = AssistantAgent(
    name="Planner",
    system_message="""Planner. 
                    You are a helpful AI assistant specialized in structural engineering.
                    Given a task, determine what information is needed to complete the conceptual design phase of a structural engineering project.
                    Focus on identifying tasks that can be detailed and executed without code execution.
            """,
    llm_config=llm_config,
)

# Engineer Agent
engineer = AssistantAgent(
    name="Engineer",
    system_message="""Strucutral Engineer. 
                    Decompose the conceptual design task into detailed steps, focusing on structural engineering principles and practices. Provide a comprehensive task list that can be used to guide the design process.
""",
    llm_config=llm_config,
)

# Writer Agent
writer = AssistantAgent(
    name="Writer",
    system_message="""Writer. Write a detailed description of the task decomposition for the conceptual design phase in structural engineering.
                     Use markdown format and place the content in a pseudo ```md``` code block. Include relevant titles and ensure the content is clear and informative.
""",
    llm_config=llm_config,
)

# Group Chat Setup
groupchat = GroupChat(
    agents=[user_proxy, planner, engineer, writer],
    messages=[
        {"role": "system", "content": "You are in a role play game. The following roles are available:\n\
Admin: An attentive HUMAN user who can answer questions about the task, and can perform tasks such as running Python code or inputting command line commands at a Linux terminal and reporting back the execution results.\n\
Planner: Planner. Given a task, determine what information is needed to complete the conceptual design phase of a structural engineering project.\n\
Focus on identifying tasks that can be detailed and executed without code execution.\n\
Engineer: Structural Engineer. Decompose the conceptual design task into detailed steps, focusing on structural engineering principles and practices. Provide a comprehensive task list that can be used to guide the design process.\n\
Writer: Writer. Write a detailed description of the task decomposition for the conceptual design phase in structural engineering. Use markdown format and place the content in a pseudo ```md``` code block. Include relevant titles and ensure the content is clear and informative.\n\
Read the following conversation. Then select the next role from ['Admin', 'Planner', 'Engineer', 'Writer'] to play. Only return the role."},
        {"role": "user", "name": "Admin", "content": "Please create a task decomposition for the conceptual design phase of a structural engineering project."}
    ],
    max_round=2,
    speaker_selection_method="round_robin",
)

# Group Chat Manager
manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# Use Cache.disk to cache LLM responses. Change cache_seed for different responses.
with Cache.disk(cache_seed=41) as cache:
    chat_history = user_proxy.initiate_chat(
        manager,
        message="Please create a task decomposition for the conceptual design of a simple parking garage for a structural engineering project.",
        cache=cache,
    )

