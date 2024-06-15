import os
import tempfile
import autogen

# Temporary directory for code execution
temp_dir = tempfile.gettempdir()

# LLM configuration for Gemma
gemma_config = {
    "config_list": [
        {
            "model": "lmstudio-ai/gemma-2b-it-GGUF/gemma-2b-it-q8_0.gguf",
            "base_url": "http://localhost:1234/v1",
            "api_key": "lm-studio",
        },
    ],
    "cache_seed": None,  # Disable caching.
}

# User Proxy Agent
user_proxy = autogen.ConversableAgent(
    name="User_proxy",
    system_message="A human admin asking for the design of a parking garage.",
    llm_config=False,
    human_input_mode="TERMINATE",
    code_execution_config={"use_docker": False, "work_dir": temp_dir},
)

# Lead Engineer Agent
lead_engineer = autogen.ConversableAgent(
    name="Lead_engineer",
    system_message="You are the lead engineer responsible for overseeing the design process of the parking garage.",
    llm_config=gemma_config,
    human_input_mode="NEVER",
)

# Planner Agent
planner = autogen.ConversableAgent(
    name="Planner",
    system_message=(
        "You are a helpful AI assistant specialized in structural engineering. "
        "Your task is to decompose the process of designing a parking garage into three main steps: "
        "1. Problem analysis "
        "2. Choosing a load-bearing system "
        "3. Material choice. "
        "Only name the main steps, do not provide context."
    ),
    llm_config=gemma_config,
    human_input_mode="NEVER",
)

# Register nested chats for lead_engineer
lead_engineer.register_nested_chats(
    [
        {
            "recipient": planner,
            "message": "Provide the main steps for designing a parking garage.",
            "summary_method": "reflection_with_llm",
            "max_turns": 1
        }
    ],
    trigger=lambda sender: sender not in [planner],
)

# Example interaction function
def run_interaction():
    # User_proxy asks the initial question to lead_engineer
    user_proxy_message = {"role": "user", "content": "Design a simple parking garage"}
    lead_engineer_reply = user_proxy.generate_reply(messages=[user_proxy_message])
    print("Reply from Lead Engineer:", lead_engineer_reply)

    # Lead_engineer asks the planner for the steps
    lead_engineer_message = {"role": "assistant", "content": "What are the planning steps for designing a parking garage?"}
    planner_reply = lead_engineer.generate_reply(messages=[lead_engineer_message])
    print("Reply from Planner:", planner_reply)

    # Lead_engineer reports back to user_proxy
    lead_engineer_summary = {"role": "assistant", "content": planner_reply}
    final_reply = user_proxy.generate_reply(messages=[lead_engineer_summary])
    print("Final Reply from Lead Engineer to User Proxy:", final_reply)

# Run the interaction
run_interaction()

