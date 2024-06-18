import os
from autogen import ConversableAgent
from agent_config import gemma_config
from planner_agent import planner_agent
from load_bearing_agent import load_bearing_agent, compressed_theory_content, compressed_table_content, table_text

# Define the Lead Engineer agent
lead_engineer_agent = ConversableAgent(
    name="Lead_Engineer_Agent",
    system_message="""You are the lead engineer.
                    You keep oversight over the project.
                    You know the planning and ask advice to experts.""",
    llm_config=gemma_config,
    human_input_mode="NEVER",
)

try:
    # Start a sequence of two-agent chats with Planner Agent (Agent B)
    chat_results_planner = lead_engineer_agent.initiate_chats(
        [
            {
                "recipient": planner_agent,
                "message": "Start planning the design of the parking garage.",
                "max_turns": 1,
                "summary_method": "last_msg",
            }
        ]
    )

    # Print the response from the Planner Agent
    for result in chat_results_planner:
        if hasattr(result, 'conversation') and result.conversation:
            print("Planner Agent Response:", result.conversation[-1]['content'])

    # Lead Engineer initiates a chat with the Load Bearing Agent to solve the second question
    chat_results_load_bearing = lead_engineer_agent.initiate_chats(
        [
            {
                "recipient": load_bearing_agent,
                "message": "We need to choose a load-bearing system for the parking garage. Here is the relevant theory:\n" + compressed_theory_content,
                "max_turns": 1,
                "summary_method": "last_msg",
            },
            {
                "recipient": load_bearing_agent,
                "message": "Here is the table (Table 9.7) from which you should choose an option. "
                           "Mention the description from the first column of your chosen option and the appropriate building class:\n" + table_text,
                "max_turns": 1,
                "summary_method": "last_msg",
            }
        ]
    )

    # Print the response from the Load Bearing Agent
    for result in chat_results_load_bearing:
        if hasattr(result, 'conversation') and result.conversation:
            print("Load Bearing Agent Response:", result.conversation[-1]['content'])

except Exception as e:
    print(f"An error occurred: {e}")
