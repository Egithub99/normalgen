import os
from autogen import ConversableAgent
from agent_config import gemma_config
from planner_agent import planner_agent
from load_bearing_agent import load_bearing_agent

# Define the Lead Engineer agent
lead_engineer_agent = ConversableAgent(
    name="Lead_Engineer_Agent",
    system_message="""Lead Engineer: Ensures the engineering integrity and supervises all technical activities.
                    Keeps oversight over the project.
                    Know the planning and ask advice to experts.
                      """,
    llm_config=gemma_config,
    human_input_mode="NEVER",
)

# Start a sequence of two-agent chats
chat_results = lead_engineer_agent.initiate_chats(
    [
        {
            "recipient": planner_agent,
            "message": "Start planning the design of the parking garage.",
            "max_turns": 1,
            "summary_method": "last_msg",
        },
        {
            "recipient": load_bearing_agent,
            "message": "Choose an appropriate load-bearing system.",
            "max_turns": 1,
            "summary_method": "last_msg",
        },
    ]
)

# Print the responses from the agents
for result in chat_results:
    if hasattr(result, 'conversation') and result.conversation:
        print(f"{result.recipient.name} Response:", result.conversation[-1]['content'])
