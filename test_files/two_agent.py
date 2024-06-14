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



# Initialize the agents
lead_engineer_agent = autogen.ConversableAgent(
    name="LeadEngineerAgent",
    system_message="You are a Lead Engineer tasked with making decisions about the materials to be used in the design of structures, such as parking garages. You will discuss with the TimberAgent and SteelAgent to gather insights on the suitability of timber and steel for the project.",
    llm_config=llm_config
)

timber_agent = autogen.ConversableAgent(
    name="TimberAgent",
    system_message="You are a Timber Specialist with expertise in timber. You provide detailed explanations on the suitability, benefits, and drawbacks of using timber for structures such as parking garages.",
    llm_config=llm_config
)

steel_agent = autogen.ConversableAgent(
    name="SteelAgent",
    system_message="You are a Steel Specialist with expertise in steel. You provide detailed explanations on the suitability, benefits, and drawbacks of using steel for structures such as parking garages.",
    llm_config=llm_config
)

# Define the workflow steps
def workflow():
    # Start the conversation with the Timber Agent
    lead_engineer_agent.initiate_chat(
        recipient=timber_agent,
        message="We are designing a new parking garage. Can you provide your insights on whether we should use timber for the structure?"
    )

    # Timber Agent responds
    timber_response = timber_agent.get_response()
    print(f"TimberAgent: {timber_response}")

    # Start the conversation with the Steel Agent
    lead_engineer_agent.initiate_chat(
        recipient=steel_agent,
        message="We are designing a new parking garage. Can you provide your insights on whether we should use steel for the structure?"
    )

    # Steel Agent responds
    steel_response = steel_agent.get_response()
    print(f"SteelAgent: {steel_response}")

    # Lead Engineer makes a decision based on responses
    if "cost-effective" in timber_response.lower() and "durability" in steel_response.lower():
        decision = "We will use steel for the parking garage due to its durability."
    elif "sustainability" in timber_response.lower():
        decision = "We will use timber for the parking garage due to its sustainability."
    else:
        decision = "Based on the provided insights, we will use steel for the parking garage."

    print(f"LeadEngineerAgent Decision: {decision}")

# Execute the workflow
workflow()
