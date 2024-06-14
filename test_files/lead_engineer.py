import autogen

# Define the configuration for the agents
llm_config = {
    "model": "lmstudio-ai/gemma-2b-it-GGUF/gemma-2b-it-q8_0.gguf",
    "base_url": "http://localhost:1234/v1",
    "api_key": "lm-studio",
}

# Initialize the agent
lead_engineer_agent = autogen.AssistantAgent(
    name="LeadEngineerAgent",
    llm_config=llm_config,
    system_message="""
    You are a highly skilled Structural Engineer. Your primary role is to assist users in planning and designing structures.
    When interacting with users, you first ask what kind of structure they want to build.
    After they provide the type of structure, you inform them that a selection of materials is needed for construction.
    """
)

# Define the workflow steps
def workflow():
    # Ask the user about the structure they want to build
    structure_type = lead_engineer_agent.execute(
        task="Ask the user what kind of structure they want to build."
    )
    
    # Inform the user about selecting materials for construction
    lead_engineer_agent.execute(
        task="Tell the user that a selection of materials is needed for construction.",
        context=structure_type
    )

# Execute the workflow
workflow()
