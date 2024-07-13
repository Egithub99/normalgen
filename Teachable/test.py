from autogen import UserProxyAgent, config_list_from_json
from autogen.agentchat.contrib.capabilities.teachability import Teachability
from autogen import ConversableAgent  # As an example


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


# Start by instantiating any agent that inherits from ConversableAgent, which we use directly here for simplicity.
teachable_agent = ConversableAgent(
    name="teachable_agent",  # The name can be anything.
    llm_config=llm_config
)

# Instantiate a Teachability object. Its parameters are all optional.
teachability = Teachability(
    reset_db=False,  # Use True to force-reset the memo DB, and False to use an existing DB.
    path_to_db_dir="./storage_testing5"  # Can be any path, but teachable agents in a group chat require unique paths.
)

# Now add teachability to the agent.
teachability.add_to_agent(teachable_agent)

# For this test, create a user proxy agent as usual.
user = UserProxyAgent("user",
                       human_input_mode="ALWAYS",
                       code_execution_config=False,
                       )

# This function will return once the user types 'exit'.
user.initiate_chat(teachable_agent, 
                              message="What are the six key steps in the conceptual design?")