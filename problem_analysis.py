import os
from typing_extensions import Annotated
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from autogen import (
    AssistantAgent,
    UserProxyAgent,
    Cache,
    register_function,
)
from autogen.coding import LocalCommandLineCodeExecutor

# Configuration list for the language model
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

# Create planner agent
planner = AssistantAgent(
    name="planner",
    llm_config=gemma_config,
    system_message=(
        "You are a helpful AI assistant specialized in structural engineering. "
        "Your task is to decompose the process of designing a parking garage into three main steps: "
        "1. Problem analysis "
        "2. Choosing a load-bearing system "
        "3. Material choice. "
        "If the plan is not good, suggest a better plan. "
        "If the execution is wrong, analyze the error and suggest a fix."
        "Only name the main steps, do not provide context."
    ),
)

# Create a planner user agent to interact with the planner
planner_user = UserProxyAgent(
    name="planner_user",
    human_input_mode="NEVER",
    code_execution_config=False,
)

# Function for asking the planner
def task_planner(question: Annotated[str, "Question to ask the planner."]) -> str:
    with Cache.disk(cache_seed=4) as cache:
        planner_user.initiate_chat(planner, message=question, max_turns=1, cache=cache)
    # Return the last message received from the planner
    return planner_user.last_message()["content"]

# Create assistant agent
assistant = AssistantAgent(
    name="assistant",
    system_message=(
        "You are a helpful AI assistant specialized in structural engineering. "
        "You can use the task planner to decompose the design process of a parking garage into three main steps: "
        "1. Problem analysis "
        "2. Choosing a load-bearing system "
        "3. Material choice. "
        "Make sure you follow through the sub-tasks. "
        "Only name the main steps, do not provide context. "
        "Return TERMINATE only if the sub-tasks are completed."
    ),
    llm_config=gemma_config,
)

# Setting up code executor
os.makedirs("planning", exist_ok=True)
code_executor = LocalCommandLineCodeExecutor(work_dir="planning")

# Create user proxy agent to interact with the assistant
user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="ALWAYS",
    is_termination_msg=lambda x: "content" in x
    and x["content"] is not None
    and x["content"].rstrip().endswith("TERMINATE"),
    code_execution_config={"executor": code_executor},
)

# Register the function to the agent pair
register_function(
    task_planner,
    caller=assistant,
    executor=user_proxy,
    name="task_planner",
    description="A task planner that can help you with decomposing the design process of a parking garage into sub-tasks.",
)

# Function to create PDF with the assistant's message
def create_pdf(message: str, filename: str = "assistant_message.pdf"):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    Story = []

    for line in message.split('\n'):
        if line.startswith("1.") or line.startswith("2.") or line.startswith("3."):
            Story.append(Spacer(1, 12))
            Story.append(Paragraph(line, styles['Heading2']))
        else:
            bullet_points = line.split(", ")
            bullet_list = ListFlowable(
                [ListItem(Paragraph(bp, styles['BodyText'])) for bp in bullet_points],
                bulletType='bullet',
            )
            Story.append(bullet_list)
            Story.append(Spacer(1, 12))

    doc.build(Story)

# Use Cache.disk to cache LLM responses
task = "The design of a parking garage for the conceptual design phase in structural engineering"
with Cache.disk(cache_seed=1) as cache:
    # The assistant receives a message from the user, which contains the task description
    user_proxy.initiate_chat(
        assistant,
        message=task,
        cache=cache,
    )

# Generate the PDF with the assistant's message to the user_proxy
assistant_message = user_proxy.last_message()["content"]
create_pdf(assistant_message, "assistant_message.pdf")

print("PDF created with the assistant's message.")
