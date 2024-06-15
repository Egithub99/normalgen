import os
import fitz  # PyMuPDF
from autogen.agentchat.contrib.capabilities.text_compressors import LLMLingua
from autogen.agentchat.contrib.capabilities.transforms import TextMessageCompressor
import autogen

# Specify the path to the task decomposition PDF file
TASK_DECOMPOSITION_PDF_PATH = 'task_decomposition.pdf'
CONTEXT_LENGTH_LIMIT = 2048  # Define the context length limit

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# Extract text from the task decomposition PDF file
task_decomposition_text = extract_text_from_pdf(TASK_DECOMPOSITION_PDF_PATH)

# Compress the extracted text
llm_lingua = LLMLingua()
text_compressor = TextMessageCompressor(text_compressor=llm_lingua)
compressed_task_decomposition_text = text_compressor.apply_transform([{"content": task_decomposition_text}])

# Log the compressed text process
print("Task Decomposition compression logs:", text_compressor.get_logs([], []))

# Truncate the text if it exceeds the context length limit
compressed_task_decomposition_content = compressed_task_decomposition_text[0]['content']
if len(compressed_task_decomposition_content) > CONTEXT_LENGTH_LIMIT:
    compressed_task_decomposition_content = compressed_task_decomposition_content[:CONTEXT_LENGTH_LIMIT]


gemma_config = {
    "config_list": [
        {
            "model": "lmstudio-ai/gemma-2b-it-GGUF/gemma-2b-it-q8_0.gguf",
            "base_url": "http://localhost:1234/v1",
            "api_key": "lm-studio",
        },
    ],
    "cache_seed": None, # Disable caching.
    "temperature": 0.1,  
}

# Define the agents
user_proxy = autogen.UserProxyAgent(
    name="Admin",
    system_message="A human admin. Give the task, and send instructions to the Engineer.",
    code_execution_config=False,
)

planner = autogen.AssistantAgent(
    name="Planner",
    system_message="""Planner.
    "You are an expert in structural engineering. 
    "Based on the provided steps from the PDF (task_decomposition.pdf), please describe the steps needed for the conceptual design phase in structural engineering. "
    "Please adhere to the following instructions:"
    "\n1. Only select the steps from the task_decomposition.pdf."
    "\n2. You cannot describe any other steps."
    """,
    llm_config=gemma_config,
)

engineer = autogen.AssistantAgent(
    name="Engineer",
    system_message= """Structural Engineer
    "Expert in structural engineering. Based on the provided steps from the PDF, "
    "please describe the steps needed for the conceptual design phase in structural engineering. "
    "Please adhere to the following instructions:"
    "\n1. Only select the steps from the task_decomposition.pdf."
    "\n2. You cannot describe any other steps.""",
    llm_config=gemma_config,
)

writer = autogen.AssistantAgent(
    name="Writer",
    llm_config=gemma_config,
    system_message="""Writer. Please write blogs in markdown format (with relevant titles) and put the content in pseudo ```md``` code block. You will write it for a task based on previous chat history. """,
)

def custom_speaker_selection_func(last_speaker, groupchat):
    messages = groupchat.messages

    if len(messages) <= 1:
        return planner

    if last_speaker is planner:
        return engineer

    elif last_speaker is engineer:
        return writer

    elif last_speaker is writer:
        return user_proxy

    else:
        return "auto"

groupchat = autogen.GroupChat(
    agents=[user_proxy, planner, engineer, writer],
    messages=[],
    max_round=20,
    speaker_selection_method=custom_speaker_selection_func,
)

manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=gemma_config)

with autogen.Cache.disk(cache_seed=41) as cache:
    task = "Please describe the steps needed for the conceptual design phase in structural engineering."
    groupchat_history_custom = user_proxy.initiate_chat(
        manager,
        message=task,
        cache=cache,
    )

print(groupchat_history_custom)
