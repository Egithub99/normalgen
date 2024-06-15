
import os
import fitz  # PyMuPDF
from autogen import ConversableAgent
from autogen.agentchat.contrib.capabilities.text_compressors import LLMLingua
from autogen.agentchat.contrib.capabilities.transforms import TextMessageCompressor
import autogen

# Specify the path to the task decomposition PDF file
TASK_DECOMPOSITION_PDF_PATH = 'task_decomposition.pdf'
CONTEXT_LENGTH_LIMIT = 6000  # Define the context length limit

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

# LLM configuration
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

# Define the planner agent
planner_agent = ConversableAgent(
    name="Planner_Agent",
    system_message="""Planner. Specialized in the conceptual design phase of structural engineering.
                       Based on the provided steps from the PDF, please list the steps needed for the conceptual design phase in structural engineering. 
                        Please adhere to the following instructions:
                        \n1. Only select the steps from the task_decomposition.pdf.
                        \n2. You cannot describe any other steps""",
    llm_config=gemma_config,
)

# Define the task initiator agent
task_initiator_agent = ConversableAgent(
    name="Task_Initiator_Agent",
    system_message="Assistant. Initiating the task for conceptual design.",
    llm_config=gemma_config,
)

# Initiate the conversation with the planner agent
chat_result = task_initiator_agent.initiate_chat(
    planner_agent,
    message=f"Please provide the steps for the conceptual design phase of structural engineering based on the following document:\n{compressed_task_decomposition_content}",
    summary_method="reflection_with_llm",
    max_turns=2,
)

# Print the chat result
print(chat_result)
