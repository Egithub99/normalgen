import os
import fitz  # PyMuPDF
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

# Define system message and agent configuration for the task decomposition agent
# task_decomposition_system_message = (
#     "You are an expert in structural engineering. Based on the provided steps from the PDF, "
#     "please describe the steps needed for the conceptual design phase in structural engineering. "
#     "This includes the following tasks:"
#     "\n1. Problem analysis"
#     "\n2. Schematizing a load-bearing system"
#     "\n3. Material selection"
#     "\nPlease provide a brief explanation for each step based on the provided content."
# )

task_decomposition_system_message = (
    "You are an expert in structural engineering. Based on the provided steps from the PDF, "
    "please describe the steps needed for the conceptual design phase in structural engineering. "
    "You are an expert in structural engineering. Based on the provided steps from PDF, "
    "Please describe the steps needed for the conceptual design phase in structural engineering. "
    "Please adhere to the following instructions:"
    "\n1. Only select the steps from the task_decomposition.pdf."
    "\n2. You cannot describe any other steps."
)


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

# Define the task decomposition agent
task_decomposition_agent = autogen.ConversableAgent(
    name="task_decomposition_assistant",
    llm_config=gemma_config,
    max_consecutive_auto_reply=1,
    system_message=task_decomposition_system_message,
    human_input_mode="NEVER",
)

# Function to describe the task decomposition steps
def describe_task_decomposition_steps(agent, task_decomposition_text):
    messages = [
        {"role": "system", "content": task_decomposition_system_message},
        {"role": "user", "content": "Here is the relevant content from the PDF on task decomposition:\n" + task_decomposition_text}
    ]
    response = agent.generate_reply(messages)
    return response

# Get the response from the task decomposition agent
response = describe_task_decomposition_steps(task_decomposition_agent, compressed_task_decomposition_content)
print(response)
