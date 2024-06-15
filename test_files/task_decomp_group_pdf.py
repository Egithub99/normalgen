import os
import fitz  # PyMuPDF
from autogen.agentchat.contrib.capabilities.text_compressors import LLMLingua
from autogen.agentchat.contrib.capabilities.transforms import TextMessageCompressor
from autogen import UserProxyAgent, AssistantAgent, GroupChat, GroupChatManager, Cache

# Configuration for the agents
llm_config = {
    "config_list": [
        {
            "model": "lmstudio-ai/gemma-2b-it-GGUF/gemma-2b-it-q8_0.gguf",
            "base_url": "http://localhost:1234/v1",
            "api_key": "lm-studio",
        },
    ],
}

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# Compress extracted text
def compress_text(text, context_length_limit=6000):
    llm_lingua = LLMLingua()
    text_compressor = TextMessageCompressor(text_compressor=llm_lingua)
    compressed_text = text_compressor.apply_transform([{"content": text}])
    compressed_content = compressed_text[0]['content']
    if len(compressed_content) > context_length_limit:
        compressed_content = compressed_content[:context_length_limit]
    return compressed_content

# Extract and compress the text from the task_decomposition.pdf
PDF_PATH = 'task_decomposition.pdf'  # Ensure the file is in the same directory as the script
extracted_text = extract_text_from_pdf(PDF_PATH)
compressed_text = compress_text(extracted_text)

# User Proxy Agent
user_proxy = UserProxyAgent(
    name="Admin",
    system_message="A human admin overseeing the task of conceptual design of a simple parking garage in structural engineering. Provide the task, and send instructions to planner, engineer, and writer for refinement.",
    code_execution_config=False,
)

# Planner Agent
planner = AssistantAgent(
    name="Planner",
    system_message=f"""Planner. 
                    You are a helpful AI assistant specialized in structural engineering.
                    Given the task and the content from the PDF (task_decomposition.pdf), determine what information is needed to complete the conceptual design phase of a structural engineering project.
                    Focus on identifying tasks that can be detailed and executed without code execution.
                    Here is the content from the PDF to guide you:
                    {compressed_text}
                    Base your response solely on the content from the PDF.
            """,
    llm_config=llm_config,
)

# Engineer Agent
engineer = AssistantAgent(
    name="Engineer",
    system_message=f"""Structural Engineer. 
                    Decompose the conceptual design task into detailed steps, focusing on structural engineering principles and practices. 
                    You decompose based on the PDF provided to you.
                    Provide a comprehensive task list that can be used to guide the design process.
                    Here is the content from the PDF to guide you:
                    {compressed_text}
                    Base your response solely on the content from the PDF and only include the steps mentioned in it.
""",
    llm_config=llm_config,
)

# Writer Agent
writer = AssistantAgent(
    name="Writer",
    system_message=f"""Writer. Write a detailed description of the task decomposition for the conceptual design phase in structural engineering.
                     Use markdown format and place the content in a pseudo ```md``` code block. Include relevant titles and ensure the content is clear and informative.
                     Here is the content from the PDF to guide you:
                     {compressed_text}
                     Base your response solely on the content from the PDF.
""",
    llm_config=llm_config,
)

# Group Chat Setup
groupchat = GroupChat(
    agents=[user_proxy, planner, engineer, writer],
    messages=[
        {"role": "system", "content": "You are in a role play game. The following roles are available:\n\
Admin: An attentive HUMAN user who can answer questions about the task, and can perform tasks such as running Python code or inputting command line commands at a Linux terminal and reporting back the execution results.\n\
Planner: Planner. Given a task, determine what information is needed to complete the conceptual design phase of a structural engineering project.\n\
Focus on identifying tasks that can be detailed and executed without code execution.\n\
Engineer: Structural Engineer. Decompose the conceptual design task into detailed steps, focusing on structural engineering principles and practices. Provide a comprehensive task list that can be used to guide the design process.\n\
Writer: Writer. Write a detailed description of the task decomposition for the conceptual design phase in structural engineering. Use markdown format and place the content in a pseudo ```md``` code block. Include relevant titles and ensure the content is clear and informative.\n\
Read the following conversation. Then select the next role from ['Admin', 'Planner', 'Engineer', 'Writer'] to play. Only return the role."},
        {"role": "user", "name": "Admin", "content": "Please create a task decomposition for the conceptual design phase of a structural engineering project."}
    ],
    max_round=2,
    speaker_selection_method="round_robin",
)

# Group Chat Manager
manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# Use Cache.disk to cache LLM responses. Change cache_seed for different responses.
with Cache.disk(cache_seed=41) as cache:
    chat_history = user_proxy.initiate_chat(
        manager,
        message="Please create a task decomposition for the conceptual design of a simple parking garage for a structural engineering project.",
        cache=cache,
    )
