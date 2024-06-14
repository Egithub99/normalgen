import os
import fitz  # PyMuPDF
from autogen.agentchat.contrib.capabilities.text_compressors import LLMLingua
from autogen.agentchat.contrib.capabilities.transforms import TextMessageCompressor
import autogen

# Specify the path to the PDF file
PDF_PATH = '9_stability-robustness.pdf'  # Ensure the file is in the same directory as the script
CONTEXT_LENGTH_LIMIT = 6000  # Define the context length limit

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# Extract text from the provided PDF file
pdf_text = extract_text_from_pdf(PDF_PATH)

# Compress the extracted text
llm_lingua = LLMLingua()
text_compressor = TextMessageCompressor(text_compressor=llm_lingua)
compressed_text = text_compressor.apply_transform([{"content": pdf_text}])

# Log the compressed text process
print(text_compressor.get_logs([], []))

# Truncate the text if it exceeds the context length limit
compressed_text_content = compressed_text[0]['content']
if len(compressed_text_content) > CONTEXT_LENGTH_LIMIT:
    compressed_text_content = compressed_text_content[:CONTEXT_LENGTH_LIMIT]

# Define system message and agent configuration
system_message = ''''"You are an expert in structural engineering. 
                    Based on the provided theory, choose the appropriate "Construction type" from 
                    Table 9.7: Approaches to disproportionate collapse for a simple parking garage.
                    You explicity in your choose a construction type from this table only.
                    Besides give a short reasoning as to why you think this is the best solution."'''

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

# Define the load-bearing agent
load_bearing_agent = autogen.ConversableAgent(
    name="load_bearing_assistant",
    llm_config=gemma_config,
    max_consecutive_auto_reply=1,
    system_message=system_message,
    human_input_mode="NEVER",
)

# Function to choose the load-bearing system
def choose_load_bearing_system(agent, compressed_text):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": "Here is the relevant theory from the PDF:\n" + compressed_text + 
         "\n\nPlease choose the appropriate construction type from Table 9.7 for a simple parking garage."}
    ]
    response = agent.generate_reply(messages)
    return response

# Get the response from the load-bearing agent
response = choose_load_bearing_system(load_bearing_agent, compressed_text_content)
print(response)
