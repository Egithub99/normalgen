import os
import fitz  # PyMuPDF
from autogen.agentchat.contrib.capabilities.text_compressors import LLMLingua
from autogen.agentchat.contrib.capabilities.transforms import TextMessageCompressor
import autogen

# Specify the paths to the PDF files
THEORY_PDF_PATH = '9_stability-robustness.pdf'  # Ensure the file is in the same directory as the script
TABLE_PDF_PATH = 'table_9_7.pdf'  # Ensure the file is in the same directory as the script
CONTEXT_LENGTH_LIMIT = 6000  # Define the context length limit

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# Extract text from the provided PDF files
theory_text = extract_text_from_pdf(THEORY_PDF_PATH)
table_text = extract_text_from_pdf(TABLE_PDF_PATH)

# Compress the extracted texts
llm_lingua = LLMLingua()
text_compressor = TextMessageCompressor(text_compressor=llm_lingua)
compressed_theory_text = text_compressor.apply_transform([{"content": theory_text}])
compressed_table_text = text_compressor.apply_transform([{"content": table_text}])

# Log the compressed text process
print("Theory compression logs:", text_compressor.get_logs([], []))
print("Table compression logs:", text_compressor.get_logs([], []))

# Truncate the texts if they exceed the context length limit
compressed_theory_content = compressed_theory_text[0]['content']
if len(compressed_theory_content) > CONTEXT_LENGTH_LIMIT:
    compressed_theory_content = compressed_theory_content[:CONTEXT_LENGTH_LIMIT]

compressed_table_content = compressed_table_text[0]['content']
if len(compressed_table_content) > CONTEXT_LENGTH_LIMIT:
    compressed_table_content = compressed_table_content[:CONTEXT_LENGTH_LIMIT]

# # Define system message and agent configuration
# system_message = (
#     "You are an expert in structural engineering. Based on the provided theory, "
#     "Choose the appropriate construction type from the provided table (Table 9.7: Approaches to disproportionate collapse)."
#     "Do this for a simple parking garage."
#     "Only select an option from the table and mention what is in the first column for this choice. "
#     "Additionally, mention the building class you consider for the parking garage."
#     "Give a brief explanation why you choose this load-bearing system."
# )

system_message = (
    "You are an expert in structural engineering. Based on the provided theory from the first document, "
    "choose the most suitable construction type from Table 9.7: Approaches to Disproportionate Collapse, provided in the second document."
    "This selection is for a simple parking garage."
    "Please adhere to the following instructions:"
    "\n1. Only select an option from the first column from Table 9.7."
    "\n2. Explicity mention the name of the Construction type from the first column of the chosen option (for instance steel frame, concrete frame or timber)."
    "\n3. Specify and name the building class you consider appropriate for the parking garage."
    "\n4. Provide a brief explanation for your choice of this load-bearing system, considering the provided theory in the first document."
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

# Define the load-bearing agent
load_bearing_agent = autogen.ConversableAgent(
    name="load_bearing_assistant",
    llm_config=gemma_config,
    max_consecutive_auto_reply=1,
    system_message=system_message,
    human_input_mode="NEVER",
)

# Function to choose the load-bearing system
def choose_load_bearing_system(agent, theory_text, table_text):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": "Here is the relevant theory from the PDF:\n" + theory_text},
        {"role": "user", "content": "Here is the table (Table 9.7) from which you should choose an option:\n" + table_text}
    ]
    response = agent.generate_reply(messages)
    return response

# Get the response from the load-bearing agent
response = choose_load_bearing_system(load_bearing_agent, compressed_theory_content, compressed_table_content)
print(response)

