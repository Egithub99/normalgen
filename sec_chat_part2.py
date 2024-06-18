import os
import fitz  # PyMuPDF
from autogen.agentchat.contrib.capabilities.text_compressors import LLMLingua
from autogen.agentchat.contrib.capabilities.transforms import TextMessageCompressor
import autogen
from autogen import ConversableAgent

# Configuration for the LLM
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

# Define the paths to the PDF files
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

# Truncate the texts if they exceed the context length limit
compressed_theory_content = compressed_theory_text[0]['content']
if len(compressed_theory_content) > CONTEXT_LENGTH_LIMIT:
    compressed_theory_content = compressed_theory_content[:CONTEXT_LENGTH_LIMIT]

compressed_table_content = compressed_table_text[0]['content']
if len(compressed_table_content) > CONTEXT_LENGTH_LIMIT:
    compressed_table_content = compressed_table_content[:CONTEXT_LENGTH_LIMIT]

# Define system message for the load-bearing agent
system_message = (
    "You are an expert in structural engineering. Based on the provided theory from the first document, "
    "choose the most suitable construction type from the 'Construction type' column in Table 9.7: Approaches to Disproportionate Collapse, provided in the second document. "
    "This selection is for a simple parking garage. "
    "Please adhere to the following instructions:"
    "\n1. Only select an option from the 'Construction type' column in Table 9.7."
    "\n2. Mention the text of the chosen option from the 'Construction type' column. "
    "\n3. Select the corresponding 'Building class' from the same row as the chosen 'Construction type'. "
    "\n4. Mention the number below 'Building class', as well as the text in this column. "
    "\n5. Provide a brief explanation for your choice of this construction type and building class, considering the provided theory."
)

# Define the load-bearing agent
load_bearing_agent = autogen.ConversableAgent(
    name="load_bearing_assistant",
    llm_config=gemma_config,
    max_consecutive_auto_reply=1,
    system_message=system_message,
    human_input_mode="NEVER",
)

# Define the Lead Engineer agent
lead_engineer_agent = ConversableAgent(
    name="Lead_Engineer_Agent",
    system_message="You are the lead engineer. You know the planning and move on to the next conversation.",
    llm_config=gemma_config,
    human_input_mode="NEVER",
)

# Function to choose the load-bearing system
def choose_load_bearing_system(agent, theory_text, table_text):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": "Here is the relevant theory from the PDF:\n" + theory_text},
        {"role": "user", "content": "Here is the table (Table 9.7) from which you should choose an option. "
                                    "Mention the description from the first column of your chosen option and the appropriate building class:\n" + table_text}
    ]
    response = agent.generate_reply(messages)
    return response

# Lead Engineer initiates a chat with the Load Bearing Agent to solve the second question
chat_results = lead_engineer_agent.initiate_chats(
    [
        {
            "recipient": load_bearing_agent,
            "message": "We need to choose a load-bearing system for the parking garage. Here is the relevant theory:\n" + compressed_theory_content,
            "max_turns": 3,
            "summary_method": "last_msg",
        },
        {
            "recipient": load_bearing_agent,
            "message": "Here is the table (Table 9.7) from which you should choose an option. "
                       "Mention the description from the first column of your chosen option and the appropriate building class:\n" + compressed_table_content,
            "max_turns": 3,
            "summary_method": "last_msg",
        }
    ]
)

# Print the response from the load-bearing agent
for result in chat_results:
    print(result["conversation"][-1]["content"])
