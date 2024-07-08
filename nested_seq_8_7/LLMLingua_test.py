import tempfile

import fitz  # PyMuPDF
import requests

from autogen.agentchat.contrib.capabilities.text_compressors import LLMLingua
from autogen.agentchat.contrib.capabilities.transforms import TextMessageCompressor

AUTOGEN_PAPER = "https://arxiv.org/pdf/2308.08155"


def extract_text_from_pdf():
    # Download the PDF
    response = requests.get(AUTOGEN_PAPER)
    response.raise_for_status()  # Ensure the download was successful

    text = ""
    # Save the PDF to a temporary file
    with tempfile.TemporaryDirectory() as temp_dir:
        with open(temp_dir + "temp.pdf", "wb") as f:
            f.write(response.content)

        # Open the PDF
        with fitz.open(temp_dir + "temp.pdf") as doc:
            # Read and extract text from each page
            for page in doc:
                text += page.get_text()

    return text


# Example usage
pdf_text = extract_text_from_pdf()

llm_lingua = LLMLingua()
text_compressor = TextMessageCompressor(
    text_compressor=llm_lingua,
    cache=None)
compressed_text = text_compressor.apply_transform([{"content": pdf_text}])

print(text_compressor.get_logs([], []))


result = user_proxy.initiate_chat(recipient=researcher, clear_history=True, message=message, silent=True)

print(result.chat_history[1]["content"])