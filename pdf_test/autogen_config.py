# autogen_config.py
from pdf_reader_tool import PDFReaderTool
import autogen

# New configuration using gemma
gemma = {
    "config_list": [
        {
            "model": "lmstudio-ai/gemma-2b-it-GGUF/gemma-2b-it-q8_0.gguf",
            "base_url": "http://localhost:1234/v1",
            "api_key": "lm-studio",
        },
    ],
    "cache_seed": None,  # Disable caching.
}

user_proxy = autogen.UserProxyAgent(
    name="User_proxy",
    system_message="A human admin.",
    code_execution_config={"last_n_messages": 2, "work_dir": "groupchat"},
    human_input_mode="TERMINATE"
)
coder = autogen.AssistantAgent(
    name="Coder",
    llm_config=gemma,
)
pm = autogen.AssistantAgent(
    name="Product_manager",
    system_message="Creative in software product ideas, focused on structural engineering.",
    llm_config=gemma,
)
groupchat = autogen.GroupChat(agents=[user_proxy, coder, pm], messages=[], max_round=12)

# New tool integration for structural engineering
class PDFReaderAgent(autogen.Agent):
    def __init__(self, name, llm_config, file_path):
        super().__init__(name, llm_config)
        self.pdf_reader = PDFReaderTool(file_path)

    def handle_message(self, message):
        if "read pdf" in message.lower():
            return self.pdf_reader.read_pdf()
        return super().handle_message(message)

pdf_agent = PDFReaderAgent(
    name="PDF_reader",
    llm_config=gemma,
    file_path="pdf_test/12_what-produce-at-end-conceptual.pdf"
)

# Add PDFReaderAgent to the group chat
groupchat.agents.append(pdf_agent)

# New agent for problem analysis in structural engineering
class StructuralEngineerAgent(autogen.Agent):
    def __init__(self, name, llm_config):
        super().__init__(name, llm_config)

    def handle_message(self, message):
        if "analyze problem" in message.lower():
            return self.analyze_problem(message)
        return super().handle_message(message)

    def analyze_problem(self, message):
        # Implement the problem analysis logic for structural engineering here
        # This is a placeholder implementation
        return "Analyzing structural engineering problem: " + message

structural_engineer = StructuralEngineerAgent(
    name="Structural_engineer",
    llm_config=gemma,
)

# Add StructuralEngineerAgent to the group chat
groupchat.agents.append(structural_engineer)

manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=gemma)
