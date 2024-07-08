import autogen
import memgpt
from memgpt.autogen.memgpt_agent import create_memgpt_autogen_agent_from_config




# # Non-MemGPT agents will still use local LLMs, but they will use the ChatCompletions endpoint
llm_config_memgpt = [
    {
        "model": "lmstudio-ai/gemma-2b-it-GGUF/gemma-2b-it-q8_0.gguf",
        "base_url": "http://localhost:1234/v1",
        "api_key": "lm-studio",
    },
]

# MemGPT-powered agents will also use local LLMs, but they need additional setup (also they use the Completions endpoint)
llm_config_memgpt = [
    {
        "preset": "memgpt_chat",
        "model": None,
        "model_wrapper": "airoboros-l2-70b-2.1",
        "model_endpoint_type": "lmstudio",
        "model_endpoint": "http://localhost:1234",  # port 1234 for LM Studio
        "context_window": 8192,
    },
]



