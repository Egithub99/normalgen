
from llama_index.llms.lmstudio import LMStudio
from llama_index.core.base.llms.types import ChatMessage, MessageRole

import nest_asyncio

nest_asyncio.apply()


llm = LMStudio(
    model_name="TheBloke/Mistral-7B-Instruct-v0.1-GGUF/mistral-7b-instruct-v0.1.Q5_0.gguf",
    base_url="http://localhost:1234/v1",
    temperature=0.7,
)

# llm_config = {
#     "config_list": [
#         {
#             "model": "TheBloke/Mistral-7B-Instruct-v0.1-GGUF/mistral-7b-instruct-v0.1.Q5_0.gguf",
#             # "model": "mistral-7b-instruct-v0.1.Q5_0",
#             # "model": "mistral",
#             "base_url": "http://localhost:1234/v1",
#             "api_key": "lm-studio",
#         },
#     ],
#     "cache_seed": None,  # Disable caching.
# }

response = llm.complete("Hey there, what is 2+2?")
print(str(response))
