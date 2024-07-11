
from llama_index.core.base.llms.types import ChatMessage, MessageRole
import nest_asyncio
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.lmstudio import LMStudio

nest_asyncio.apply()

llm = LMStudio(
    model_name="TheBloke/Mistral-7B-Instruct-v0.1-GGUF/mistral-7b-instruct-v0.1.Q5_0.gguf",
    base_url="http://localhost:1234/v1",
    temperature=0.1,
)

Settings.llm = llm
Settings.embed_model='local'


documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("Give me a brief summary.")
print(response)