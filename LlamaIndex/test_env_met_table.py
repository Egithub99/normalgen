# bring in our LLAMA_CLOUD_API_KEY
from dotenv import load_dotenv
load_dotenv()

# bring in deps
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
import nest_asyncio
nest_asyncio.apply()
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.lmstudio import LMStudio


documents = SimpleDirectoryReader("data2").load_data()

# bge-base embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

Settings.llm = LMStudio(
    model_name="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
    base_url="http://localhost:1234/v1",
    temperature=0,
    request_timeout=360,
)


# documents2 = LlamaParse(result_type="markdown").load_data(
#     "./data3/9_stability-robustness.pdf"
# )

documents3 = LlamaParse().load_data(
    "./data4/table_9_7.pdf"
)
index3 = VectorStoreIndex.from_documents(documents3)
query_engine3 = index3.as_query_engine()

response3 = query_engine3.query(
    "Can you name the Construction types from the table?"
)
print(response3)