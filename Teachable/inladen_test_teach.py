# bring in our LLAMA_CLOUD_API_KEY
from dotenv import load_dotenv
load_dotenv()
# bring in deps
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.lmstudio import LMStudio
import nest_asyncio
nest_asyncio.apply()
import os.path

# bge-base embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

Settings.llm = LMStudio(
    model_name="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
    base_url="http://localhost:1234/v1",
    temperature=0,
    request_timeout=360,
)

# check if storage already exists
PERSIST_DIR = "./storage_testing5"
if not os.path.exists(PERSIST_DIR):
    # load the documents and create the index
    documents3 = LlamaParse().load_data("./data5/8_developing-a-concept.pdf")
    index3 = VectorStoreIndex.from_documents(documents3)
    # store it for later
    index3.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index3 = load_index_from_storage(storage_context)

query_engine3 = index3.as_query_engine()


# response4 = query_engine3.query(
#     "What are six key steps in the conceptual design phase?"
# )
# print(response4)