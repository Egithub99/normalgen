from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)

from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.lmstudio import LMStudio
import nest_asyncio
nest_asyncio.apply()
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# bge-base embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = LMStudio(
    model_name="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
    base_url="http://localhost:1234/v1",
    temperature=0,
    request_timeout=360,
)

try:
    storage_context = StorageContext.from_defaults(
        persist_dir="./data4/table_9_7.pdf"
    )
    lyft_index = load_index_from_storage(storage_context)

    index_loaded = True
except:
    index_loaded = False


if not index_loaded:
    # load data
    lyft_docs = SimpleDirectoryReader(
        input_files=["./data4/table_9_7.pdf"]
    ).load_data()

    # build index
    lyft_index = VectorStoreIndex.from_documents(lyft_docs)

    # persist index
    lyft_index.storage_context.persist(persist_dir="./storage2/lyft")

lyft_engine = lyft_index.as_query_engine(similarity_top_k=3)


query_engine_tools = [
    QueryEngineTool(
        query_engine=lyft_engine,
        metadata=ToolMetadata(
            name="lyft_10k",
            description=(
                "Provides information about construcion types."
            ),
        ),
    ),
]

from llama_index.core.agent import ReActAgent




agent = ReActAgent.from_tools(
    query_engine_tools,
    llm=Settings.llm,
    verbose=True,
    # context=context
)

response = agent.chat("Name the construction types.")
print(str(response))
