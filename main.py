# main.py

import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger, EmbeddingFunc
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.llm.ollama import ollama_embed

# Set up logging
setup_logger("lightrag", level="INFO")

WORKING_DIR = "./rag_storage"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# LLM function using OpenRouter (DeepSeek model)
async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
):
    # Get API key from environment variable
    api_key = os.getenv("LLM_BINDING_API_KEY")
    if not api_key:
        raise ValueError("LLM_BINDING_API_KEY environment variable is not set")
        
    return await openai_complete_if_cache(
        "deepseek/deepseek-r1-0528:free",  # or "deepseek-chat" if you prefer
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        **kwargs
    )

# Embedding function using Ollama (all-minilm:latest)
embedding_func = EmbeddingFunc(
    embedding_dim=384,  # all-minilm:latest is 384-dim
    max_token_size=8192,
    func=lambda texts: ollama_embed(
        texts,
        embed_model="all-minilm:latest"
    )
)

async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag

async def main():
    rag = await initialize_rag()
    rag.insert("Your text here")
    result = await rag.query(
        "What are the top themes in this story?",
        param=QueryParam(mode="hybrid")
    )
    print(result)
    await rag.finalize_storages()

if __name__ == "__main__":
    asyncio.run(main())
