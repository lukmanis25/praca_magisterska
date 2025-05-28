import os
import logging
import json
import asyncio
import logging.config
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.llm.hf import hf_model_complete, hf_embed
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from transformers import AutoModel, AutoTokenizer
from lightrag.kg.shared_storage import initialize_pipeline_status
import textract
import datetime
import asyncio
import nest_asyncio

nest_asyncio.apply()

def configure_logging():
    """Configure logging for the application"""

    # Reset any existing handlers to ensure clean configuration
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag"]:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.handlers = []
        logger_instance.filters = []

    # Get log directory path from environment variable or use current directory
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(
        os.path.join(log_dir, "lightrag_compatible_demo.log")
    )

    print(f"\nLightRAG compatible demo log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    # Get log file max size and backup count from environment variables
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["file"],
                    "level": "DEBUG",
                    "propagate": False,
                },
            },
        }
    )
    logger.setLevel(logging.DEBUG)
    set_verbose_debug(os.getenv("VERBOSE_DEBUG", "false").lower() == "true")

WORKING_DIR = os.getenv("WORKING_DIR")

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        os.getenv("LLM_MODEL_NAME"),
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
        **kwargs,
    )



async def initialize_rag():
    
    fun = llm_model_func if os.getenv("METHOD") == 'API' else hf_model_complete
    print("MODAL NAME AAA:")
    print(os.getenv("LLM_MODEL_NAME"))
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=fun,
        llm_model_name=os.getenv("LLM_MODEL_NAME"),
        graph_storage="Neo4JStorage",
        embedding_func=EmbeddingFunc(
            embedding_dim=384,
            max_token_size=5000,
            func=lambda texts: openai_embed(
                texts,
                model= os.getenv("EMBED_MODEL"),
                base_url= os.getenv("EMBED_URL"),
                api_key= os.getenv("EMBED_TOKEN")
            ),
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag

def save_answer_to_file(question, answer):
    os.makedirs("ans", exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ans/answer_{timestamp}_{os.path.basename(os.getenv('WORKING_DIR'))}.txt"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Question:\n{question}\n\nAnswer:\n{answer}\n")


def clear_hybrid_field(path):
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    def clear_hybrid(obj):
        if isinstance(obj, dict):
            if "hybrid" in obj:
                obj["hybrid"] = None
            for value in obj.values():
                clear_hybrid(value)
        elif isinstance(obj, list):
            for item in obj:
                clear_hybrid(item)

    clear_hybrid(data)

    with open(path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

    print(f"Cache 'hybrid' cleared in file: {path}")
    
def clear_local_field(path):
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    def clear_local(obj):
        if isinstance(obj, dict):
            if "local" in obj:
                obj["local"] = None
            for value in obj.values():
                clear_local(value)
        elif isinstance(obj, list):
            for item in obj:
                clear_local(item)

    clear_local(data)

    with open(path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

    print(f"Cache 'local' cleared in file: {path}")
        
async def init_rag():
    clear_hybrid_field(f"{WORKING_DIR}/kv_store_llm_response_cache.json")
    clear_local_field(f"{WORKING_DIR}/kv_store_llm_response_cache.json")
    configure_logging()
    return await initialize_rag()

async def query(rag, question):
    result=None
    try:
        result = rag.query(question, param=QueryParam(mode=os.getenv("ANS_MODE"), top_k=int(os.getenv("TOP_K"))))
        save_answer_to_file(question, str(result))
    except Exception as e:
        logging.error(f"An error occurred while processing your query")
    
    return result
