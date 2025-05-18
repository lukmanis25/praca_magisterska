import os
import logging
import asyncio
import logging.config
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
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
                    "handlers": ["console", "file"],
                    "level": "DEBUG",
                    "propagate": False,
                },
            },
        }
    )

    # Set the logger level to INFO
    logger.setLevel(logging.DEBUG)
    # Enable verbose debug if needed
    set_verbose_debug(os.getenv("VERBOSE_DEBUG", "false").lower() == "true")

WORKING_DIR = os.getenv("WORKING_DIR")

# if not os.path.exists(WORKING_DIR):
#     os.mkdir(WORKING_DIR)


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
            func=lambda texts: hf_embed(
                texts,
                tokenizer=AutoTokenizer.from_pretrained(
                    "sentence-transformers/all-MiniLM-L6-v2"
                ),
                embed_model=AutoModel.from_pretrained(
                    "sentence-transformers/all-MiniLM-L6-v2"
                ),
            ),
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag

def save_answer_to_file(question, answer):
    # Ensure the 'ans' directory exists
    os.makedirs("ans", exist_ok=True)

    # Create a unique filename based on timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ans/answer_{timestamp}_{os.path.basename(os.getenv('WORKING_DIR'))}.txt"

    # Write the question and answer to the file
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Question:\n{question}\n\nAnswer:\n{answer}\n")

async def main():
    rag =  await initialize_rag()
    print("Welcome! Type your question below. Type 'exit', 'quit' or 'q' to end the session.")

    while True:
        question = input("Question: ")

        if question.lower() in {"exit", "quit", "q"}:
            print("Session ended.")
            break

        # result = rag.query(question, param=QueryParam(mode=os.getenv("ANS_MODE"), top_k=int(os.getenv("TOP_K"))))
        # print("Answer:", result)
        # save_answer_to_file(question, str(result))
        try:
            result = rag.query(question, param=QueryParam(mode=os.getenv("ANS_MODE"), top_k=int(os.getenv("TOP_K"))))
            print("Answer:", result)
            save_answer_to_file(question, str(result))
        except Exception as e:
            print("An error occurred while processing your query:", e)


if __name__ == "__main__":
    configure_logging()
    asyncio.run(main())
    print("\nDone!")
