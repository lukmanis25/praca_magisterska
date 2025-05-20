from langchain_localai import LocalAIEmbeddings

embeddings = LocalAIEmbeddings(
    openai_api_base="http://localhost:54321/v1", openai_api_key="test-123", model="sentence-transformers/all-MiniLM-L6-v2"
)


vecs = embeddings.embed_documents(["LangChain jest super!", "LocalAI działa lokalnie."])
print(vecs)