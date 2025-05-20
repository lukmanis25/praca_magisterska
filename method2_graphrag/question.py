import json
from graphrag.rag import GraphRAG
from graphrag.query import load_graph_rag

# Ścieżka do katalogu z indeksem (analogicznie do --root .)
graphrag_root = "."  # lub np. "my_indexed_docs"

# Lista pytań
questions = [
    "How students can graduate from school?"
]

# Załaduj GraphRAG z plików
graph_rag: GraphRAG = load_graph_rag(
    root=graphrag_root,
    method="drift",       
    load_cached=True,      
)

results = []
for question in questions:
    print("\n############### QUESTION ##############\n")
    print(question)
    response = graph_rag.answer(question)
    results.append({
        "question": question,
        "answer": response
    })
    print("\n############### ANSWERE ##############\n")
    print(response)
