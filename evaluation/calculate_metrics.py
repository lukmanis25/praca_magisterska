import os
import pandas as pd
from ragas.metrics import answer_relevancy, faithfulness, answer_correctness
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from datasets import Dataset
from langchain_localai import LocalAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

data = pd.DataFrame({
    "question": [
        "How is the weighted average grade calculated, and which courses are excluded from this calculation?"
    ],
    "answer": [
        """
        The weighted average grade is calculated using the following formula:

        weighted average grade = ∑ o p / ∑ p i

        where:

        * o i is the grade for a given subject
        * p i is the number of ECTS credits assigned to a given subject

        The courses that are excluded from this calculation are those that were not assigned ECTS credits or were passed without a grade.

        According to the rules, the weighted average of the grades for the semester is calculated taking into account the grades in all subjects. Missing grades in subjects result in no weighted average for a given semester.
        """
    ],
    "ground_truth": [
        "The weighted average grade is calculated by dividing the sum of each subject's grade multiplied by its ECTS credits by the total ECTS credits. Courses without assigned ECTS credits or passed without a grade are excluded from this calculation."
    ]
    # ,"contexts": [  
    #     ["Paris is the capital and most populous city of France."]
    # ]
})

dataset = Dataset.from_pandas(data)


llm = ChatOpenAI(
    openai_api_base=os.getenv("OPENAI_API_BASE"), 
    openai_api_key=os.getenv("OPENAI_API_KEY"),              
    model_name=os.getenv("LLM_MODEL_NAME")          
)
llm = LangchainLLMWrapper(llm)


embedding = LocalAIEmbeddings(
    openai_api_base=os.getenv("EMBED_URL"), 
    openai_api_key=os.getenv("EMBED_TOKEN"), 
    model=os.getenv("EMBED_MODEL")
)
embedding = LangchainEmbeddingsWrapper(embedding)


results = evaluate(
    dataset=dataset,
    metrics=[answer_correctness, answer_relevancy],
    llm=llm,
    embeddings=embedding
)

print("📊 Wyniki ewaluacji:")
print(results)
