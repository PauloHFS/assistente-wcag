import typing

import pandas as pd
from datasets import Dataset
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas import evaluate
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_recall,
    faithfulness,
)


# TODO: remover isso quando tiver o agent
def mock_rag_pipeline(question: str, ground_truth: str) -> typing.Dict[str, typing.Any]:
    generated_answer = f"De acordo com as diretrizes, {ground_truth.lower().strip('.')}, o que é crucial para a conformidade."

    retrieved_contexts = [
        f"A documentação principal afirma claramente que: '{ground_truth}'.",
        f"Um artigo secundário discute o tópico da pergunta '{question}' e menciona pontos relacionados.",
        "Outra fonte fala sobre a história e a evolução dessas diretrizes de acessibilidade.",
    ]
    return {"answer": generated_answer, "contexts": retrieved_contexts}


def main():
    df = pd.read_csv("eval/test-set.csv")

    results = []
    for _, row in df.iterrows():
        # TODO: remover isso quando tiver o agent
        result = mock_rag_pipeline(row["question"], row["ground_truth"])
        results.append(result)

    df["answer"] = [r["answer"] for r in results]
    df["contexts"] = [r["contexts"] for r in results]
    dataset = Dataset.from_pandas(df)

    gemini_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
    gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    evaluation_result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            answer_correctness,
            context_recall,
        ],
        llm=gemini_llm,
        embeddings=gemini_embeddings,
    )

    print("\n--- Relatório de Avaliação Quantitativa (RAGAS) ---")
    print("--- LLM: Google Gemini | Embeddings: Google ---")
    print(evaluation_result)
    print("-----------------------------------------------------")


if __name__ == "__main__":
    main()
