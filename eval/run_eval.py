import typing

import pandas as pd
from datasets import Dataset
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness


def mock_rag_pipeline(question: str) -> typing.Dict[str, typing.Any]:
    generated_answer = f"Com base nos dados, a resposta para '{question}' parece ser a que foi encontrada nas fontes."
    retrieved_contexts = [
        f"Fonte A menciona informações gerais sobre a pergunta: '{question}'.",
        "Fonte B confirma detalhes específicos relacionados ao tópico.",
        "Fonte C oferece um contexto histórico sobre o assunto.",
    ]
    return {"answer": generated_answer, "contexts": retrieved_contexts}


def main():
    df = pd.read_csv("eval/test-set.csv")
    results = [mock_rag_pipeline(q) for q in df["question"]]
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
