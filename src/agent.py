import os
from typing import List, Literal, Optional, TypedDict

from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field


# Definição da estrutura para o Grader
class GradeDocuments(BaseModel):
    """Avalia documentos usando uma pontuação binária para uma checagem de relevância."""

    binary_score: str = Field(
        description="Score de relevância: 'yes' se relevante, ou 'no' se não relevante"
    )


class State(TypedDict):
    """
    Representa o estado do nosso grafo de RAG.

    Atributos:
        question (str): A pergunta feita pelo usuário.
        documents (List[Document]): Documentos recuperados que são relevantes para a pergunta.
        generation (str): A resposta gerada pelo LLM com base nos documentos.
        sources (List[str]): Lista de links de origem dos documentos usados na resposta.
    """

    question: str
    documents: List[Document]
    generation: str
    sources: List[str]


class RAGAgent:
    """
    Encapsula toda a lógica, componentes e o workflow de um agente RAG
    com verificação de relevância e reescrita de perguntas.
    """

    DISCLAIMER_TEXT = (
        "\n\n---"
        "\n**Aviso**: Esta ferramenta é uma Prova de Conceito (PoC) e suas "
        "respostas podem conter imprecisões ou ser incompletas. Verifique "
        "as informações antes de utilizá-las."
    )

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        retriever: Optional[BaseRetriever] = None,
    ):
        """
        Inicializa o agente, carregando seus componentes e compilando o workflow.
        """
        print("--- Inicializando o RAGAgent... ---")

        self.llm = llm
        self.grader_model = self.llm
        self.rewriter_model = self.llm

        if retriever:
            self.retriever = retriever
        else:
            # Lógica para carregar o ChromaDB...
            # (Omitida por brevidade, igual à sua versão original)
            pass

        self.workflow = self._create_workflow()
        print("--- Agente inicializado ---")

    # --- ADICIONADO: Método para extrair as fontes ---
    @staticmethod
    def format_docs(docs: List[Document]) -> List[str]:
        """
        Extrai os links de origem únicos dos metadados dos documentos.
        """
        sources = [doc.metadata["source"] for doc in docs if "source" in doc.metadata]
        # Remove fontes duplicadas mantendo a ordem de aparição
        return list(dict.fromkeys(sources))

    def _create_workflow(self):
        """
        Cria e compila o grafo StateGraph com o ciclo de verificação de relevância.
        """
        workflow = StateGraph(State)

        workflow.add_node("retrieve", self.retrieve_documents)
        workflow.add_node("rewrite_question", self.rewrite_question)
        workflow.add_node("generate", self.generate_answer)
        workflow.add_node("safety", self.safety_node)

        workflow.set_entry_point("retrieve")

        workflow.add_conditional_edges(
            "retrieve",
            self.grade_documents,
            {
                "generate": "generate",
                "rewrite_question": "rewrite_question",
            },
        )

        workflow.add_edge("rewrite_question", "retrieve")
        workflow.add_edge("generate", "safety")
        workflow.add_edge("safety", END)

        return workflow.compile()

    def retrieve_documents(self, state: State) -> State:
        """Recupera documentos usando o retriever da instância."""
        print("--- RECUPERANDO DOCUMENTOS ---")
        question = state["question"]
        documents = self.retriever.invoke(question)
        print(f"--- {len(documents)} DOCUMENTOS RECUPERADOS ---")
        return {"documents": documents, "question": question}

    @staticmethod
    def format_docs_with_link(docs: List[Document]) -> str:
        """Formata os documentos recuperados para incluir links e títulos para o contexto do LLM."""
        if not docs:
            return "Nenhum documento encontrado."
        formatted = [
            f"""Source Link: {doc.metadata.get("source", "N/A")}\nArticle Title: {doc.metadata.get("title", "N/A")}\n
            Article Snippet: {doc.page_content}"""
            for doc in docs
        ]
        return "\n\n" + "\n\n".join(formatted)

    def grade_documents(self, state: State) -> Literal["generate", "rewrite_question"]:
        """
        Determina se os documentos retornados são relevantes à questão.
        """
        print("--- VERIFICANDO RELEVÂNCIA DOS DOCUMENTOS ---")
        question = state["question"]
        documents = state["documents"]

        if not documents:
            print(
                "--- DECISÃO: DOCUMENTOS NÃO RELEVANTES (VAZIO), REESCREVENDO A PERGUNTA ---"
            )
            return "rewrite_question"

        # ... (Lógica do grader_chain, igual à sua versão original) ...
        # (Omitida por brevidade)
        score = "yes"  # Simulação para o exemplo

        if score.lower() == "yes":
            print(
                "--- DECISÃO: DOCUMENTOS RELEVANTES, INDO PARA A GERAÇÃO DA RESPOSTA ---"
            )
            return "generate"
        else:
            print("--- DECISÃO: DOCUMENTOS NÃO RELEVANTES, REESCREVENDO A PERGUNTA ---")
            return "rewrite_question"

    def rewrite_question(self, state: State) -> State:
        """Reescreve a pergunta original do usuário para melhorar a busca."""
        print("--- REESCREVENDO PERGUNTA ---")
        question = state["question"]
        # ... (Lógica do rewriter_chain, igual à sua versão original) ...
        # (Omitida por brevidade)
        new_question = f"improved: {question}"  # Simulação
        print(f"--- NOVA PERGUNTA: {new_question} ---")
        return {"question": new_question, "documents": []}

    def generate_answer(self, state: State) -> State:
        """Gera uma resposta e extrai as fontes dos documentos."""
        print("--- GERANDO RESPOSTA ---")
        question = state["question"]
        documents = state["documents"]
        formatted_docs = self.format_docs_with_link(documents)

        prompt_template = """
        Você é um assistente especializado em tarefas de perguntas e respostas.
        Use os seguintes trechos de contexto recuperado para responder à pergunta.
        Se os trechos não apresentam uma resposta satisfatória apenas diga que não sabe.
        Mantenha a resposta concisa, a não ser que o usuário peça por detalhes.

        Pergunta: {question}
        Contexto: {context}
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm
        response = chain.invoke({"question": question, "context": formatted_docs})
        generation = response.content
        print("--- RESPOSTA GERADA ---")

        # Chama o novo método para obter a lista de fontes
        sources = self.format_docs(documents)
        print("--- FONTES EXTRAÍDAS ---")

        # Retorna tanto a geração quanto as fontes para o estado
        return {"generation": generation, "sources": sources}

    def safety_node(self, state: State) -> State:
        """Adiciona o texto de aviso à resposta gerada."""
        print("--- ADICIONANDO AVISO DE SEGURANÇA ---")
        current_generation = state.get("generation", "")
        updated_generation = current_generation + self.DISCLAIMER_TEXT
        # Passa as fontes adiante sem modificá-las
        return {"generation": updated_generation, "sources": state["sources"]}

    def invoke(self, question: str) -> dict:
        """Ponto de entrada público para executar o workflow do agente."""
        # O estado inicial agora inclui o campo 'sources'
        initial_state = {
            "question": question,
            "documents": [],
            "generation": "",
            "sources": [],
        }

        # O resultado do workflow é o estado final completo
        final_state = self.workflow.invoke(initial_state)

        # Retorna um dicionário limpo contendo apenas a geração e as fontes
        return {
            "generation": final_state.get("generation"),
            "sources": final_state.get("sources"),
        }
