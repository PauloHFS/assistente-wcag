import os
from typing import List, TypedDict, Optional, Literal
from pydantic import BaseModel, Field

from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import END, StateGraph

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
    """

    question: str
    documents: List[Document]
    generation: str


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

    def __init__(self, llm: Optional[BaseChatModel] = None, retriever: Optional[BaseRetriever] = None, gemini_api_key: Optional[str] = None):
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
            embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
            chroma_db_path = "./chroma_langchain_db"
            if not os.path.exists(chroma_db_path):
                raise FileNotFoundError(f"Diretório do ChromaDB não encontrado em '{chroma_db_path}'.")

            vector_store = Chroma(
                collection_name="WCAG",
                embedding_function=embeddings,
                persist_directory=chroma_db_path,
            )
            self.retriever = vector_store.as_retriever()

        self.workflow = self._create_workflow()
        print("--- Agente inicializado ---")

    def _create_workflow(self):
        """
        Cria e compila o grafo StateGraph com o ciclo de verificação de relevância.
        A lógica condicional agora acontece imediatamente após a recuperação de documentos.
        """
        workflow = StateGraph(State)

        # Nós do Grafo
        workflow.add_node("retrieve", self.retrieve_documents)
        workflow.add_node("rewrite_question", self.rewrite_question)
        workflow.add_node("generate", self.generate_answer)
        workflow.add_node("safety", self.safety_node)

        # Estrutura do Grafo (Edges)
        workflow.set_entry_point("retrieve")
        
        # ### ALTERADO ###
        # Adiciona uma borda condicional a partir de "retrieve".
        # A função `grade_documents` agora decide se vai para "generate" ou "rewrite_question".
        workflow.add_conditional_edges(
            "retrieve",
            self.grade_documents, # A função de avaliação agora é o roteador
            {
                "generate": "generate",
                "rewrite_question": "rewrite_question",
            },
        )
        
        workflow.add_edge("rewrite_question", "retrieve")
        workflow.add_edge("generate", "safety")
        workflow.add_edge("safety", END)

        # A chamada .compile() é essencial para tornar o grafo executável.
        return workflow.compile()

    # --- Nós do Grafo (métodos da classe) ---

    def retrieve_documents(self, state: State) -> State:
        """Recupera documentos usando o retriever da instância."""
        print("--- RECUPERANDO DOCUMENTOS ---")
        question = state["question"]
        documents = self.retriever.invoke(question)
        print(f"--- {len(documents)} DOCUMENTOS RECUPERADOS ---")
        return {"documents": documents, "question": question}


    @staticmethod
    def format_docs_with_link(docs: List[Document]) -> str:
        """Formata os documentos recuperados para incluir links e títulos."""
        if not docs:
            return "Nenhum documento encontrado."
        formatted = [
            f"""Source Link: {doc.metadata.get("source", "N/A")}\nArticle Title: {doc.metadata.get("title", "N/A")}\n
            Article Snippet: {doc.page_content}"""
            for doc in docs
        ]
        return "\n\n" + "\n\n".join(formatted)

    # ### ALTERADO ###
    # Esta função agora avalia os documentos e retorna o nome do próximo nó.
    def grade_documents(self, state: State) -> Literal["generate", "rewrite_question"]:
        """
        Determina se os documentos retornados são relevantes à questão e retorna a rota a seguir.
        """
        print("--- VERIFICANDO RELEVÂNCIA DOS DOCUMENTOS ---")
        question = state["question"]
        documents = state["documents"]
        
        if not documents:
            print("--- DECISÃO: DOCUMENTOS NÃO RELEVANTES (VAZIO), REESCREVENDO A PERGUNTA ---")
            return "rewrite_question"

        formatted_docs = self.format_docs_with_link(documents)

        prompt_template = (
            "You are a grader assessing relevance of a retrieved document to a user question. \n "
            "Here is the retrieved document context: \n\n {context} \n\n"
            "Here is the user question: {question} \n"
            "If the document context contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
            "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
        )
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        grader_chain = prompt | self.grader_model.with_structured_output(GradeDocuments)
        
        response = grader_chain.invoke({"question": question, "context": formatted_docs})
        score = response.binary_score

        if score.lower() == "yes":
            print("--- DECISÃO: DOCUMENTOS RELEVANTES, INDO PARA A GERAÇÃO DA RESPOSTA ---")
            return "generate"
        else:
            print("--- DECISÃO: DOCUMENTOS NÃO RELEVANTES, REESCREVENDO A PERGUNTA ---")
            return "rewrite_question"

    # ### REMOVIDO ###
    # A função `decide_to_generate_or_rewrite` não é mais necessária,
    # pois sua lógica foi incorporada em `grade_documents`.

    def rewrite_question(self, state: State) -> State:
        """Reescreve a pergunta original do usuário para melhorar a busca."""
        print("--- REESCREVENDO PERGUNTA ---")
        question = state["question"]

        prompt_template = (
            "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
            "Here is the initial question:"
            "\n ------- \n"
            "{question}"
            "\n ------- \n"
            "Formulate only an improved question based on the original, making it more specific or clearer for a vector database search:"
        )
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        rewriter_chain = prompt | self.rewriter_model
        response = rewriter_chain.invoke({"question": question})
        new_question = response.content
        
        print(f"--- NOVA PERGUNTA: {new_question} ---")
        return {"question": new_question, "documents": []}

    def generate_answer(self, state: State) -> State:
        """Gera uma resposta usando o LLM da instância."""
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

        Contexto:
        {context}
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm
        response = chain.invoke({"question": question, "context": formatted_docs})
        generation = response.content
        print("--- RESPOSTA GERADA ---")
        return {"generation": generation}

    def safety_node(self, state: State) -> State:
        """Adiciona o texto de aviso à resposta gerada."""
        print("--- ADICIONANDO AVISO DE SEGURANÇA ---")
        current_generation = state.get("generation", "")
        updated_generation = current_generation + self.DISCLAIMER_TEXT
        return {"generation": updated_generation}

    def invoke(self, question: str) -> dict:
        """Ponto de entrada público para executar o workflow do agente."""
        initial_state = {"question": question, "documents": [], "generation": ""}
        return self.workflow.invoke(initial_state)