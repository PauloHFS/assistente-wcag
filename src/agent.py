import os
from typing import List, TypedDict, Optional
from pydantic import BaseModel, Field

from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import END, StateGraph

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
    Encapsula toda a lógica, componentes e o workflow de um agente RAG.
    """
    DISCLAIMER_TEXT = (
        "\n\n---"
        "\n**Aviso**: Esta ferramenta é uma Prova de Conceito (PoC) e suas "
        "respostas podem conter imprecisões ou ser incompletas. Verifique "
        "as informações antes de utilizá-las."
    )

    def __init__(self, llm: Optional[BaseChatModel] = None, retriever: Optional[BaseRetriever] = None):
        """
        Inicializa o agente, carregando seus componentes e compilando o workflow.
        """
        print("--- Inicializando o RAGAgent... ---")
        
        self.llm = llm or ChatOllama(model="llama3.1:8b", temperature=0)

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
        Cria e compila o grafo StateGraph.
        """
        workflow = StateGraph(State)

        workflow.add_node("retrieve", self.retrieve_documents)
        workflow.add_node("generate", self.generate_answer)
        workflow.add_node("safety", self.safety_node)

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "safety")
        workflow.add_edge("safety", END)

        return workflow.compile()

    # --- Nós do Grafo (métodos da classe) ---
    def retrieve_documents(self, state: State) -> State:
        """Recupera documentos usando o retriever da instância."""
        print("--- RECUPERANDO DOCUMENTOS ---")
        question = state["question"]
        documents = self.retriever.invoke(question)
        print(f"--- {len(documents)} DOCUMENTOS RECUPERADOS ---")
        # O retorno é um dicionário para ATUALIZAR o estado, não precisa conter todos os campos.
        return {"documents": documents, "question": question}
    
    @staticmethod
    def format_docs_with_link(docs: List[Document]) -> str:
        """Formata os documentos recuperados para incluir links e títulos."""
        formatted = [
            f"""Source Link: {doc.metadata.get('source', 'N/A')}\nArticle Title: {doc.metadata.get('title', 'N/A')}\n
            Article Snippet: {doc.page_content}"""
            for doc in docs
        ]
        return "\n\n" + "\n\n".join(formatted)

    def generate_answer(self, state: State) -> State:
        """Gera uma resposta usando o LLM da instância."""
        print("--- GERANDO RESPOSTA ---")
        question = state["question"]
        documents = state["documents"]
        # CORREÇÃO: Chamando o método estático corretamente.
        formatted_docs = RAGAgent.format_docs_with_link(documents)

        prompt_template = """
        Você é um assistente especializado em tarefas de perguntas e respostas.
        Use os seguintes trechos de contexto recuperado para responder à pergunta.
        Se os trechos não apresentam uma resposta satisfatória apenas diga que não sabe.
        Mantenha a resposta concisa, a não ser que o usuário peça por detalhes.
        No final da resposta, inclua uma linha citando o link de onde a informação foi retirada.

        Pergunta: {question}

        Contexto:
        {context}
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm
        response = chain.invoke({"question": question, "context": formatted_docs})
        generation = response.content
        print("--- RESPOSTA GERADA ---")
        # Atualiza o estado com a geração.
        return {"generation": generation}

    def safety_node(self, state: State) -> State:
        """
        Adiciona o texto de aviso à resposta gerada.
        """
        print("--- ADICIONANDO AVISO DE SEGURANÇA ---")
        current_generation = state.get("generation", "")
        updated_generation = current_generation + self.DISCLAIMER_TEXT
        
        # Atualiza o estado com a geração final.
        return {"generation": updated_generation}

    def invoke(self, question: str) -> dict:
        """
        Ponto de entrada público para executar o workflow do agente.
        """
        initial_state = {"question": question, "documents": [], "generation": ""}
        return self.workflow.invoke(initial_state)
