import os
from typing import List, TypedDict, Optional

from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import END, StateGraph, CompiledGraph



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
    # Adicionamos o texto do aviso como uma constante da classe
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
        print("--- 🛠️  Inicializando o RAGAgent... ---")
        
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
        print("--- ✅ Agente pronto para uso! ---")

    def _create_workflow(self) -> CompiledGraph:
        """
        Cria e compila o grafo StateGraph, agora incluindo o nó de segurança.
        """
        workflow = StateGraph(GraphState)

        workflow.add_node("retrieve", self.retrieve_documents)
        workflow.add_node("generate", self.generate_answer)
        # NOVO: Adiciona o nó de segurança ao grafo
        workflow.add_node("safety", self.safety_node)

        # Define o novo fluxo de execução: retrieve -> generate -> safety -> END
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        # ATUALIZADO: A saída de 'generate' agora vai para 'safety'
        workflow.add_edge("generate", "safety")
        # NOVO: A saída de 'safety' finaliza o fluxo
        workflow.add_edge("safety", END)

        return workflow.compile()

    # --- Nós do Grafo (métodos da classe) ---

    def retrieve_documents(self, state: GraphState) -> GraphState:
        """Recupera documentos usando o retriever da instância."""
        print("--- 📄 RECUPERANDO DOCUMENTOS ---")
        question = state["question"]
        documents = self.retriever.invoke(question)
        print(f"--- ✅ {len(documents)} DOCUMENTOS RECUPERADOS ---")
        return {"documents": documents, "question": question, "generation": ""}

    def generate_answer(self, state: GraphState) -> GraphState:
        """Gera uma resposta usando o LLM da instância."""
        print("--- 🤖 GERANDO RESPOSTA ---")
        question = state["question"]
        documents = state["documents"]

        prompt_template = """
        Você é um assistente especializado em tarefas de perguntas e respostas.
        Use os seguintes trechos de contexto recuperado para responder à pergunta.
        Se você não sabe a resposta, apenas diga que não sabe.
        Use no máximo três frases e mantenha a resposta concisa.

        Pergunta: {question}

        Contexto:
        {context}
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm
        response = chain.invoke({"question": question, "context": documents})
        generation = response.content
        print("--- ✅ RESPOSTA GERADA ---")
        return {"documents": documents, "question": question, "generation": generation}

    def safety_node(self, state: GraphState) -> GraphState:
        """
        Adiciona o texto de aviso à resposta gerada.
        """
        print("--- 🛡️  ADICIONANDO AVISO DE SEGURANÇA ---")
        current_generation = state.get("generation", "")
        # Sua função original adiciona o aviso no final, o que é ideal para um disclaimer.
        # Mantive essa lógica. Se quisesse no começo, seria: self.DISCLAIMER_TEXT + current_generation
        updated_generation = current_generation + self.DISCLAIMER_TEXT
        
        return {
            "generation": updated_generation,
            "question": state["question"],
            "documents": state["documents"]
        }

    def invoke(self, question: str) -> dict:
        """
        Ponto de entrada público para executar o workflow do agente.
        """
        initial_state = {"question": question}
        return self.workflow.invoke(initial_state)

