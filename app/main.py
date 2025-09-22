# Injeta a implementação do SQLite do pysqlite3-binary no sys.modules
# Isso é necessário para que o ChromaDB funcione em ambientes com versões mais antigas do SQLite
__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import os

# Adiciona o diretório raiz do projeto ao sys.path
# Isso permite que o Python encontre o módulo 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import re
import time
import streamlit as st
import threading
from queue import Queue
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.agent import RAGAgent
from src.utils import OLLAMA_HOST


def display_llm_response(response: str):
    """
    Processa e exibe a resposta de um LLM no Streamlit.

    Esta função procura por tags <think> na resposta. O conteúdo dessas tags
    é exibido dentro de um componente st.expander, permitindo que o usuário
    veja o "processo de pensamento" do modelo. O restante da resposta é
    exibido normalmente.

    Args:
        response (str): A string de resposta completa do LLM.
    """
    think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    
    # Encontra todos os blocos de pensamento
    thoughts = think_pattern.findall(response)
    
    # Remove os blocos de pensamento da resposta principal
    clean_response = think_pattern.sub("", response).strip()

    # Se houver pensamentos, exibe-os em um expander
    if thoughts:
        with st.expander("Ver processo de pensamento do assistente"):
            for thought in thoughts:
                st.text(thought.strip())

    # Exibe a resposta principal e limpa
    st.markdown(clean_response)


# Função para carregar o agente RAG com cache do Streamlit
@st.cache_resource
def load_rag_agent():
    """
    Inicializa e retorna o RAGAgent, carregando seus componentes.
    O decorator @st.cache_resource garante que o agente seja carregado apenas uma vez.
    """
    # Inicializa o modelo de linguagem (LLM) que será usado pelo agente
    llm = ChatOllama(base_url=OLLAMA_HOST, model="qwen3:1.7b", temperature=0)

    # Define o modelo de embeddings para vetorizar o texto
    embeddings_model_name = "thenlper/gte-small"
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    # Conecta-se ao banco de dados vetorial ChromaDB que armazena os documentos
    vector_store = Chroma(
        collection_name="WCAG",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",
    )

    # Cria o retriever, que é responsável por buscar os documentos relevantes
    retriever = vector_store.as_retriever()

    # Instancia o agente RAG com o LLM e o retriever
    agent = RAGAgent(llm=llm, retriever=retriever)
    return agent


# Carrega o agente (a função com cache garante a eficiência)
agent = load_rag_agent()

st.set_page_config(layout="wide")
st.title("Assistente de Pesquisa com IA")
st.markdown(
    "Faça uma pergunta para obter uma resposta consolidada a partir de múltiplas fontes, juntamente com as referências utilizadas."
)

# Formulário para o usuário inserir a pergunta
with st.form(key="query_form"):
    user_question = st.text_input(
        "Qual é a sua pergunta?",
        key="user_question",
        placeholder="Ex: Quais são os critérios de sucesso para legendas em vídeos?",
    )
    submit_button = st.form_submit_button(label="Perguntar")

# Processamento quando o formulário é enviado
if submit_button and user_question:
    result_queue = Queue()
    timer_placeholder = st.empty()
    start_time = time.time()

    # Define a função que executa o agente em uma thread separada
    def run_agent_in_thread(question):
        result = agent.invoke(question)
        result_queue.put(result)

    # Cria e inicia a thread
    agent_thread = threading.Thread(
        target=run_agent_in_thread, args=(user_question,)
    )
    agent_thread.start()

    # Exibe o spinner e atualiza o cronômetro enquanto a thread está ativa
    with st.spinner("Analisando fontes e compilando a resposta..."):
        while agent_thread.is_alive():
            elapsed_time = time.time() - start_time
            timer_placeholder.caption(f"⏳ Tempo decorrido: {elapsed_time:.1f} segundos")
            time.sleep(0.1)
        agent_thread.join()  # Garante que a thread terminou

    # Atualiza o cronômetro com o tempo final e o mantém na tela
    total_time = time.time() - start_time
    timer_placeholder.caption(f"✅ Resposta gerada em {total_time:.1f} segundos")
    result = result_queue.get()  # Obtém o resultado da fila

    st.divider()

    # Exibe a resposta gerada usando a nova função
    st.subheader("Resposta")
    display_llm_response(result["generation"])

    # Exibe as fontes utilizadas
    st.subheader("Fontes")
    if result["sources"]:
        for source_url in result["sources"]:
            st.markdown(f"- {source_url}")
    else:
        st.markdown("Nenhuma fonte foi utilizada para gerar esta resposta.")
