import re
from uuid import uuid4

import langchain
from langchain_chroma import Chroma
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_parse(link: str):
    """
    Carrega o conteúdo de uma URL e as sub-páginas, extrai o texto principal
    utilizando um transformador BeautifulSoup, e retorna uma lista de objetos
    LangChain Document.

    Args:
        link: A URL inicial da página web a ser carregada.

    Returns:
        Uma lista de objetos Document, cada um contendo o texto e
        metadados da página.
    """
    # 1. Carrega recursivamente a URL e sub-páginas
    loader = RecursiveUrlLoader(url=link, prevent_outside=False)
    pages = loader.load()

    # 2. Transforma os documentos extraídos para filtrar tags e conteúdo
    bs4_transformer = BeautifulSoupTransformer()
    tags = ["main", "article", "p", "h1", "h2", "h3", "h4", "h5", "h6"]
    unwanted = ["nav", "footer", "aside"]

    docs = bs4_transformer.transform_documents(
        pages, tags_to_extract=tags, unwanted_classnames=unwanted
    )
    return docs


def chunk_docs(docs: list[Document]) -> list[Document]:
    # Instancia objeto responsável por separar os documentos originais em documentos menores
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )

    texts = text_splitter.split_documents(docs)
    return texts


def chunk_docs_list(docs, chunk_size):
    """
    Divide uma lista de documentos em pedaços (chunks) de tamanho especificado.
    Args:
        docs: A lista de documentos a ser dividida.
        chunk_size: O tamanho máximo de cada pc(docs: list[Dedaço.

    Yields:
        Sub-listas de documentos, cada uma com até chunk_size elementos.
    """
    for i in range(0, len(docs), chunk_size):
        yield docs[i : i + chunk_size]


def create_vector_store(docs: list[Document]) -> Chroma:
    """
    Vetoriza uma lista de documentos e os armazena em uma vector store Chroma.

    Args:
        docs: Uma lista de objetos Document do LangChain.

    Returns:
        A vector store Chroma preenchida com os documentos vetorizados.
    """
    # Instancia o modelo de embeddings
    model = "thenlper/gte-small"
    embeddings_model = HuggingFaceEmbeddings(model_name=model)

    # Instancia a vector store com persistência
    vector_store = Chroma(
        collection_name="WCAG",
        embedding_function=embeddings_model,
        persist_directory="./chroma_langchain_db",
    )

    # Cria IDs únicos para cada documento
    uuids = [str(uuid4()) for _ in range(len(docs))]
    max_batch_size = 5000

    # Separa os documentos e os IDs em lotes (chunks)
    document_chunks = chunk_docs_list(docs, max_batch_size)
    uuid_chunks = chunk_docs_list(uuids, max_batch_size)

    # Adiciona os lotes à vector store com tratamento de erros
    for docs_chunk, ids_chunk in zip(document_chunks, uuid_chunks):
        try:
            vector_store.add_documents(documents=docs_chunk, ids=ids_chunk)
            print(f"Adicionado com sucesso um lote de {len(docs_chunk)} documentos.")
        except Exception as e:
            print(f"Falha ao adicionar um lote. Erro: {e}")

    return vector_store
