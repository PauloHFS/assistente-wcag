import time

import streamlit as st


def mock_agent_graph(query: str):
    print(f"Buscando resposta para: {query}")
    time.sleep(3)

    response = {
        "answer": "A computação em nuvem oferece escalabilidade, flexibilidade e custos reduzidos, permitindo que as empresas acessem recursos de computação sob demanda pela internet. Os principais modelos de serviço são IaaS, PaaS e SaaS.",
        "sources": [
            {
                "url": "https://aws.amazon.com/what-is-cloud-computing/",
                "snippet": "A computação em nuvem é a entrega sob demanda de poder computacional, banco de dados, armazenamento, aplicações e outros recursos de TI através de uma plataforma de serviços de nuvem via internet com definição de preço de pagamento conforme o uso.",
            },
            {
                "url": "https://azure.microsoft.com/en-us/resources/cloud-computing-dictionary/what-is-cloud-computing/",
                "snippet": "Simply put, cloud computing is the delivery of computing services—including servers, storage, databases, networking, software, analytics, and intelligence—over the Internet (“the cloud”) to offer faster innovation, flexible resources, and economies of scale.",
            },
            {
                "url": "https://cloud.google.com/learn/what-is-cloud-computing",
                "snippet": "Cloud computing is a way to deliver on-demand computing services over the internet on a pay-as-you-go basis. Instead of buying, owning, and maintaining your own data centers and servers, you can access technology services, such as computing power, storage, and databases, from a cloud provider like Google Cloud.",
            },
        ],
    }
    return response


st.set_page_config(layout="wide")
st.title("Assistente de Pesquisa com IA")
st.markdown(
    "Faça uma pergunta para obter uma resposta consolidada a partir de múltiplas fontes, juntamente com as referências utilizadas."
)

with st.form(key="query_form"):
    user_question = st.text_input(
        "Qual é a sua pergunta?",
        key="user_question",
        placeholder="Ex: O que é computação em nuvem?",
    )
    submit_button = st.form_submit_button(label="Perguntar")

if submit_button and user_question:
    with st.spinner("Analisando fontes e compilando a resposta..."):
        result = mock_agent_graph(user_question)

    st.divider()

    st.subheader("Resposta")
    st.markdown(result["answer"])

    st.subheader("Fontes")
    for source in result["sources"]:
        with st.expander(f"**Fonte:** {source['url']}"):
            st.markdown(f"> {source['snippet']}")
