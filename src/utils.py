import os
import sys

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

OLLAMA_HOST = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

def check_ollama_connection():
    try:
        llm = ChatOllama(base_url=OLLAMA_HOST, model="qwen3:8b", temperature=0)
        prompt = ChatPromptTemplate.from_messages([("system", "Responda 'OK' e nada mais."), ("human", "{user_input}")])
        chain = prompt | llm
        response = chain.invoke({"user_input": "Teste"})
        
        if "OK" in response.content:
            print("Conexão com o serviço Ollama bem-sucedida.")
            return True
        else:
            print("Serviço Ollama respondeu, mas a resposta não foi a esperada.")
            return False
            
    except Exception as e:
        print(f"Erro ao conectar com o serviço Ollama: {e}", file=sys.stderr)
        print("Verifique se o Ollama está em execução localmente.", file=sys.stderr)
        return False
