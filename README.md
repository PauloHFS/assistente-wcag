# Assistente Técnico WCAG

## Como rodar o projeto

```bash
devcontainer open . 
```


```bash
streamlit run app/main.py --server.address=0.0.0.0
```

```bash
uv pip compile pyproject.toml --extra dev --extra ingestion --extra evaluation -o requirements.txt
```