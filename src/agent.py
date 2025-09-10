from typing import TypedDict


class State(TypedDict):
    generation: str

DISCLAIMER_TEXT = (
    "\n\n---"
    "\n**Aviso**: Esta ferramenta é uma Prova de Conceito (PoC) e suas "
    "respostas podem conter imprecisões ou ser incompletas. Verifique "
    "as informações antes de utilizá-las."
)

def safety_agent(state: State) -> State:
    current_generation = state.get("generation", "")
    updated_generation = current_generation + DISCLAIMER_TEXT
    
    state["generation"] = updated_generation
    
    return state
