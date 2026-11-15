from nlp_lab.retrieval import KnowledgeRetriever


def simulate_booking(date: str | None = None) -> dict:
    """Ferramenta local de demonstração: não salva nem confirma um compromisso real."""
    return {"status": "simulated", "date_requested": date,
            "message": "Simulação apenas. Nenhuma consulta foi marcada."}


class SupportAgent:
    """Orquestração explícita de ferramentas sem LLM; política e roteamento são inspecionáveis."""
    def __init__(self, retriever: KnowledgeRetriever | None = None):
        self.retriever = retriever or KnowledgeRetriever()
        self.tools = {"search_knowledge": self.retriever.answer,
                      "simulate_booking": simulate_booking}

    def run(self, message: str, intent: str | None = None) -> dict:
        normalized = message.casefold()
        if intent == "emergencia" or any(word in normalized for word in ("convulsão", "atropelado", "veneno", "respirar")):
            return {"route": "human_or_emergency", "tool": None,
                    "answer": "Procure imediatamente um serviço veterinário de emergência. Não consigo avaliar ou diagnosticar seu pet."}
        if intent == "agendamento" or "agendar" in normalized or "marcar consulta" in normalized:
            result = self.tools["simulate_booking"]()
            return {"route": "booking_demo", "tool": "simulate_booking", **result}
        result = self.tools["search_knowledge"](message)
        return {"route": "retrieval", "tool": "search_knowledge", **result}
