"""Servidor MCP opcional com ferramentas locais para consulta da base e agendamento simulado."""
from nlp_lab.agent import simulate_booking
from nlp_lab.retrieval import KnowledgeRetriever


def main() -> None:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise SystemExit("Instale a dependência opcional: pip install -e '.[mcp]'") from exc
    mcp = FastMCP("nlp-engineering-lab")
    retriever = KnowledgeRetriever()

    @mcp.tool()
    def search_veterinary_faq(query: str) -> dict:
        """Busca uma resposta e fontes na base demonstrativa de perguntas frequentes."""
        return retriever.answer(query)

    @mcp.tool()
    def simulate_appointment(date: str | None = None) -> dict:
        """Simula um pedido de agendamento; não cria compromisso real."""
        return simulate_booking(date)

    mcp.run()


if __name__ == "__main__":
    main()
