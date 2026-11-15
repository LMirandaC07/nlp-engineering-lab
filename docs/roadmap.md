# Roadmap priorizado para AI Engineer / GenAI

## Entregue nesta base

Classificação tradicional, dados sintéticos validados, MLflow local, API, agente Gemini opcional com function calling controlado, fluxo determinístico local, recuperação lexical com fonte/abstenção, adaptador MCP e CI.

## Próximas iterações

1. Ampliar o conjunto inicial de seis consultas sintéticas para gold set revisado por humano; medir groundedness, relevância e abstenção das respostas.
2. Comparar embeddings multilíngues/português, reranking e estratégias de chunking, registrando custo e latência.
3. Versionar prompts e avaliações; adicionar defesa em camadas contra prompt injection e validação de saídas.
4. Integrar timeouts, limites de taxa, autorização e confirmação explícita para ferramentas com efeitos externos; as ferramentas do protótipo são locais/simuladas.
5. Executar BERTimbau ou alternativa em português no Colab somente após coletar dataset adequado; versionar dataset, checkpoint e avaliação.
6. Adicionar observabilidade, avaliação em CI, testes de segurança e plano de deploy com segredos externos.

O escopo prioriza sistemas de GenAI de ponta a ponta sem abandonar ML tradicional. Cada capacidade permanece marcada como pendente até existir implementação executada e evidência verificável.
