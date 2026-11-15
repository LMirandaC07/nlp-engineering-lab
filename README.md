# NLP Engineering Lab

Laboratório reproduzível de NLP e AI Engineering em português. O primeiro caso usa mensagens **sintéticas** de atendimento veterinário para estudar classificação de intenção, recuperação de conhecimento, roteamento explícito para ferramentas, avaliação e serving. A inspiração de domínio vem do VetVoice; os dados e respostas deste repositório são apenas demonstrativos.

O projeto mira competências da trilha **AI Engineer** do Programa de Estágio GenAI do iFood: integração de software e IA, modelos de linguagem, operacionalização e pipelines. Esta é uma escolha de portfólio baseada na descrição pública da vaga; não representa conhecimento interno do iFood nem garante resultado seletivo.

## O que funciona

- Dataset local pequeno com 8 intenções, validação, deduplicação e alerta sobre dados sintéticos.
- Comparação de Dummy, Naive Bayes, Logistic Regression e Linear SVM com TF-IDF.
- Métricas de teste, matriz de confusão, relatório por classe e exemplos de erro, com seed fixa.
- Rastreamento opcional dos experimentos com MLflow local.
- API FastAPI para predição e para um fluxo demonstrativo que escolhe entre recuperação, simulação de agendamento e encaminhamento de urgência.
- Agente Gemini opcional com tool calling manual, allowlist de ferramentas, limite de iterações e métricas de uso/latência.
- Recuperação lexical TF-IDF de FAQs com fontes citadas e abstinência quando não encontra evidência suficiente.
- Avaliação offline de recuperação com Recall@1, Recall@3 e MRR sobre seis consultas sintéticas.
- Servidor MCP opcional com ferramentas de busca e simulação.
- Testes, Docker e GitHub Actions.

O endpoint `/agent` continua determinístico e sem custo externo. `/agent/llm` oferece a versão opcional com Gemini e chamadas de ferramentas controladas. O recuperador usa TF-IDF lexical; não é busca vetorial por embeddings. Não há fine-tuning de Transformer nesta primeira entrega. Veja [docs/roadmap.md](docs/roadmap.md) para a evolução GenAI.

## Executar no Windows

Requer Python 3.11+.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,mcp,genai]"
python -m nlp_lab.train
python -m nlp_lab.eval_rag --tracking
pytest
uvicorn nlp_lab.api:app --reload
```

Abra `http://127.0.0.1:8000/docs`. Para rastrear experimentos, a execução do treinamento cria o banco local `mlflow.db`; visualize com `mlflow ui --backend-store-uri sqlite:///mlflow.db`. Desative o tracking passando `tracking=False` ao chamar `train()` em código Python.

Exemplos:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict `
  -ContentType 'application/json' -Body '{"text":"Quero marcar consulta para meu gato"}'
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/agent `
  -ContentType 'application/json' -Body '{"text":"Quais são os horários da clínica?"}'
```

`LinearSVC` retorna margens (`decision_function`), não probabilidades calibradas. A resposta da API identifica a semântica do score.

## Agente Gemini e MCP

`POST /agent/llm` habilita o agente opcional de tool calling Gemini. Configure `GEMINI_API_KEY` e `GEMINI_MODEL` no ambiente local depois de instalar o extra `genai`. A chave nunca deve ser salva no `.env` versionado. Sem configuração, a rota responde 503 e o restante da API funciona normalmente. A execução das chamadas é manual, validada por allowlist, limitada a três rodadas e devolve latência e contagem de tokens; não estima custo monetário.

O function calling segue o fluxo documentado pelo SDK Google GenAI: o modelo propõe chamadas, e o servidor executa ferramentas explicitamente permitidas. Consulte a [documentação oficial](https://ai.google.dev/gemini-api/docs/function-calling).

O servidor MCP Python expõe `search_veterinary_faq` e `simulate_appointment`. A ferramenta de agendamento é uma simulação local sem persistência ou integração externa.

## Arquitetura

```text
CSV sintético → validação/deduplicação → split estratificado
                                  └── TF-IDF + 4 classificadores → métricas/MLflow/artefatos
                                                                  └── FastAPI /predict
Pergunta → classificador/intenção → política de roteamento ────────┬── FAQ TF-IDF + fonte
                                                                   ├── agendamento simulado
                                                                   └── encaminhamento de urgência
MCP client → MCP server → ferramentas de FAQ e simulação
```

Detalhes, decisões, limites e perguntas de entrevista: [docs/architecture-and-interview.md](docs/architecture-and-interview.md).

## Evidências e limites

O dataset tem 96 mensagens sintéticas (12 por classe), elaboradas para exercitar o pipeline. É pequeno, artificial e próximo do domínio do protótipo. O split de teste tem poucas amostras; as métricas geradas localmente servem para validar execução e comparação, não generalização, produção ou qualquer alegação clínica. Resultados reais desta execução estão em `artifacts/metrics.json`, `artifacts/comparison.csv` e `artifacts/error_analysis.csv`; o modelo serializado serve para inferência local com scikit-learn compatível. Leia o [Data Card](docs/data-card.md) e o [Model Card](docs/model-card.md).

Consulte `docs/technical-report.md` para o protocolo e os detalhes desta execução reproduzível.

Resultado desta execução em 15 exemplos de teste: Dummy Macro F1 0,029; Naive Bayes 0,404; Logistic Regression 0,296; Linear SVM 0,296. SVM foi selecionado pela validação (Macro F1 0,650), não pelo teste. Uma ou duas frases por classe no holdout tornam essa pontuação muito instável.

O retriever teve Recall@1 0,50, Recall@3 0,667 e MRR@3 0,583 em seis consultas de avaliação sintéticas; duas consultas falharam no top 3. O fluxo Gemini foi exercitado com cliente falso, sem uma chamada real ao provider ou medição de custo/latência de rede.

## Roadmap GenAI

1. Aumentar o conjunto sintético inicial de 6 consultas para um gold set maior revisado por humano; avaliar groundedness, relevância e abstinência de respostas.
2. Comparar TF-IDF a embeddings em português e um reranker, medindo latência e custo.
3. Versionar prompts e adicionar proteção dedicada contra prompt injection; o system prompt atual não substitui uma validação de segurança.
4. Evoluir autorização, timeouts, rate limits, logs sem dados sensíveis e confirmação de ações para ferramentas com efeitos externos.
5. Treinar e comparar BERTimbau (ou modelo português equivalente) quando dataset maior e ambiente de GPU estiverem disponíveis; registrar a execução, sem prometer métrica antecipada.
6. Versionar datasets e modelos, avaliar drift, exportar telemetria agregada e planejar deploy após validação.

## Segurança

Não insira credenciais, dados reais de clientes ou dados pessoais. A aplicação não oferece conselho veterinário, diagnóstico ou prescrição. O endpoint de agendamento não marca consultas reais. Em emergência, encaminha para atendimento profissional.

Licença: MIT. Consulte `LICENSE`.
