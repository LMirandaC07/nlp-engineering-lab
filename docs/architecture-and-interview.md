# Arquitetura, decisões e entrevista

## Arquitetura e escolhas

- **Python + scikit-learn:** permite um baseline pequeno, rápido, reproduzível e fácil de inspecionar antes de adicionar modelos maiores.
- **TF-IDF + n-gramas:** referência lexical barata para português. Não entende semântica como embeddings; serve para medir um ponto de partida.
- **Dummy, Naive Bayes, Logistic Regression e Linear SVM:** baseline trivial mais modelos lineares adequados a texto esparso. Macro F1 evita esconder classes fracas atrás das classes mais frequentes.
- **Split estratificado e seed:** preservam representação das classes e facilitam repetição. A base é pequena, portanto variação continua alta.
- **MLflow local:** registra parâmetros e métricas sem enviar dados para serviço externo.
- **FastAPI:** contrato HTTP validado, documentação OpenAPI e endpoints separados para predição e fluxo de ferramentas.
- **RAG lexical:** índice pequeno e local, retorna IDs de fonte e abstém quando não há evidência. A resposta determinística é trechada da fonte; o agente Gemini pode usar a mesma ferramenta como contexto.
- **Gemini tool calling:** provider configurado via variável de ambiente, ferramentas allowlisted e execução controlada em até três rodadas; o endpoint isolado pode ser testado sem API key usando um cliente falso.
- **MCP:** adaptador expõe ferramentas por protocolo interoperável. A validação do servidor real deve ser feita com cliente MCP após instalar o extra.

## Decisões e limitações

A amostra sintética permite executar o ciclo inteiro sem compartilhar dados pessoais, mas não sustenta uma alegação de generalização. O fluxo agentic não invoca LLM; separar roteamento e ferramentas primeiro torna efeitos e permissões auditáveis. O próximo incremento deve adicionar um conjunto de avaliação de recuperação antes de mudar para embeddings e geração. A simulação de agendamento não persiste nem chama sistemas externos.

## Problemas previstos e respostas

- **Overfitting/vazamento:** remover duplicatas antes do split; ajustar TF-IDF dentro de Pipeline somente no treino; validar com conjunto maior e estratificado.
- **Alucinação:** nesta versão não há geração livre; recuperador cita fonte ou abstém. Com LLM, limitar contexto e medir groundedness.
- **Prompt injection:** não há LLM ainda; ao adicionar, tratar documentos recuperados como dados não confiáveis, limitar ferramentas e exigir confirmação para ações.
- **Métricas otimistas:** dataset sintético e pequeno; publicar suporte por classe e erros, coletar exemplos independentes, evitar linguagem de produção.
- **Emergência classificada incorretamente:** incluir regras de segurança independentes e avaliação de recall para urgências; a demonstração encaminha, não diagnostica.
- **MCP indisponível:** é extra opcional. A API principal mantém as mesmas funções locais sem o servidor de protocolo.

## Perguntas de entrevista e respostas defendíveis

**Por que começou com TF-IDF e modelos lineares?**

Quis ter um baseline barato e interpretável para verificar qualidade dos dados e estabelecer uma referência antes de pagar o custo de um Transformer. A comparação também mostra se a complexidade extra traz ganho mensurável.

**Por que Macro F1?**

Cada intenção importa, inclusive as menos frequentes. Macro F1 calcula F1 por classe e depois tira média simples, em vez de deixar classes maiores dominarem a pontuação.

**Você fez RAG com embeddings?**

Ainda não. A primeira versão faz recuperação lexical com TF-IDF, retorna fontes e abstém abaixo de um limiar. Embeddings estão no roadmap e precisam ser avaliados contra esse baseline com consultas rotuladas.

**O agente usa um LLM?**

Há duas rotas. `/agent` é determinística e local; `/agent/llm` usa Gemini opcional com function calling manual. O servidor valida o nome/argumentos da ferramenta, executa no máximo três rodadas e pode ser testado com um cliente falso sem enviar dados ao provider.

**Por que integrar MCP?**

Para tornar ferramentas descobríveis por clientes compatíveis sem acoplar a lógica a um único chat. As ferramentas expostas são deliberadamente limitadas e a de agendamento só simula.

**O que faria antes de produção?**

Substituir dados artificiais por dados governados e revisados, separar teste externo, medir recall de urgências, adicionar autenticação/limites, logs sem conteúdo sensível, avaliação contínua, observabilidade de latência/custo e revisão de privacidade.

**Como compararia embeddings?**

Montaria perguntas com fonte relevante conhecida. Compararia Recall@k e MRR de TF-IDF e embeddings; mediria latência e custo. Para respostas, avaliaria fidelidade à fonte, completude e abstinência por revisão humana e métricas automáticas calibradas.

**Como protegeria tool calling?**

Validaria argumentos no servidor, autorizaria cada ferramenta por política, usaria timeout e idempotência, aplicaria rate limit e exigiria confirmação para efeitos externos. O modelo não recebe autoridade implícita só por gerar uma chamada.

**Qual resultado você pode afirmar hoje?**

Somente o que eu rodei localmente e consigo reproduzir. O conjunto é sintético e pequeno; suas métricas não demonstram desempenho de produção.
