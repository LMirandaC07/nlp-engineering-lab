# Relatório técnico

## Protocolo

Execute `python -m nlp_lab.train` no commit que será citado. O script usa seed 42, remove duplicatas exatas normalizadas, cria treino/validação/teste estratificados e ajusta TF-IDF somente dentro do pipeline de treino. Com 96 exemplos, cada divisão é pequena e as métricas têm incerteza alta.

## Resultados

Execução local em 15/11/2026, Python 3.12.13, seed 42, MLflow 3.18.0 e scikit-learn 1.9.1. A seleção na validação escolheu Linear SVM (Macro F1 0,650). No teste de 15 mensagens, Linear SVM obteve Accuracy 0,333 e Macro F1 0,296. Logistic Regression também obteve 0,333 e 0,296; Naive Bayes 0,400 e 0,404; Dummy 0,133 e 0,029. Os resultados completos, inclusive precisão/recall, estão em `artifacts/comparison.csv` e `artifacts/metrics.json`.

O resultado de teste do Naive Bayes é superior ao do campeão escolhido na validação; isso é um sinal claro de instabilidade causada pelo split mínimo (uma ou duas mensagens por classe em cada holdout), não justificativa para selecionar novamente com base no teste. Em validação, o SVM foi escolhido. Nenhum modelo demonstrou generalização. O modelo salvo reproduz a escolha por validação; artefatos foram gerados localmente e incluídos junto dos números.

A avaliação do retriever TF-IDF usa seis perguntas e IDs de documentos relevantes definidos manualmente. Seus valores são calculados em `artifacts/retrieval_metrics.json`; esse microconjunto verifica o pipeline e não valida a qualidade geral de um RAG. Ele não mede groundedness de texto gerado.

Nesta execução, recuperação obteve Recall@1 = 0,50, Recall@3 = 0,667 e MRR@3 = 0,583 em seis consultas sintéticas. Duas consultas não recuperaram o documento ouro no top 3 (vacinação e preços), mostrando lacunas lexicais reais. O fluxo Gemini foi testado com cliente falso; nenhuma chamada de rede ao provider foi feita, pois não havia chave configurada. Portanto, não existem métricas de resposta LLM, latência de provider ou custo.

## Próximo protocolo de avaliação

Reunir um conjunto externo e governado; reservar teste uma única vez; avaliar por classe e matriz de confusão; medir erro de urgências separadamente; usar validação cruzada estratificada para estabilidade; e comparar modelos GenAI com métricas de recuperação, grounding, latência e custo.
