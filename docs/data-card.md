# Data Card — intent messages

## Resumo

`data/intents.csv` contém 96 frases sintéticas em português brasileiro, com 12 exemplos para cada intenção: `agendamento`, `cancelamento`, `precos`, `horarios`, `emergencia`, `vacinacao`, `sintomas` e `localizacao`.

## Origem e coleta

Os exemplos foram escritos manualmente para este protótipo. Não vieram de clientes, clínicas, iFood, prontuários ou coleta de internet. Não são exemplos de conversas reais. Não contêm identificadores pessoais.

## Uso pretendido

Ensino e teste local de validação, split estratificado, TF-IDF, classificação e análise de erros. Não usar para decisões clínicas, atendimento real ou estimar performance de produção.

## Limitações e riscos

O tamanho e a diversidade linguística são reduzidos. Há pouca variação regional, erros de digitação, mensagens multilíngues, múltiplas intenções, contexto conversacional e exemplos ambíguos. O vocabulário foi escrito pelos autores e pode favorecer as fronteiras das classes. Scores de teste são instáveis com duas amostras por classe no holdout.

## Governança

Antes de uso real: coletar dados com base legal e consentimento apropriados, remover dados pessoais, estabelecer revisão humana, criar conjunto de teste independente e revisar segurança/privacidade. Atualizações devem registrar origem, versão e alterações do dataset.
