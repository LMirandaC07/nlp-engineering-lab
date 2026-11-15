# Model Card — classificador de intenções

## Estado

Pipeline local de referência com TF-IDF e modelos scikit-learn. O modelo salvo só passa a existir após `python -m nlp_lab.train`. O treinamento seleciona pelo Macro F1 na validação e usa o teste separado para reportar a comparação final.

## Uso pretendido

Demonstração técnica de classificação de mensagens sintéticas em oito intenções. Não é serviço veterinário nem componente aprovado para produção.

## Avaliação

O script registra Accuracy, Macro Precision, Macro Recall, Macro F1, classification report, confusion matrix e erros. O tempo registrado é de ajuste local, não latência de inferência de produção. Scores SVM são margens, não probabilidades calibradas.

## Limitações

Dados pequenos e sintéticos; pouca robustez a distribuição diferente, novas classes, linguagem informal variada, entradas adversariais e múltiplas intenções. Não deve decidir atendimento clínico. Mensagens de urgência têm uma rota de segurança, mas regras e classificador precisam de avaliação dedicada antes de qualquer uso real.

## Próximos passos

Mais dados representativos e revisados, avaliação temporal/externa, calibração de confiança, política explícita de rejeição, avaliação de subgrupos e comparação com Transformers em português.
