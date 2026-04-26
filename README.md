# SVM do Zero One-Vs-One para Classificação Estelar
## Breve introdução 

Esse repositório tem como objetivo mostrar a criação de uma SVM do zero para o problema de multiclassificação utilizando a base de dados de classificação estelar (estrelas, galáxias e quasares) referenciada abaixo. 

## Potenciais melhorias 

Apesar da alta acurácia de classificação proporcionada pela SVM, a estruturação do código pode ser melhorada, pois o componente de multiclassificação da SVM, por ser OVO, treina um número $\frac{n * (n-1)}{2}$ de classificadores binários. 
Assim, o loop criado para a atualização dos pesos por parte dos classificadores binários da SVM está sendo um potencial gargalo para o modelo, levando cerca de 1 minuto e meio para ser treinado no total. 

## Treinamento 

 - Treino: 70% e Teste: 30% ('train_test_split()'); '$\rightarrow$' Mas, pode-se tentar kFold Cross-Validation para avaliar a mudança da performance 
  
 - O que foi aplicado para o pré-processamento do BD:

'StandardScaler()';
remoção de outliers 'df_stellar_filtrado = df_stellar[df_stellar['u'] > -9999]';
corte brusco nos dados quanto à classe "GALAXY" num intervalo de '[:25000]' para lidar com o desbalanceamento gigantesco.

# Resultados
| Class  | Precision | Recall | F1-Score | Support |
| ------ | --------- | ------ | -------- | ------- |
| GALAXY | 0.94      | 0.92   | 0.93     | 7470    |
| QSO    | 0.96      | 0.93   | 0.94     | 5703    |
| STAR   | 0.95      | 1.00   | 0.97     | 6494    |

| Metric       | Value  |
| ------------ | ------ |
| Accuracy     | 0.9488 |
| Macro Avg F1 | 0.95   |
| Weighted F1  | 0.95   |

<img width="448" height="399" alt="image" src="https://github.com/user-attachments/assets/90a6b199-906e-4e51-a3c4-b096ecc9c153" />

# Referência

- Fedesoriano. (2022). Stellar Classification Dataset - SDSS17. Kaggle. Retrieved April 26, 2026, from https://www.kaggle.com/fedesoriano/stellar-classification-dataset-sdss17
