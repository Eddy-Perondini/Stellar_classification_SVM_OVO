import numpy as np
from itertools import combinations
from collections import Counter
from tqdm import tqdm
# import time 

class SVM:
    def __init__(self, learning_rate = 1e-3, MaxIter = 200, lambdaParam = 1e-4, class_weights = None):
        self.w = None
        self.b = None
        self.MaxIter = MaxIter
        self.class_weights = class_weights
        self.learning_rate = learning_rate
        self.lambdaParam = lambdaParam

    def fit(self, data, target):

        y = np.where(target <= 0, -1, 1) 

        n_samples, n_features = data.shape # (qtd linhas, qtd colunas)

        self.w = np.zeros(n_features)
        self.b = 0
        for iter in tqdm(range(self.MaxIter), desc = 'Treinando a SVM'):

            perm = np.random.permutation(n_samples)

            lr = self.learning_rate / (1 + self.lambdaParam * iter)
                                                                        #O(n^2) ==> O(n) // rodando por volta de 1 minuto por svm binaria (3 * 2)/2 == 3 * 1m50s | 4 min treino total
            for idx in perm: 
                x_i = data[idx]
                y_i = y[idx]
                
                weight = 1.0
                if self.class_weights is not None: 
                    weight = self.class_weights[y_i]

                condition = y_i * (np.dot(x_i, self.w) + self.b) >= 1                       #Novo
                #condition = y_i *  target[idx] * (np.dot(x_i, self.w) - self.b) >= 1       #Antigo

                if condition:
                    self.w -= lr * (2 *  self.lambdaParam * self.w)

                else:
                    self.w -= lr * (2 * self.lambdaParam * self.w - weight * y_i * x_i) 
                    self.b += lr * (weight * y_i) #Corrigido: += aplicado para manter a consistencia da predição 

    def svmPrediction(self, data):
        return np.dot(data, self.w) + self.b #Corrigido: troquei + pelo - na predição (burro kkkk)
    
    def predict(self, data):
        return np.where(self.svmPrediction(data) >= 0, 1, -1)


class MulticlassSVM:
    def __init__(self, **svm_params):
        self.svm_params = svm_params
        self.svm_classifier = {}
    
    def fit(self, data, target): 
        self.classes = np.unique(target)

        for c1, c2 in combinations(self.classes, 2): 

            idx = np.where((target == c1) | (target == c2))[0]
            X_sub, y_sub = data[idx], target[idx]

            y_binario = np.where(y_sub == c1, 1, -1) #Corrigido: estava definindo os parâmetros ao contrário kkkk 

            counts = Counter(y_binario)

            W_neg = len(y_binario) / (2*counts[-1])
            W_pos = len(y_binario) / (2*counts[1])

            class_weights = { 
                -1: W_neg, 
                1: W_pos
            }

            classificador = SVM(**self.svm_params, class_weights=class_weights)
            classificador.fit(X_sub, y_binario)
            self.svm_classifier[(c1, c2)] = classificador

    def predict(self, data):
        votos = np.zeros((data.shape[0], len(self.classes)))
        
        class_idx = {c: i for i, c in enumerate(self.classes)}
 
        for (c1, c2), classificador in self.svm_classifier.items():
            pontos = classificador.svmPrediction(data)
            preds = np.where(pontos >= 0, c1, c2)
 
            for i, p in enumerate(preds):
                votos[i, class_idx[p]] += 1
 
        return self.classes[np.argmax(votos, axis=1)]




