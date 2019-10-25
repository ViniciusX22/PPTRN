import numpy as np


def nonlin(x, deriv=False):
    if(deriv == True):
        return (x*(1-x))

    return 1/(1+np.exp(-x))

# teste para simular o operador lógico OR

# dado de entrada
input_data = np.array([[1, 0],
                       [0, 1],
                       [0, 0],
                       [1, 1]])

# dado de saída
output_nada = np.array([[1],
                        [1],
                        [0],
                        [1]])

np.random.seed(1) # fixa a geração dos números para a mesma seed

# define, aleatoriamente, os pesos de cada conexão criando uma matrix 2x3 (2 entradas x 3 neurônios na segunda camada)
syn0 = 2*np.random.random((2, 3)) - 1
# define, aleatoriamente, os pesos de cada conexão criando uma matrix 3x1 (3 neurônios na segunda camada x 1 saída)
syn1 = 2*np.random.random((3, 1)) - 1

for j in range(60000):

    # calcula a ativação de cada neurônio
    l0 = input_data
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))

    # propaga dos erros de trás para frente
    l2_error = output_nada - l2

    if(j % 10000) == 0:
        print("Erro: " + str(np.mean(np.abs(l2_error))))

    l2_delta = l2_error*nonlin(l2, deriv=True)

    l1_error = l2_delta.dot(syn1.T)

    l1_delta = l1_error * nonlin(l1, deriv=True)

    # reajusta os valores dos pesos
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

print("Saída após treinamento:")
print(l2)
