import numpy as np

# função de ativação


def nonlin(x, deriv=False):
    if(deriv == True):  # se deriv = true, retorna o valor custo
        return (x*(1-x))

    return 1/(1+np.exp(-x))


def train():
    global syn0, syn1
    # dados de entrada - [pedra, papel, tesoura]
    input_data = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])

    # dado de sáida
    output_data = np.array([[0, 1, 0],
                            [0, 0, 1],
                            [1, 0, 0]])
    for j in range(65000):

        # calcula a ativação de cada neurônio
        layer0 = input_data
        layer1 = nonlin(np.dot(layer0, syn0))
        layer2 = nonlin(np.dot(layer1, syn1))

        # propaga dos erros de trás para frente
        layer2_error = output_data - layer2

        layer2_delta = layer2_error*nonlin(layer2, deriv=True)

        layer1_error = layer2_delta.dot(syn1.T)

        layer1_delta = layer1_error * nonlin(layer1, deriv=True)

        # reajusta os valores dos pesos
        syn1 += layer1.T.dot(layer2_delta)
        syn0 += layer0.T.dot(layer1_delta)


def test(input_data):
    # calcula a ativação de cada neurônio
    layer0 = input_data
    layer1 = nonlin(np.dot(layer0, syn0))
    layer2 = nonlin(np.dot(layer1, syn1))

    if max(layer2) == layer2[0]:
        return 'Pedra'
    if max(layer2) == layer2[1]:
        return 'Papel'
    if max(layer2) == layer2[2]:
        return 'Tesoura'
        

# fixa a semente para a geração dos números aleatórios
np.random.seed(1)

# define, aleatoriamente, os pesos de cada conexão criando uma matrix 3x3 (3 entradas x 3 neurônios na segunda camada)
syn0 = 2*np.random.random((3, 3)) - 1
# define, aleatoriamente, os pesos de cada conexão criando uma matrix 3x3 (3 neurônios na segunda camada x 3 saídas)
syn1 = 2*np.random.random((3, 3)) - 1

print("Treinando a rede neural...")
train()

print("\n\tPPTRN - Pedra, Papel e Tesoura (Rede Neural)")
print("\nEntre com uma das escolhas do jogo e a IA irá tentar vencê-lo.\n(Escolha entre pedra, papel ou tesoura)")
while True:
    x = input("\n\nVocê: ")
    while x.lower() not in ['pedra', 'papel', 'tesoura']:
        x = input("Entrada inválida!\nVocê: ")

    # transforma a entrada do usuário em um array que pode ser processesado pela RN
    if x.lower() == 'pedra':
        input_data = np.array([1, 0, 0])
    if x.lower() == 'papel':
        input_data = np.array([0, 1, 0])
    if x.lower() == 'tesoura':
        input_data = np.array([0, 0, 1])

    print("IA: " + str(test(input_data)))
