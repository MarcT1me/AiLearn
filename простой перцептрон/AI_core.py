import numpy as np


# Ввод-вывод
training_inputs = np.array([
    [0, 0, 1],
    [0, 1, 1],
    [1, 1, 0],
])
training_outputs = np.array([
    [1,
     1,
     0]
]).T

outputs = None

# зерно и веса (для синопса)
np.random.seed(1)

synaptic_weights = 2 * np.random.random((3, 1)) - 1

"""
print('Случайный инициализатор веса:')
print(synaptic_weights)
"""


def sigmoid(x):
    return 1/(1+np.exp(-x))


# Обучение
def ai_learn(len_=100000):
    global synaptic_weights, outputs

    for i in range(len_):
        input_layer = training_inputs
        outputs = sigmoid(np.dot(input_layer, synaptic_weights))

        err = training_outputs - outputs
        adjustments = np.dot(input_layer.T, err * (outputs*(1-outputs)))

        synaptic_weights += adjustments


"""
print('\nВеса после обучения :')
print(synaptic_weights)

print('Результат:')
print(outputs)
"""


def ai_run(person_input):
    return sigmoid(np.dot(np.array(person_input), synaptic_weights))


print(ai_run([1, 1, 0]))
