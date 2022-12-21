from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense
from keras.models import Model

# Define a camada de entrada com 10 timesteps(intervalos de tempo) e 1 característica(valor)
input_layer = Input(shape=(10, 1))

# Define uma camada de convolução com 32 filtros de tamanho 3
conv_layer = Conv1D(32, 3, activation='relu')(input_layer)

# Define uma camada de pooling com tamanho de pooling 2
pooling_layer = MaxPooling1D(2)(conv_layer)

# Define uma camada de flatten para aplanar as saídas da camada de pooling
flatten_layer = Flatten()(pooling_layer)

# Define uma camada densa com 32 neurônios
dense_layer = Dense(32, activation='relu')(flatten_layer)

# Define a camada de saída com 1 neurônio
output_layer = Dense(1, activation='sigmoid')(dense_layer)

# Cria o modelo da rede neural com a camada de entrada, a camada de convolução, a camada de pooling, a camada de flatten, a camada densa e a camada de saída
model = Model(inputs=input_layer, outputs=output_layer)

# Compila o modelo com o otimizador Adam e a função de perda binary_crossentropy
model.compile(optimizer='adam', loss='binary_crossentropy')

# Fornece os dados de treinamento para o modelo
X = [
    [[0], [1], [1], [0], [0], [0], [0], [0], [0], [0]], 
    [[0], [0], [0], [0], [0], [0], [0], [0], [1], [1]]
]
y = [0, 1]
model.fit(X, y, epochs=100)

# Faz uma previsão com o modelo treinado
print(model.predict([[[0], [1], [1], [0], [0], [0], [0], [0], [0], [0]]])) # deve imprimir algo próximo de 0
print(model.predict([[[0], [0], [0], [0], [0], [0], [0], [0], [1], [1]]])) # deve imprimir algo próximo de 1
