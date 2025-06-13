import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
from keras.utils import to_categorical
import pickle

# Carga un diccionario grande de palabras en español
with open("es.txt", "r", encoding="utf-8") as f:
    palabras = [line.strip().lower() for line in f if line.strip().isalpha()]

# Prepara datos
chars = sorted(list(set("".join(palabras))))
char_indices = {c: i+1 for i, c in enumerate(chars)}  # +1 para padding=0
indices_char = {i+1: c for i, c in enumerate(chars)}
char_indices["PAD"] = 0
indices_char[0] = ""

maxlen = max(len(p) for p in palabras)
X = []
y = []
for palabra in palabras:
    for i in range(1, len(palabra)):
        seq = palabra[:i]
        next_char = palabra[i]
        seq_encoded = [char_indices[c] for c in seq]
        # Padding
        while len(seq_encoded) < maxlen - 1:
            seq_encoded.insert(0, 0)
        X.append(seq_encoded)
        y.append(char_indices[next_char])

X = np.array(X)
y = to_categorical(y, num_classes=len(char_indices))

# Modelo
model = Sequential()
model.add(Embedding(len(char_indices), 16, input_length=maxlen-1))
model.add(LSTM(64))
model.add(Dense(len(char_indices), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=10, batch_size=256, verbose=1)

# Guarda modelo y diccionarios
model.save("autocompletar_palabra.h5")
with open("char_indices.pkl", "wb") as f:
    pickle.dump(char_indices, f)
with open("indices_char.pkl", "wb") as f:
    pickle.dump(indices_char, f)