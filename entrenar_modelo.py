import numpy as np
import tensorflow as tf # Se importa TensorFlow, que incluye Keras
from tensorflow.keras.models import Sequential # Se utiliza la API Sequential para construir el modelo
from tensorflow.keras.layers import LSTM, Dense, Embedding # Capas de la red neuronal: Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical # Utilidad para convertir etiquetas a formato one-hot
import pickle # Módulo para serializar (guardar y cargar) objetos Python
import os # Módulo para interactuar con el sistema operativo (por ejemplo, verificar si un archivo existe)

print("Iniciando el script de entrenamiento del modelo de autocompletado...")

# --- Configuración del archivo de corpus ---
# Define la ruta al archivo que contiene tus palabras.
# ¡IMPORTANTE! Asegúrate de que este archivo esté en el mismo directorio que este script,
# o especifica la ruta completa si está en otro lugar.
CORPUS_FILE = "corpus.txt" # ¡Aquí hemos cambiado 'es.txt' a 'corpus.txt'!

# --- Parte 1: Carga y Preparación de Datos ---
print(f"Cargando diccionario de palabras desde '{CORPUS_FILE}'...")
try:
    # Intenta abrir y leer el archivo del corpus.
    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        # Lee cada línea, elimina espacios en blanco al inicio/final (.strip()),
        # convierte a minúsculas (.lower()), y filtra para incluir solo palabras alfabéticas.
        palabras = [line.strip().lower() for line in f if line.strip().isalpha()]
    print(f"Se cargaron {len(palabras)} palabras del corpus.")

    # Si el archivo está vacío o no contiene palabras válidas, usa un corpus de prueba.
    if not palabras:
        print(f"Advertencia: El archivo '{CORPUS_FILE}' está vacío o no contiene palabras válidas. Usando un corpus de prueba para el entrenamiento.")
        palabras = ["hola", "mundo", "proyecto", "ayuda", "inteligencia", "artificial", "teclado", "mano"]

except FileNotFoundError:
    # Si el archivo no se encuentra, se informa y se usa un corpus de prueba.
    print(f"Error: El archivo '{CORPUS_FILE}' no fue encontrado. Creando un corpus básico de prueba en '{CORPUS_FILE}'.")
    palabras = ["hola", "mundo", "proyecto", "ayuda", "inteligencia", "artificial", "teclado", "mano"]
    # Se crea un archivo 'corpus.txt' básico con algunas palabras si no existe.
    with open(CORPUS_FILE, "w", encoding="utf-8") as f:
        for p in ["hola", "mundo", "ayuda", "inteligencia", "artificial", "teclado", "mano", "sistema", "computadora", "lenguaje", "desarrollo"]:
            f.write(p + "\n")
    print("Archivo 'corpus.txt' básico creado. ¡Considera añadir más palabras para un mejor rendimiento del modelo!")

# Prepara el mapeo de caracteres a índices numéricos y viceversa.
# Esto es necesario porque las redes neuronales trabajan con números.
print("Preparando mapeo de caracteres a índices...")
chars = sorted(list(set("".join(palabras)))) # Obtiene una lista ordenada de todos los caracteres únicos presentes en las palabras.
# char_indices: Asigna un número entero a cada carácter único. Se suma 1 porque el 0 se reserva para el padding.
char_indices = {c: i + 1 for i, c in enumerate(chars)}
# indices_char: Es el mapeo inverso, de número entero a carácter.
indices_char = {i + 1: c for i, c in enumerate(chars)}
# Añade el token de "padding" (relleno) con el índice 0.
char_indices["PAD"] = 0
indices_char[0] = "" # El índice 0 corresponde a un caracter vacío para el padding.

# Calcula la longitud máxima de palabra encontrada en el corpus.
# Esto se usará para asegurar que todas las secuencias de entrada tengan el mismo tamaño.
maxlen = max(len(p) for p in palabras)
print(f"Longitud máxima de palabra en el corpus: {maxlen}")
print(f"Tamaño del vocabulario de caracteres: {len(char_indices)} (incluyendo 'PAD')")

# Prepara los datos de entrenamiento para la red neuronal.
# 'X' contendrá las secuencias de caracteres (prefijos) de entrada.
# 'y' contendrá el siguiente caracter que el modelo debe predecir.
print("Generando secuencias de entrenamiento (pares prefijo -> siguiente caracter)...")
X = [] # Lista para almacenar las secuencias de entrada codificadas.
y = [] # Lista para almacenar los caracteres de salida codificados.

for palabra in palabras:
    for i in range(1, len(palabra)): # Itera desde el segundo carácter hasta el final de la palabra.
        seq = palabra[:i]       # El prefijo de la palabra (ej., "h" de "hola", "ho" de "hola").
        next_char = palabra[i]  # El siguiente caracter en la secuencia (ej., "o" de "hola", "l" de "hola").

        # Codifica el prefijo (secuencia) a una lista de números usando 'char_indices'.
        seq_encoded = [char_indices[c] for c in seq if c in char_indices]

        # Realiza el "padding" (relleno) de la secuencia.
        # Las redes neuronales LSTM esperan secuencias de entrada de tamaño fijo.
        # Si la secuencia es más corta que (maxlen - 1), se añaden ceros al principio.
        while len(seq_encoded) < maxlen - 1: # maxlen - 1 porque el último carácter de la palabra es 'y'.
            seq_encoded.insert(0, 0) # Inserta 0s (representando 'PAD') al inicio.

        X.append(seq_encoded) # Añade la secuencia de entrada codificada y rellena a la lista X.
        y.append(char_indices[next_char]) # Añade el siguiente caracter codificado a la lista y.

# Convierte las listas de Python a arrays de NumPy, que son más eficientes para Keras.
X = np.array(X)
# Convierte los caracteres de salida 'y' a formato "one-hot encoding".
# En este formato, cada caracter se representa como un vector donde solo la posición
# correspondiente a ese caracter es 1 y el resto son 0s.
y = to_categorical(y, num_classes=len(char_indices))

print(f"Número total de secuencias de entrenamiento generadas: {len(X)}")
print(f"Forma del array de entrada (X): {X.shape}") # (número_de_muestras, longitud_de_secuencia_de_entrada)
print(f"Forma del array de salida (y): {y.shape}")   # (número_de_muestras, tamaño_del_vocabulario_de_caracteres)

# --- Parte 2: Creación y Entrenamiento del Modelo de Red Neuronal ---
print("Configurando la arquitectura del modelo de red neuronal LSTM...")
model = Sequential() # Se inicia un modelo secuencial (capas apiladas una tras otra).

# Capa de Embedding:
# Mapea cada índice de caracter (número) a un vector denso de tamaño fijo (16 en este caso).
# Esto ayuda al modelo a aprender relaciones semánticas entre caracteres.
model.add(Embedding(input_dim=len(char_indices), output_dim=16, input_length=maxlen - 1))

# Capa LSTM (Long Short-Term Memory):
# Una capa recurrente que es muy buena para procesar secuencias y capturar dependencias a largo plazo.
# Los 64 representan el número de unidades LSTM (la "memoria" de la capa).
model.add(LSTM(64))

# Capa Densa de Salida:
# Es una capa de clasificación que producirá la probabilidad de cada caracter posible.
# El número de neuronas es igual al tamaño total de caracteres en nuestro vocabulario.
# La función de activación 'softmax' convierte las salidas en probabilidades que suman 1.
model.add(Dense(len(char_indices), activation='softmax'))

# Compila el modelo:
# Se define la función de pérdida (categorical_crossentropy para clasificación multiclase),
# el optimizador (adam es una buena opción general) y las métricas a monitorear (precisión).
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Muestra un resumen de la arquitectura del modelo, incluyendo el número de parámetros entrenables.
model.summary()

print("Iniciando el entrenamiento del modelo (esto puede tardar varios minutos, por favor sé paciente)...")
# Entrena el modelo usando los datos preparados.
# epochs: El número de veces que el modelo procesará todo el conjunto de datos de entrenamiento.
# batch_size: El número de muestras que se procesan antes de que el modelo actualice sus pesos.
# verbose=1: Muestra una barra de progreso y métricas en la consola durante el entrenamiento.
model.fit(X, y, epochs=10, batch_size=256, verbose=1)

print("¡Entrenamiento del modelo completado!")

# --- Parte 3: Guardado del Modelo y Diccionarios ---
# Define los nombres de los archivos donde se guardarán los resultados.
MODEL_FILE = "autocompletar_palabra.h5"
CHAR_INDICES_FILE = "char_indices.pkl"
INDICES_CHAR_FILE = "indices_char.pkl"

print(f"Guardando el modelo entrenado en '{MODEL_FILE}'...")
# Guarda el modelo completo (su arquitectura y sus pesos aprendidos) en un archivo .h5.
model.save(MODEL_FILE)

print(f"Guardando el diccionario char_indices en '{CHAR_INDICES_FILE}'...")
# Guarda el diccionario que mapea caracteres a números usando pickle.
# Esto es esencial para que la aplicación sepa cómo codificar las entradas al modelo.
with open(CHAR_INDICES_FILE, "wb") as f:
    pickle.dump(char_indices, f)

print(f"Guardando el diccionario indices_char en '{INDICES_CHAR_FILE}'...")
# Guarda el diccionario que mapea números a caracteres usando pickle.
# Esto es esencial para que la aplicación pueda decodificar las predicciones del modelo.
with open(INDICES_CHAR_FILE, "wb") as f:
    pickle.dump(indices_char, f)

print("\n--- Proceso de entrenamiento y guardado completado ---")
print(f"¡Todos los archivos necesarios para la predicción han sido guardados correctamente en el directorio actual!")
print(f"- {MODEL_FILE}")
print(f"- {CHAR_INDICES_FILE}")
print(f"- {INDICES_CHAR_FILE}")
print("\nAhora puedes modificar la clase 'NeuralWordPredictor' en tu script principal para cargar estos archivos en lugar de entrenar el modelo en cada inicio. Esto hará que tu aplicación sea mucho más rápida.")
