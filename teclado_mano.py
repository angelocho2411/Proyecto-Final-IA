# main.py (Versión con Predictor Neuronal LSTM - 12 de Junio, 2025)
# REQUISITOS:
# pip install pygame opencv-python mediapipe numpy tensorflow

import pygame
import sys
import cv2
import mediapipe as mp
import numpy as np
from collections import defaultdict
import time
import os
# Desactiva los logs de TensorFlow para una consola más limpia
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# ==============================================================================
# SECCIÓN 1: CONSTANTES DE CONFIGURACIÓN (Diseño y funcionalidad)
# ==============================================================================
# Esta sección define todas las constantes utilizadas en la aplicación,
# desde las dimensiones de la pantalla hasta la paleta de colores y el diseño del teclado.
# Agruparlas aquí facilita la configuración y personalización.

# --- Configuración de Pantalla ---
USE_FULLSCREEN = False  # Si es True, la aplicación se inicia en pantalla completa.
SCREEN_WIDTH = 1280     # Ancho predeterminado de la ventana si no está en pantalla completa.
SCREEN_HEIGHT = 720     # Alto predeterminado de la ventana si no está en pantalla completa.
FPS = 30                # Fotogramas por segundo a los que se ejecutará la aplicación.

# --- Paleta de colores ---
# Definición de colores en formato RGB (Rojo, Verde, Azul).
COLOR_BACKGROUND = (255, 255, 255)         # Fondo principal de la aplicación (blanco).
COLOR_TEXT_ON_DARK = (240, 240, 240)       # Color de texto para fondos oscuros.
COLOR_TEXT_ON_LIGHT = (32, 32, 32)         # Color de texto para fondos claros.
COLOR_KEY_HOVER = (0, 150, 130)            # Color de la tecla cuando el cursor la está "tocando".
COLOR_SUGGESTION_DEFAULT = (107, 118, 139) # Color de fondo para las sugerencias de palabras.
COLOR_TEXT_AREA_BG = (183, 206, 212)       # Color de fondo del área donde se muestra el texto escrito.
COLOR_HAND_CURSOR = (230, 0, 0)            # Color del cursor que sigue la mano (rojo).
COLOR_CAM_BG = (210, 210, 210)             # Color de fondo para la visualización de la cámara.
COLOR_STATUS_TEXT = (100, 100, 100)        # Color para mensajes de estado.
COLOR_CALIBRATION_TARGET = (0, 150, 255)   # Color del objetivo durante la calibración.

# --- Paleta de colores del Teclado ---
# Colores específicos para diferentes filas de teclas para una mejor distinción visual.
COLOR_KEY_ROW_NUM = (45, 45, 45)           # Fila de números y QWERTY (gris oscuro).
COLOR_KEY_ROW_A = (53, 70, 93)             # Fila ASDF (azul oscuro).
COLOR_KEY_ROW_Z = (107, 118, 139)          # Fila ZXCV (gris azulado).
COLOR_KEY_SPECIAL_NEW = (183, 206, 212)    # Teclas especiales como Mayús, Borrar, Espacio (azul claro).

# --- Diseño del Teclado y Fuentes ---
VIDEO_FEED_WIDTH = 240      # Ancho del recuadro de video de la cámara.
VIDEO_FEED_HEIGHT = 180     # Alto del recuadro de video de la cámara.

# Disposición de las teclas en el teclado virtual.
# Cada sublista representa una fila.
KEY_LAYOUT = [
    ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
    ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
    ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'Ñ'],
    ['MAYÚS', 'Z', 'X', 'C', 'V', 'B', 'N', 'M', '←'],
    ['␣'] # Barra espaciadora, '␣' es un símbolo placeholder
]

FONT_SIZE_KEYS = 32         # Tamaño de fuente para las letras en las teclas.
FONT_SIZE_TEXT = 30         # Tamaño de fuente para el texto en el área de escritura.
FONT_SIZE_STATUS = 22       # Tamaño de fuente para los mensajes de estado.
FONT_SIZE_CALIBRATION = 40  # Tamaño de fuente para los mensajes de calibración.

# --- Constantes funcionales ---
INDEX_FINGER_TIP = 8    # Índice del punto de la punta del dedo índice en el modelo de MediaPipe.
THUMB_TIP = 4           # Índice del punto de la punta del dedo pulgar en el modelo de MediaPipe.
CLICK_THRESHOLD = 0.04  # Distancia umbral entre pulgar e índice para detectar un "clic".
CLICK_COOLDOWN_MS = 500 # Tiempo en milisegundos para evitar clics múltiples rápidos.
SMOOTHING_FACTOR = 0.2  # Factor de suavizado para el movimiento del cursor de la mano.
                        # Un valor más alto significa menos suavizado (más reactivo).
CALIBRATION_PADDING = 0.05 # Margen adicional para el área de calibración de la mano.

# --- Archivos Externos ---
# Rutas a archivos de recursos, construidas de forma robusta para diferentes sistemas operativos.
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Directorio base del script.
SOUND_FILE_CLICK = os.path.join(BASE_DIR, "click.wav") # Ruta al archivo de sonido de clic.
CORPUS_FILE = os.path.join(BASE_DIR, "corpus.txt")     # Ruta al archivo de texto para el corpus del predictor.


# ==============================================================================
# SECCIÓN 2: CLASES DE MÓDULOS FUNDAMENTALES
# ==============================================================================
# Esta sección contiene las definiciones de clases que encapsulan lógicas
# específicas y auto-contenidas, como el seguimiento de la mano,
# el predictor de palabras y la gestión de botones del teclado.

class HandTracker:
    """
    Encapsula la lógica de OpenCV y MediaPipe para el seguimiento de la mano.
    Se encarga de inicializar la cámara, procesar los fotogramas para detectar manos
    y extraer la posición del cursor de la mano y el estado de "clic".
    """
    def __init__(self, camera_index=0):
        """
        Inicializa el rastreador de manos.
        :param camera_index: Índice de la cámara a usar (0 es la predeterminada).
        """
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            # Lanza un error si la cámara no se puede abrir, lo que es crítico.
            raise IOError(f"No se puede abrir la cámara con índice {camera_index}")
        
        # Inicializa el modelo de detección de manos de MediaPipe.
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=1,                 # Detectar solo una mano.
            min_detection_confidence=0.7,    # Confianza mínima para la detección inicial.
            min_tracking_confidence=0.5      # Confianza mínima para el seguimiento.
        )
        self.mp_drawing = mp.solutions.drawing_utils # Utilidad para dibujar los landmarks de MediaPipe.
        self.results = None # Almacenará los resultados del procesamiento de la mano.

    def get_frame(self):
        """
        Captura un fotograma de la cámara, lo procesa para detectar la mano
        y devuelve el fotograma con los dibujos, la posición normalizada del cursor
        y si se detecta un "clic".
        :return: Tupla (frame_con_dibujos, posicion_normalizada, esta_clicando).
                 Devuelve (None, None, False) si no se puede leer el fotograma.
        """
        ret, frame = self.cap.read() # Lee un fotograma de la cámara.
        if not ret:
            return None, None, False # Si no hay fotograma, devuelve None.

        frame = cv2.flip(frame, 1) # Voltea el fotograma horizontalmente para simular un espejo.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convierte BGR a RGB para MediaPipe.
        self.results = self.hands.process(rgb_frame) # Procesa el fotograma para encontrar manos.

        normalized_pos, is_clicking = None, False

        if self.results.multi_hand_landmarks:
            # Si se detectaron manos (solo tomamos la primera).
            hand_landmarks = self.results.multi_hand_landmarks[0]
            # Dibuja los landmarks y las conexiones en el fotograma.
            self.mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            
            # Obtiene la posición de la punta del dedo índice.
            index_tip = hand_landmarks.landmark[INDEX_FINGER_TIP]
            normalized_pos = (index_tip.x, index_tip.y) # Posición normalizada (0.0 a 1.0).
            
            # Obtiene la posición de la punta del dedo pulgar.
            thumb_tip = hand_landmarks.landmark[THUMB_TIP]
            # Calcula la distancia euclidiana entre la punta del índice y el pulgar.
            distance = np.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)
            # Si la distancia es menor que el umbral, se considera un "clic".
            if distance < CLICK_THRESHOLD:
                is_clicking = True
            
        return frame, normalized_pos, is_clicking

    def stop(self):
        """
        Libera los recursos de MediaPipe y la cámara al detener la aplicación.
        """
        self.hands.close()
        if self.cap: self.cap.release()

class NeuralWordPredictor:
    """
    *** MEJORA: Predictor de palabras basado en una Red Neuronal LSTM. ***
    Aprende a completar palabras basándose en un corpus de texto.
    Este modelo se entrena al inicio de la aplicación y luego se utiliza
    para sugerir la continuación de la palabra que el usuario está escribiendo.
    """
    def __init__(self, corpus_path):
        """
        Inicializa el predictor neuronal y entrena el modelo.
        :param corpus_path: Ruta al archivo de texto que contiene el corpus.
        """
        print("Inicializando predictor neuronal...")
        self.model = None
        self.char_to_int = {} # Mapeo de caracteres a enteros.
        self.int_to_char = {} # Mapeo de enteros a caracteres.
        self.max_len = 0      # Longitud máxima de secuencia para el padding.
        self.vocab_size = 0   # Tamaño del vocabulario de caracteres.
        self._train_model(corpus_path)

    def _train_model(self, corpus_path):
        """
        Prepara los datos, construye y entrena el modelo LSTM.
        En una aplicación real, este modelo se entrenaría offline y se cargaría.
        Aquí, lo entrenamos al inicio para que el script sea autocontenido.
        """
        try:
            with open(corpus_path, 'r', encoding='utf-8') as f:
                # Lee las palabras del corpus, las convierte a minúsculas y añade un espacio.
                # El espacio final ayuda al modelo a aprender cuándo una palabra termina.
                words = [line.strip().lower() + ' ' for line in f if line.strip()]
        except FileNotFoundError:
            print(f"Advertencia: '{corpus_path}' no encontrado. Usando corpus por defecto para el predictor.")
            # Corpus de emergencia si el archivo no se encuentra.
            words = ["hola ", "mundo ", "ayuda ", "proyecto ", "inteligencia ", "artificial ", "teclado ", "mano "]
        
        text = "".join(words) # Concatena todas las palabras en una sola cadena.
        chars = sorted(list(set(text))) # Obtiene todos los caracteres únicos y los ordena.
        self.vocab_size = len(chars) # Tamaño del vocabulario de caracteres.
        self.char_to_int = {c: i for i, c in enumerate(chars)} # Crea el mapeo char -> int.
        self.int_to_char = {i: c for i, c in enumerate(chars)} # Crea el mapeo int -> char.
        
        sequences = []
        # Crea secuencias de entrada/salida para el entrenamiento.
        # Por ejemplo, para "hola ": "h" -> "o", "ho" -> "l", "hol" -> "a", "hola" -> " "
        for word in words:
            for i in range(1, len(word)):
                sequences.append(word[:i+1])
        
        self.max_len = max([len(seq) for seq in sequences]) # Encuentra la longitud de secuencia más larga.
        
        # Codifica las secuencias a números enteros y aplica padding.
        encoded_sequences = [[self.char_to_int[char] for char in seq] for seq in sequences]
        padded_sequences = pad_sequences(encoded_sequences, maxlen=self.max_len, padding='pre')
        
        # Divide los datos en entradas (X) y salidas (y).
        # X son todas las columnas excepto la última, y es la última columna (el siguiente caracter).
        X, y = padded_sequences[:, :-1], padded_sequences[:, -1]
        # Convierte 'y' a formato one-hot encoding para la clasificación multiclase.
        y = to_categorical(y, num_classes=self.vocab_size)
        
        print("Entrenando modelo de predicción de palabras (esto puede tardar unos segundos)...")
        # Definición del modelo LSTM (Red Neuronal Recurrente).
        self.model = Sequential([
            # Capa de Embedding: Convierte los enteros a vectores densos de tamaño fijo.
            Embedding(self.vocab_size, 50, input_length=self.max_len-1),
            # Capa LSTM: Procesa secuencias y aprende dependencias a largo plazo.
            LSTM(100),
            # Capa Densa (salida): Produce probabilidades para cada carácter en el vocabulario.
            Dense(self.vocab_size, activation='softmax')
        ])
        # Compila el modelo con una función de pérdida y optimizador adecuados para clasificación.
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # Entrena el modelo. verbose=0 suprime la salida detallada del entrenamiento.
        self.model.fit(X, y, epochs=20, verbose=0)
        print("Modelo de predicción entrenado y listo.")

    def predict(self, prefix, max_sugg=1):
        """
        Genera una compleción de palabra para el prefijo dado utilizando el modelo LSTM.
        :param prefix: La parte de la palabra que el usuario ha escrito.
        :param max_sugg: Número máximo de sugerencias (actualmente limitado a 1).
        :return: Una lista de palabras sugeridas.
        """
        if not self.model or not prefix:
            return [] # No hay modelo o prefijo, no hay sugerencias.
            
        original_prefix = prefix.lower() # Trabaja con el prefijo en minúsculas.
        
        # Itera para predecir los siguientes caracteres hasta que la palabra se complete
        # o se alcance la longitud máxima permitida por el modelo.
        for _ in range(self.max_len - len(original_prefix)):
            # Codifica el prefijo actual a enteros, filtrando caracteres desconocidos.
            encoded = [self.char_to_int[char] for char in original_prefix if char in self.char_to_int]
            if not encoded: # Si el prefijo no contiene caracteres conocidos, sale.
                break
            # Aplica padding al prefijo codificado para que tenga la longitud requerida por el modelo.
            padded = pad_sequences([encoded], maxlen=self.max_len-1, padding='pre')
            
            # Realiza la predicción.
            pred_probs = self.model.predict(padded, verbose=0)[0]
            pred_index = np.argmax(pred_probs) # Obtiene el índice del carácter con mayor probabilidad.
            
            # Obtiene el siguiente carácter predicho.
            next_char = self.int_to_char.get(pred_index, '')
            
            if next_char == ' ': # Si predice un espacio, la palabra se considera terminada.
                break
            original_prefix += next_char # Añade el carácter predicho al prefijo.
        
        # Devuelve la palabra completada solo si es más larga que el prefijo original.
        # Esto evita sugerir la misma palabra si ya está completa.
        return [original_prefix] if len(original_prefix) > len(prefix.lower()) else []


class Button:
    """
    Representa un botón interactivo en la interfaz gráfica (tecla del teclado o sugerencia).
    Maneja su posición, texto, valor asociado, estado de hover y dibujo.
    """
    def __init__(self, rect, text, value, font, is_suggestion=False, row_index=None):
        """
        Inicializa un botón.
        :param rect: Un objeto pygame.Rect que define la posición y tamaño del botón.
        :param text: El texto a mostrar en el botón.
        :param value: El valor que el botón representa (ej. 'A', 'ESPACIO').
        :param font: La fuente de pygame a usar para el texto del botón.
        :param is_suggestion: True si este botón es una sugerencia de palabra.
        :param row_index: Índice de la fila del teclado a la que pertenece (para colores).
        """
        self.rect = pygame.Rect(rect)
        self.text = text
        self.value = value
        self.font = font
        self.is_hovered = False      # Estado de si el cursor de la mano está sobre el botón.
        self.is_suggestion = is_suggestion # Si es un botón de sugerencia.
        self.row_index = row_index     # Para asignar colores según la fila.

    def draw(self, screen):
        """
        Dibuja el botón en la pantalla.
        :param screen: La superficie de la pantalla de Pygame.
        """
        text_color = COLOR_TEXT_ON_DARK
        is_special = self.value in ['MAYÚS', '←', '␣'] # Define si es una tecla especial.

        # Determina el color de fondo del botón.
        if self.is_hovered:
            bg_color = COLOR_KEY_HOVER # Color cuando el cursor está encima.
        elif is_special:
            bg_color = COLOR_KEY_SPECIAL_NEW # Color para teclas especiales.
            text_color = COLOR_TEXT_ON_LIGHT # Cambia el color del texto para teclas especiales.
        elif self.is_suggestion:
            bg_color = COLOR_SUGGESTION_DEFAULT # Color para botones de sugerencia.
        else:
            # Colores basados en la fila para las teclas normales.
            if self.row_index in [0, 1]: bg_color = COLOR_KEY_ROW_NUM
            elif self.row_index == 2: bg_color = COLOR_KEY_ROW_A
            elif self.row_index == 3: bg_color = COLOR_KEY_ROW_Z
            else: bg_color = (0,0,0) # Color por defecto si no hay fila definida.
        
        # Dibuja el rectángulo del botón con bordes redondeados.
        pygame.draw.rect(screen, bg_color, self.rect, border_radius=12)
        
        # Dibuja un borde alrededor del botón si no está en estado de hover.
        if not self.is_hovered:
            border_color = (120, 120, 120) if is_special else (20, 20, 20)
            pygame.draw.rect(screen, border_color, self.rect, 2, border_radius=12)

        # Renderiza y dibuja el texto en el centro del botón.
        text_surf = self.font.render(self.text, True, text_color)
        screen.blit(text_surf, text_surf.get_rect(center=self.rect.center))

class VirtualKeyboard:
    """
    Gestiona la disposición y el comportamiento del teclado virtual,
    incluyendo la creación de teclas y la actualización de sugerencias de palabras.
    """
    def __init__(self, predictor, w, h):
        """
        Inicializa el teclado virtual.
        :param predictor: Instancia de NeuralWordPredictor para las sugerencias.
        :param w: Ancho de la pantalla.
        :param h: Alto de la pantalla.
        """
        self.predictor = predictor
        self.w, self.h = w, h
        self.key_buttons = []       # Lista para almacenar los objetos Button de las teclas.
        self.suggestion_buttons = [] # Lista para almacenar los objetos Button de las sugerencias.
        # Fuentes para las teclas y sugerencias.
        self.font_keys = pygame.font.SysFont("Arial", FONT_SIZE_KEYS, bold=True)
        self.font_sugg = pygame.font.SysFont("Arial", FONT_SIZE_TEXT - 4, bold=True)
        self._build_keyboard() # Construye la disposición inicial del teclado.

    def _build_keyboard(self):
        """
        Construye dinámicamente los botones del teclado basándose en KEY_LAYOUT
        y las dimensiones de la pantalla.
        """
        self.key_buttons.clear() # Limpia cualquier botón existente.
        keyboard_width = self.w * 0.70 # El teclado ocupa el 70% del ancho de la pantalla.
        num_keys_longest_row = 10 # Número de teclas en la fila más larga.
        margin_ratio = 0.15 # Relación del margen respecto al tamaño de la tecla.
        # Calcula el tamaño base de una tecla para que el teclado encaje bien.
        key_size = keyboard_width / (num_keys_longest_row + (num_keys_longest_row - 1) * margin_ratio)
        key_margin = key_size * margin_ratio # Margen entre teclas.
        key_height = key_size * 0.9 # Altura de la tecla, ligeramente menor que su ancho.
        
        # Calcula la posición inicial Y para el teclado, dejando espacio abajo.
        total_keyboard_height = len(KEY_LAYOUT) * (key_height + key_margin)
        start_y = self.h - total_keyboard_height - 20
        y = start_y

        # Itera sobre cada fila definida en KEY_LAYOUT para crear los botones.
        for i, row in enumerate(KEY_LAYOUT):
            # Factores de multiplicación para el ancho de teclas especiales.
            spacebar_mult = 4; backspace_mult = 1.5; caps_mult = 1.5
            
            # Calcula el ancho total de la fila para centrarla.
            row_width = 0
            for char in row:
                if char == '␣': row_width += key_size * spacebar_mult
                elif char == '←': row_width += key_size * backspace_mult
                elif char == 'MAYÚS': row_width += key_size * caps_mult
                else: row_width += key_size
            row_width += (len(row) - 1) * key_margin
            x = (self.w - row_width) / 2 # Posición X de inicio para centrar la fila.
            
            # Crea los botones individuales para cada carácter en la fila.
            for char in row:
                w = key_size # Ancho predeterminado de la tecla.
                # Ajusta el ancho para teclas especiales.
                if char == "␣": w *= spacebar_mult
                elif char == "←": w *= backspace_mult
                elif char == "MAYÚS": w *= caps_mult
                
                # Asigna una etiqueta más descriptiva a las teclas especiales.
                label = "ESPACIO" if char == "␣" else ("BORRAR" if char == "←" else char)
                # Usa una fuente más pequeña para etiquetas largas.
                font = self.font_sugg if len(label) > 1 else self.font_keys
                
                # Crea el objeto Button y lo añade a la lista.
                button = Button((x, y, w, key_height), label, char, font, row_index=i)
                self.key_buttons.append(button)
                x += w + key_margin # Mueve la posición X para la siguiente tecla.
            y += key_height + key_margin # Mueve la posición Y para la siguiente fila.
            
    def update_suggestions(self, text):
        """
        Actualiza los botones de sugerencia de palabras basándose en el texto actual.
        :param text: El texto que el usuario ha escrito hasta ahora.
        """
        self.suggestion_buttons.clear() # Limpia las sugerencias anteriores.
        # Obtiene la última palabra no terminada para predecir.
        last_word = text.split(" ")[-1] if text and not text.endswith(" ") else ""
        if not last_word: return # No hay palabra para sugerir si está vacía o termina en espacio.
        
        suggestions = self.predictor.predict(last_word) # Obtiene sugerencias del predictor.
        if not suggestions: return # Si no hay sugerencias, termina.

        # Calcula la posición y tamaño de los botones de sugerencia.
        sugg_w, sugg_h, margin = 250, 50, 15
        x_start = (self.w - sugg_w) / 2 # Centra el botón de sugerencia.
        
        # Posiciona la sugerencia encima de la primera fila de teclas.
        first_key_y = self.key_buttons[0].rect.top if self.key_buttons else self.h
        y_pos = first_key_y - sugg_h - 20
        
        # Solo se espera una sugerencia (la palabra completada)
        sug = suggestions[0]
        # Crea el botón de sugerencia.
        btn = Button((x_start, y_pos, sugg_w, sugg_h), sug.upper(), sug, self.font_sugg, is_suggestion=True)
        self.suggestion_buttons.append(btn)

    def draw(self, screen):
        """
        Dibuja todas las teclas y botones de sugerencia en la pantalla.
        :param screen: La superficie de la pantalla de Pygame.
        """
        for btn in self.key_buttons + self.suggestion_buttons:
            btn.draw(screen)
            
    def check_hover(self, pos):
        """
        Verifica si el cursor de la mano está sobre algún botón y actualiza su estado.
        :param pos: La posición (x, y) del cursor de la mano en la pantalla.
        :return: El objeto Button sobre el que está el cursor, o None si no hay ninguno.
        """
        all_buttons = self.key_buttons + self.suggestion_buttons
        hovered_button = None
        for btn in all_buttons:
            if pos and btn.rect.collidepoint(pos):
                btn.is_hovered = True
                hovered_button = btn
            else:
                btn.is_hovered = False # Si no está sobre este botón, desactiva el hover.
        return hovered_button

# ==============================================================================
# SECCIÓN 3: CLASE PRINCIPAL DE LA APLICACIÓN
# ==============================================================================
# La clase MainApp es el corazón de la aplicación.
# Se encarga de la inicialización de Pygame, la gestión del bucle principal,
# la interacción entre los módulos (rastreador de mano, teclado, predictor)
# y el dibujo de la interfaz de usuario.

class MainApp:
    """
    Clase principal que orquesta todos los componentes de la aplicación
    de teclado virtual controlada por la mano.
    """
    def __init__(self):
        """
        Inicializa Pygame, los módulos de seguimiento y predicción,
        y configura el estado inicial de la aplicación.
        """
        pygame.init() # Inicializa todos los módulos de Pygame.
        pygame.mixer.init() # Inicializa el mezclador de sonido.
        
        global SCREEN_WIDTH, SCREEN_HEIGHT # Permite modificar las variables globales.
        if USE_FULLSCREEN:
            info = pygame.display.Info() # Obtiene información de la pantalla actual.
            SCREEN_WIDTH, SCREEN_HEIGHT = info.current_w, info.current_h # Usa la resolución máxima.
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.NOFRAME) # Sin bordes.
        else:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE) # Redimensionable.
            
        pygame.display.set_caption("Teclado con Mano y IA") # Establece el título de la ventana.
        self.clock = pygame.time.Clock() # Objeto para controlar la velocidad de fotogramas.

        # Inicializa las fuentes a usar en la interfaz.
        self.font_status = pygame.font.SysFont("Arial", FONT_SIZE_STATUS, bold=True)
        self.font_text_area = pygame.font.SysFont("Arial", FONT_SIZE_TEXT)
        self.font_calibration = pygame.font.SysFont("Arial", FONT_SIZE_CALIBRATION, bold=True)

        self.hand_tracker = HandTracker() # Crea una instancia del rastreador de manos.
        self.predictor = NeuralWordPredictor(CORPUS_FILE) # *** MEJORA: Inicializa el predictor neuronal ***
        self.keyboard = VirtualKeyboard(self.predictor, SCREEN_WIDTH, SCREEN_HEIGHT) # Crea el teclado virtual.

        self.typed_text = "" # Cadena que almacena el texto que el usuario ha escrito.
        self.hand_cursor_pos = None # Posición suavizada del cursor de la mano en pantalla.
        self.is_caps = False # Estado de la tecla MAYÚS (activado/desactivado).
        self.status_text = "Iniciando calibración..." # Mensaje de estado inicial.
        self.last_click_time = 0 # Marca de tiempo del último clic para el cooldown.
        self.state = "CALIBRATING_START" # Estado inicial de la aplicación.
        # Límites de calibración (valores normalizados de 0.0 a 1.0).
        self.calibration_bounds = {'min_x': 1.0, 'max_x': 0.0, 'min_y': 1.0, 'max_y': 0.0}
        self.calibration_samples = [] # Muestras de posiciones de mano durante la calibración.

        try:
            # Intenta cargar el sonido de clic.
            self.click_sound = pygame.mixer.Sound(SOUND_FILE_CLICK)
        except (pygame.error, FileNotFoundError):
            self.click_sound = None
            print(f"Advertencia: No se pudo cargar '{SOUND_FILE_CLICK}'. El sonido de clic estará desactivado.")

    def run(self):
        """
        Bucle principal de la aplicación. Captura eventos, actualiza el estado
        y dibuja la interfaz continuamente.
        """
        while True:
            self.handle_events() # Maneja los eventos de Pygame (cerrar, redimensionar).
            
            # Obtiene el fotograma de la cámara, la posición de la mano y el estado de clic.
            frame, normalized_pos, is_clicking = self.hand_tracker.get_frame()
            if frame is None: 
                self.status_text = "Error: No se puede acceder a la cámara."
            
            # Lógica diferente según el estado de la aplicación.
            if "CALIBRATING" in self.state:
                self.handle_calibration(normalized_pos, is_clicking) # Proceso de calibración.
            else:
                self.handle_typing(normalized_pos, is_clicking) # Proceso de escritura.

            self.draw(frame) # Dibuja todos los elementos en la pantalla.

    def handle_events(self):
        """
        Procesa los eventos de Pygame, como cerrar la ventana,
        presionar Escape o redimensionar la ventana.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.stop() # Cierra la aplicación.
            if event.type == pygame.VIDEORESIZE and not USE_FULLSCREEN:
                # Si la ventana se redimensiona (y no está en pantalla completa), ajusta el tamaño.
                global SCREEN_WIDTH, SCREEN_HEIGHT
                SCREEN_WIDTH, SCREEN_HEIGHT = event.w, event.h
                self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
                # Reconstruye el teclado para que se adapte al nuevo tamaño.
                self.keyboard = VirtualKeyboard(self.predictor, SCREEN_WIDTH, SCREEN_HEIGHT)
                self.state = "CALIBRATING_START" # Reinicia la calibración al redimensionar.

    def _map_hand_to_screen(self, normalized_pos):
        """
        Mapea las coordenadas normalizadas (0.0-1.0) de la mano a las coordenadas
        de píxeles de la pantalla, usando los límites de calibración.
        :param normalized_pos: Tupla (x, y) normalizada de la posición de la mano.
        :return: Tupla (x, y) en píxeles de la pantalla, o None si no hay posición o calibración inválida.
        """
        if not normalized_pos: return None
        # Calcula el rango del movimiento de la mano basado en la calibración.
        x_range = self.calibration_bounds['max_x'] - self.calibration_bounds['min_x']
        y_range = self.calibration_bounds['max_y'] - self.calibration_bounds['min_y']
        
        # Evita división por cero o rangos extremadamente pequeños.
        if x_range < 1e-6 or y_range < 1e-6: return None
        
        # Mapea la posición normalizada a la pantalla.
        # Se escala y se desplaza para ajustarse al área calibrada.
        mapped_x = ((normalized_pos[0] - self.calibration_bounds['min_x']) / x_range) * SCREEN_WIDTH
        mapped_y = ((normalized_pos[1] - self.calibration_bounds['min_y']) / y_range) * SCREEN_HEIGHT
        return (mapped_x, mapped_y)

    def handle_calibration(self, normalized_pos, is_clicking):
        """
        Gestiona el proceso de calibración de la mano.
        Pide al usuario que apunte a esquinas para definir el área de movimiento.
        :param normalized_pos: Posición actual normalizada de la mano.
        :param is_clicking: Estado de clic actual de la mano.
        """
        now = pygame.time.get_ticks() # Tiempo actual en milisegundos.

        if self.state == "CALIBRATING_START":
            self.state = "CALIBRATING_TOP_LEFT" # Pasa al primer paso de calibración.
            # Reinicia los límites y las muestras de calibración.
            self.calibration_bounds = {'min_x': 1.0, 'max_x': 0.0, 'min_y': 1.0, 'max_y': 0.0}
            self.calibration_samples = []
            return

        if not normalized_pos:
            self.status_text = "Mano no detectada. Asegúrese de que sea visible."
            return
        
        self.calibration_samples.append(normalized_pos) # Añade la posición actual a las muestras.
        
        # Si se está clicando y ha pasado suficiente tiempo desde el último "clic" de calibración.
        if is_clicking and (now - self.last_click_time > CLICK_COOLDOWN_MS * 2):
            self.last_click_time = now
            # Calcula el promedio de las últimas 10 muestras para mayor estabilidad.
            avg_pos = np.mean(self.calibration_samples[-10:], axis=0) if len(self.calibration_samples) > 10 else normalized_pos
                
            if self.state == "CALIBRATING_TOP_LEFT":
                # Guarda la posición promedio como la esquina superior izquierda.
                self.calibration_bounds['min_x'], self.calibration_bounds['min_y'] = avg_pos[0], avg_pos[1]
                self.state = "CALIBRATING_BOTTOM_RIGHT" # Pasa al siguiente paso.
                self.calibration_samples = [] # Reinicia las muestras para la siguiente calibración.
            elif self.state == "CALIBRATING_BOTTOM_RIGHT":
                # Guarda la posición promedio como la esquina inferior derecha.
                self.calibration_bounds['max_x'], self.calibration_bounds['max_y'] = avg_pos[0], avg_pos[1]
                # Aplica un padding para asegurar que el movimiento no se restrinja demasiado.
                self.calibration_bounds['min_x'] -= CALIBRATION_PADDING
                self.calibration_bounds['max_x'] += CALIBRATION_PADDING
                self.calibration_bounds['min_y'] -= CALIBRATION_PADDING
                self.calibration_bounds['max_y'] += CALIBRATION_PADDING
                self.state = "TYPING" # La calibración ha terminado, pasa al modo de escritura.
                self.status_text = "¡Calibración completa! Puede empezar a escribir."

    def _perform_click(self, button):
        """
        Realiza la acción asociada a un botón clicado.
        :param button: El objeto Button que ha sido clicado.
        """
        if button is None: return
        if self.click_sound: self.click_sound.play() # Reproduce el sonido de clic.
        value = button.value # Obtiene el valor asociado al botón.
        
        if value == "MAYÚS":
            self.is_caps = not self.is_caps # Alterna el estado de mayúsculas/minúsculas.
        elif button.is_suggestion:
            # Si es una sugerencia, reemplaza la última palabra escrita con la sugerencia.
            words = self.typed_text.split(' ')
            words[-1] = value
            self.typed_text = ' '.join(words) + ' ' # Añade un espacio después de la palabra completada.
        elif value == '←':
            self.typed_text = self.typed_text[:-1] # Elimina el último carácter (Borrar).
        elif value == '␣':
            self.typed_text += ' ' # Añade un espacio.
        else:
            # Añade el carácter al texto escrito, respetando el estado de mayúsculas.
            self.typed_text += str(value) if self.is_caps else str(value).lower()
        
        # Actualiza las sugerencias después de cada entrada.
        self.keyboard.update_suggestions(self.typed_text)

    def handle_typing(self, normalized_pos, is_clicking):
        """
        Gestiona el modo de escritura: mueve el cursor y procesa los clics en las teclas.
        :param normalized_pos: Posición actual normalizada de la mano.
        :param is_clicking: Estado de clic actual de la mano.
        """
        # Mapea la posición normalizada de la mano a las coordenadas de la pantalla.
        screen_pos = self._map_hand_to_screen(normalized_pos)
        
        if screen_pos:
            if self.hand_cursor_pos is None:
                self.hand_cursor_pos = screen_pos # Inicializa el cursor si es la primera vez.
            else:
                # Aplica suavizado al movimiento del cursor para una experiencia más fluida.
                px, py = screen_pos; gx, gy = self.hand_cursor_pos
                self.hand_cursor_pos = (SMOOTHING_FACTOR * px + (1 - SMOOTHING_FACTOR) * gx,
                                        SMOOTHING_FACTOR * py + (1 - SMOOTHING_FACTOR) * gy)
        else:
            self.hand_cursor_pos = None # Si no se detecta la mano, el cursor desaparece.

        # Verifica si el cursor está sobre algún botón y actualiza su estado de hover.
        hovered_button = self.keyboard.check_hover(self.hand_cursor_pos)
        now = pygame.time.get_ticks()

        # Si se está clicando, hay un botón en hover y ha pasado el tiempo de cooldown.
        if is_clicking and hovered_button and (now - self.last_click_time > CLICK_COOLDOWN_MS):
            self._perform_click(hovered_button) # Ejecuta la acción del botón.
            self.last_click_time = now # Actualiza el tiempo del último clic.

    def draw(self, frame):
        """
        Dibuja todos los elementos gráficos de la interfaz en la pantalla.
        :param frame: El fotograma actual de la cámara para mostrar.
        """
        self.screen.fill(COLOR_BACKGROUND) # Rellena el fondo con el color predeterminado.

        if frame is not None:
            # Dibuja el fondo del recuadro de la cámara.
            cam_bg_rect = pygame.Rect(20, 20, VIDEO_FEED_WIDTH, VIDEO_FEED_HEIGHT)
            pygame.draw.rect(self.screen, COLOR_CAM_BG, cam_bg_rect, border_radius=12)
            
            # Convierte y escala el fotograma de OpenCV para Pygame.
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = np.transpose(frame_rgb, (1, 0, 2)) # Transpone para Pygame.
            frame_surface = pygame.surfarray.make_surface(frame_rgb)
            scaled_frame = pygame.transform.scale(frame_surface, (VIDEO_FEED_WIDTH, VIDEO_FEED_HEIGHT))
            self.screen.blit(scaled_frame, cam_bg_rect.topleft) # Dibuja el fotograma de la cámara.
            
        if "CALIBRATING" in self.state:
            # Dibuja mensajes y objetivos específicos para el modo de calibración.
            prompt_text, target_pos = "", None
            if self.state == "CALIBRATING_TOP_LEFT":
                prompt_text, target_pos = "Apunte a la esquina superior izquierda y pellizque para confirmar", (100, 100)
            elif self.state == "CALIBRATING_BOTTOM_RIGHT":
                prompt_text, target_pos = "Ahora apunte a la esquina inferior derecha y pellizque", (SCREEN_WIDTH - 100, SCREEN_HEIGHT - 100)
            
            if target_pos:
                # Dibuja el círculo objetivo y su borde.
                pygame.draw.circle(self.screen, COLOR_CALIBRATION_TARGET, target_pos, 40)
                pygame.draw.circle(self.screen, COLOR_BACKGROUND, target_pos, 43, 4)
            
            # Dibuja el texto del mensaje de calibración centrado.
            prompt_surf = self.font_calibration.render(prompt_text, True, COLOR_TEXT_ON_LIGHT)
            self.screen.blit(prompt_surf, prompt_surf.get_rect(center=self.screen.get_rect().center))
        else:
            # Dibuja elementos para el modo de escritura normal.
            # Dibuja el mensaje de estado en la parte superior central.
            status_surf = self.font_status.render(self.status_text, True, COLOR_STATUS_TEXT)
            self.screen.blit(status_surf, status_surf.get_rect(centerx=self.screen.get_rect().centerx, y=20))
            
            # Calcula la posición del área de texto.
            first_key_y = self.keyboard.key_buttons[0].rect.top if self.keyboard.key_buttons else self.h
            text_area_y = first_key_y - (50 + 20) - 60
            if self.keyboard.suggestion_buttons: # Ajusta si hay sugerencias.
                text_area_y = self.keyboard.suggestion_buttons[0].rect.top - 60
            
            # Dibuja el rectángulo de fondo para el área de texto.
            text_area_rect = pygame.Rect(40, text_area_y, SCREEN_WIDTH - 80, 50)
            pygame.draw.rect(self.screen, COLOR_TEXT_AREA_BG, text_area_rect, border_radius=12)
            
            # Agrega un cursor parpadeante al texto escrito.
            cursor_char = "|" if int(time.time() * 2) % 2 == 0 else ""
            display_text = self.typed_text + cursor_char
            
            # Dibuja el texto escrito dentro del área de texto.
            text_surf = self.font_text_area.render(display_text, True, COLOR_TEXT_ON_LIGHT)
            self.screen.blit(text_surf, (text_area_rect.x + 15, text_area_rect.y + 10))
            
            self.keyboard.draw(self.screen) # Dibuja el teclado y las sugerencias.
            
            if self.hand_cursor_pos:
                # Dibuja el cursor de la mano si está visible.
                pygame.draw.circle(self.screen, COLOR_HAND_CURSOR, (int(self.hand_cursor_pos[0]), int(self.hand_cursor_pos[1])), 12, 3)
                pygame.draw.circle(self.screen, COLOR_BACKGROUND, (int(self.hand_cursor_pos[0]), int(self.hand_cursor_pos[1])), 14, 1)

        pygame.display.flip() # Actualiza toda la pantalla.
        self.clock.tick(FPS) # Limita los fotogramas por segundo.

    def stop(self):
        """
        Detiene todos los módulos y cierra Pygame, terminando la aplicación.
        """
        print("Cerrando la aplicación...")
        self.hand_tracker.stop() # Detiene el rastreador de manos y libera la cámara.
        pygame.quit() # Desinicializa Pygame.
        sys.exit() # Sale del programa.

# ==============================================================================
# SECCIÓN 4: PUNTO de ENTRADA
# ==============================================================================
# Esta sección es el punto de inicio de la ejecución del script.
# Se encarga de la configuración inicial de archivos y del manejo de errores
# a nivel de aplicación.

if __name__ == '__main__':
    # Verifica si el archivo de sonido existe y notifica si no.
    if not os.path.exists(SOUND_FILE_CLICK):
        print(f"Nota: Archivo de sonido '{SOUND_FILE_CLICK}' no encontrado.")
    
    # Verifica si el archivo del corpus existe. Si no, crea uno básico.
    if not os.path.exists(CORPUS_FILE):
        print(f"Nota: Archivo de corpus '{CORPUS_FILE}' no encontrado. Creando uno básico.")
        with open(CORPUS_FILE, 'w', encoding='utf-8') as f:
            f.write("hola\nmundo\nproyecto\nayuda\ninteligencia\nartificial\nteclado\nmano\nsistema\ncomputadora\n")
    
    try:
        app = MainApp() # Crea una instancia de la aplicación principal.
        app.run() # Ejecuta el bucle principal de la aplicación.
    except Exception as e:
        # Captura cualquier excepción no manejada para un cierre más elegante
        # y proporciona información de depuración.
        print(f"Ha ocurrido un error fatal: {e}")
        import traceback
        traceback.print_exc() # Imprime el rastreo completo del error.
    finally:
        # Asegura que Pygame se cierre correctamente, incluso si hay un error.
        pygame.quit()
        sys.exit()