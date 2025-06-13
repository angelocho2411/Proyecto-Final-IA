Teclado Virtual de Mano con Predicción Neuronal
Este proyecto experimental implementa un teclado virtual completamente funcional que puede ser controlado únicamente con el dedo índice, detectado a través de la cámara web. Utiliza visión por computadora (MediaPipe) para el seguimiento de la mano y una Red Neuronal LSTM (Long Short-Term Memory) para la predicción y autocompletado dinámico de palabras. Esta interfaz alternativa elimina la necesidad de dispositivos físicos, abriendo posibilidades para entornos accesibles o sin contacto.

### Características Principales:
Control Intuitivo: Escritura de texto completa moviendo el dedo índice y realizando un gesto de "pellizco" (acercando el pulgar al índice) para hacer clic.

- Predicción Inteligente: Autocompletado dinámico de palabras comunes gracias a un modelo de Red Neuronal LSTM entrenado con un corpus de texto.
- Funcionalidades Esenciales: Incluye funciones de borrado (carácter por carácter) y espacio.
- Interfaz Adaptada: Teclado QWERTY virtual con un diseño claro y responsivo.
- Seguimiento Preciso: Detección de mano en tiempo real y seguimiento de puntos clave mediante la librería MediaPipe.
- Calibración Interactiva: Un proceso de calibración guiado al inicio o al redimensionar la ventana para adaptar el área de control a tu entorno.

### Casos de Uso
- Accesibilidad: Proporciona una interfaz de entrada alternativa para personas con movilidad reducida en sus extremidades superiores.
- Interacción Sin Contacto: Ideal para entornos donde la higiene es crucial, como clínicas, laboratorios o puntos de información pública.
- Proyectos Educativos: Excelente ejemplo de aplicación práctica de visión por computadora, aprendizaje automático e interfaces hombre-máquina.
- Prototipado: Base para el desarrollo de sistemas de accesibilidad avanzados o experiencias de usuario innovadoras.

### Requisitos
Antes de comenzar, asegúrate de tener instalados los siguientes requisitos:
- Python: Versión 3.8 o superior.
- Cámara Web: Una cámara web activa con buena iluminación para una detección precisa de la mano.

## Instalación
Sigue estos pasos para configurar y ejecutar el proyecto en tu máquina:

### Clonar el Repositorio
Abre tu terminal o línea de comandos y ejecuta:

cd teclado_virtual_mano


### Instalar Dependencias
Es altamente recomendable crear un entorno virtual para el proyecto.

python -m venv venv
.\venv\Scripts\activate

Una vez activado el entorno virtual, instala las librerías necesarias:

pip install pygame opencv-python mediapipe numpy tensorflow

- Configurar el Archivo de Corpus (corpus.txt)
- El modelo de predicción de palabras se entrena utilizando un archivo llamado corpus.txt. Este archivo debe contener una lista de palabras (una por línea) que el modelo aprenderá a predecir.


### Ejecutar la Aplicación Principal
Abre tu terminal en el directorio del proyecto (con el entorno virtual activado).

### Ejecuta el script principal:

python main.py

### Proceso de Calibración al Inicio:
Al ejecutar main.py, la aplicación comenzará en modo de calibración para adaptar el seguimiento de tu mano a tu entorno:

Esquina Superior Izquierda: Se te pedirá que muevas tu dedo índice a la posición que desees que sea la esquina superior izquierda de tu área de control en la cámara. Una vez en posición, realiza un gesto de "pellizco" (junta el pulgar y el índice) para registrar este punto.

Esquina Inferior Derecha: Después de registrar la primera esquina, se te indicará que muevas tu dedo índice a la posición que será la esquina inferior derecha de tu área de control. De nuevo, realiza un gesto de "pellizco" para confirmar.

Una vez completada la calibración, la aplicación pasará automáticamente al modo de escritura, y podrás controlar el teclado con tu dedo índice.
