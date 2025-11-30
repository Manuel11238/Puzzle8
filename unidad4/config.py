
import os

# --- Configuración global ---
IMG_HEIGHT, IMG_WIDTH = 48, 48 # Tamaño de imagen FER2013
NUM_CLASSES = 4
EMOTION_LABELS = ['angry', 'happy', 'neutral', 'sad']
# Mapeo de etiquetas enteras (usadas por PyTorch) a etiquetas string
EMOTION_LABEL_MAP = {i: label for i, label in enumerate(EMOTION_LABELS)}
LABEL_TO_INT = {label: i for i, label in enumerate(EMOTION_LABELS)} # Para búsqueda conveniente

# --- Rutas del dataset original ---
# Se debe de ajustar el DATASET_ROOT a donde están las carpetas 'train' y 'test' de FER2013
DATASET_ROOT = r'C:\Users\manue\proyectosPy\fer2013_raw' #lo modifique ./fer2013_raw
ORIGINAL_TRAIN_DIR = os.path.join(DATASET_ROOT, 'train')
ORIGINAL_TEST_DIR = os.path.join(DATASET_ROOT, 'test') # Usado para validación y test final

# --- Rutas de datos procesados y modelo ---
# Directorio donde se guardarán las imágenes aumentadas
AUGMENTED_OUTPUT_DIR = 'fer2013_augmented_data'

# El script de entrenamiento usará imágenes de este directorio
TRAIN_DATA_DIR_FOR_MODEL = AUGMENTED_OUTPUT_DIR
# Validación y test usarán el directorio original de test
VALIDATION_DATA_DIR_FOR_MODEL = ORIGINAL_TEST_DIR
TEST_DATA_DIR_FOR_MODEL = ORIGINAL_TEST_DIR

# --- Configuración de entrenamiento e inferencia del modelo ---
MODEL_SAVE_DIR = 'models' # Carpeta para guardar checkpoints del modelo
MODEL_FILENAME = 'deep_emotion_best_model.pth' # Nombre específico para el mejor modelo
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_FILENAME) # Ruta completa

BATCH_SIZE = 8
LEARNING_RATE = 0.001
NUM_EPOCHS = 30 # O el número que prefieras para entrenar

# Ruta al archivo Haar Cascade XML para detección de rostros
# se puede descárgar de: https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
FACE_CASCADE_PATH = 'haarcascade_frontalface_default.xml'

# --- Configuración de aumento de datos (Albumentations) ---
# Cada imagen original generará AUGMENTATION_FACTOR nuevas copias aumentadas.
# Por cada imagen original, tendrás 1 original procesada + AUGMENTATION_FACTOR aumentadas.
AUGMENTATION_FACTOR = 1
NUM_VIZ_AUGMENTED_SAMPLES = 50 # Número de muestras aumentadas a visualizar en FiftyOne

# --- Configuración de FiftyOne (para visualización) ---
FIFTYONE_APP_PORT_ORIGINAL = 5151
FIFTYONE_APP_PORT_AUGMENTED = 5152
