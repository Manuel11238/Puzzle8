
import cv2
import numpy as np
import os
import shutil
import fiftyone as fo
import fiftyone.utils.random as four
import albumentations as A
from PIL import Image 

# Importa configuración desde config.py
from config import (
    IMG_HEIGHT, IMG_WIDTH, EMOTION_LABELS, EMOTION_LABEL_MAP,
    DATASET_ROOT, ORIGINAL_TRAIN_DIR, ORIGINAL_TEST_DIR,
    AUGMENTED_OUTPUT_DIR, AUGMENTATION_FACTOR, NUM_VIZ_AUGMENTED_SAMPLES,
    FIFTYONE_APP_PORT_ORIGINAL, FIFTYONE_APP_PORT_AUGMENTED
)

# --- 0. Configuración inicial ---
print("--- Paso 1: Preparación del dataset y aumento de datos ---")

# Elimina directorios de datos aumentados previos si existen
if os.path.exists(AUGMENTED_OUTPUT_DIR):
    shutil.rmtree(AUGMENTED_OUTPUT_DIR)
    print(f"Directorio existente '{AUGMENTED_OUTPUT_DIR}' eliminado.")

# Crea subcarpetas para imágenes aumentadas según la emoción
for emotion in EMOTION_LABELS:
    os.makedirs(os.path.join(AUGMENTED_OUTPUT_DIR, emotion), exist_ok=True)
print(f"Directorios para imágenes aumentadas creados en '{AUGMENTED_OUTPUT_DIR}'.")

print("Verificando directorios de datos originales...")
if not os.path.exists(ORIGINAL_TRAIN_DIR):
    print(f"Error: Carpeta de entrenamiento original '{ORIGINAL_TRAIN_DIR}' no existe. Ajusta DATASET_ROOT en config.py.")
    exit()
if not os.path.exists(ORIGINAL_TEST_DIR):
    print(f"Error: Carpeta de test/validación original '{ORIGINAL_TEST_DIR}' no existe. Ajusta DATASET_ROOT en config.py.")
    exit()
print("Directorios de datos originales verificados.")


# --- 1. Carga el dataset original con FiftyOne ---
print("\nCargando el dataset original con FiftyOne...")

def load_dataset_fo(path, split_name):
    dataset_name = f"fer2013_{split_name}"
    if dataset_name in fo.list_datasets():
        fo.delete_dataset(dataset_name)
    dataset = fo.Dataset.from_dir(
        dataset_dir=path,
        dataset_type=fo.types.ImageClassificationDirectoryTree,
        name=dataset_name,
        tags=[split_name],
        overwrite=True
    )
    # Asigna las clases por defecto para visualización
    dataset.default_classes = list(EMOTION_LABEL_MAP.values())
    dataset.save()
    return dataset

# Elimina el dataset combinado si ya existe
if "fer2013_full_original" in fo.list_datasets():
    fo.delete_dataset("fer2013_full_original")

train_dataset_fo = load_dataset_fo(ORIGINAL_TRAIN_DIR, "train_original")
test_dataset_fo = load_dataset_fo(ORIGINAL_TEST_DIR, "test_original")

full_original_dataset_fo = fo.Dataset(name="fer2013_full_original")
full_original_dataset_fo.add_samples(train_dataset_fo.iter_samples())
full_original_dataset_fo.add_samples(test_dataset_fo.iter_samples())

print(f"Dataset original cargado en FiftyOne. Total de muestras: {len(full_original_dataset_fo)}")
print(f"Muestras de entrenamiento originales: {len(train_dataset_fo)}")
print(f"Muestras de test/validación originales: {len(test_dataset_fo)}")


# --- 2. Visualiza el dataset original con FiftyOne ---
print(f"\nAbriendo FiftyOne para visualizar el dataset original (puerto {FIFTYONE_APP_PORT_ORIGINAL})...")
session_original = fo.launch_app(full_original_dataset_fo, port=FIFTYONE_APP_PORT_ORIGINAL)
print("Revisa la interfaz de FiftyOne (tu navegador, puerto 5151) para visualizar el dataset original.")
input("Presiona Enter para continuar con el aumento de datos y guardado de imágenes...")
session_original.close()


# --- 3. Aumento de datos con Albumentations y guardado a disco ---
print("\nAplicando aumento de datos con Albumentations y guardando imágenes aumentadas...")

# Define el pipeline de transformaciones de Albumentations
# Convertimos a escala de grises *después* de Albumentations y luego redimensionamos.
transform = A.Compose([
    A.ToGray(p=1.0), # Convierte a escala de grises
    A.Resize(IMG_HEIGHT, IMG_WIDTH), # Redimensiona
    A.Rotate(limit=25, p=0.8, border_mode=cv2.BORDER_CONSTANT, value=0), # Rota
    A.RandomScale(scale_limit=0.2, p=0.8), # Escalado aleatorio
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.9), # Brillo/contraste
    A.HorizontalFlip(p=0.5), # Flip horizontal
    A.GaussNoise(p=0.2), # Ruido gaussiano
    A.CoarseDropout(max_holes=8, max_height=8, max_width=8, fill_value=0, p=0.2), # Dropout aleatorio
])

augmented_samples_fo_for_viz = [] # Guarda un subconjunto para visualización en FiftyOne
image_counter = 0

for i, sample in enumerate(train_dataset_fo):
    label = sample.ground_truth.label
    
    # Lee la imagen con OpenCV y convierte a RGB (Albumentations espera RGB)
    img_bgr = cv2.imread(sample.filepath)
    if img_bgr is None:
        print(f"Advertencia: No se pudo cargar la imagen {sample.filepath}. Se omite.")
        continue
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Guarda la imagen original (redimensionada y convertida a escala de grises)
    # Se aplica una transformación mínima para procesamiento consistente antes de guardar
    original_processed = A.Compose([
        A.ToGray(p=1.0),
        A.Resize(IMG_HEIGHT, IMG_WIDTH)
    ])(image=img_rgb)["image"]
    
    original_save_path = os.path.join(AUGMENTED_OUTPUT_DIR, label, f"original_{i:05d}.png")
    cv2.imwrite(original_save_path, original_processed)
    
    # Agrega a la lista de visualización de FiftyOne 
    if len(augmented_samples_fo_for_viz) < NUM_VIZ_AUGMENTED_SAMPLES * (AUGMENTATION_FACTOR + 1) / (AUGMENTATION_FACTOR + 1):
        augmented_samples_fo_for_viz.append(
            fo.Sample(
                filepath=original_save_path,
                ground_truth=fo.Classification(label=label),
                tags=["original_processed"]
            )
        )
    image_counter += 1

    # Aplica transformaciones y guarda nuevas imágenes aumentadas
    for k in range(AUGMENTATION_FACTOR):
        transformed_img = transform(image=img_rgb)["image"]
        
        filename = f"aug_{i:05d}_{k:02d}.png"
        new_path = os.path.join(AUGMENTED_OUTPUT_DIR, label, filename)
        cv2.imwrite(new_path, transformed_img) 

        # Agrega a la lista de visualización de FiftyOne 
        if len(augmented_samples_fo_for_viz) < NUM_VIZ_AUGMENTED_SAMPLES * (AUGMENTATION_FACTOR + 1):
            augmented_samples_fo_for_viz.append(
                fo.Sample(
                    filepath=new_path,
                    ground_truth=fo.Classification(label=label),
                    tags=["augmented"]
                )
            )
        image_counter += 1

print(f"Generadas y guardadas {image_counter} imágenes (originales procesadas + aumentadas) en '{AUGMENTED_OUTPUT_DIR}'.")

# --- 4. Visualiza imágenes aumentadas con FiftyOne ---
if augmented_samples_fo_for_viz:
    # Elimina el dataset temporal si ya existe
    if "temp_generated_dataset" in fo.list_datasets():
        fo.delete_dataset("temp_generated_dataset")
    temp_augmented_dataset = fo.Dataset(name="temp_generated_dataset", persistent=True)
    temp_augmented_dataset.add_samples(augmented_samples_fo_for_viz)
    temp_augmented_dataset.default_classes = list(EMOTION_LABEL_MAP.values())
    temp_augmented_dataset.save()

    # Limita el número de muestras para visualización
    if len(temp_augmented_dataset) > NUM_VIZ_AUGMENTED_SAMPLES:
        # Elimina el dataset de visualización si ya existe
        if "fer2013_augmented_visualization" in fo.list_datasets():
            fo.delete_dataset("fer2013_augmented_visualization")
        viz_samples_view = temp_augmented_dataset.view().shuffle(seed=42).take(NUM_VIZ_AUGMENTED_SAMPLES)
        augmented_dataset_fo = fo.Dataset(name="fer2013_augmented_visualization", persistent=True)
        augmented_dataset_fo.add_samples(viz_samples_view)
        print(f"Cargadas {len(augmented_dataset_fo)} imágenes (subconjunto aleatorio) en FiftyOne para visualización.")
    else:
        augmented_dataset_fo = temp_augmented_dataset
        print(f"Cargadas {len(augmented_samples_fo_for_viz)} imágenes aumentadas en FiftyOne para visualización.")

    print(f"\nAbriendo FiftyOne para visualizar imágenes aumentadas (puerto {FIFTYONE_APP_PORT_AUGMENTED})...")
    session_augmented = fo.launch_app(augmented_dataset_fo, port=FIFTYONE_APP_PORT_AUGMENTED)
    input("Presiona Enter para cerrar la visualización de imágenes aumentadas y terminar este script...")
    session_augmented.close()
else:
    print("No se generaron imágenes aumentadas para visualizar.")

print("\nScript preProcesarDataset.py finalizado.")
