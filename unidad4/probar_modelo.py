import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Importa configuraciones
from config import (
    BATCH_SIZE, MODEL_SAVE_PATH, EMOTION_LABELS, TEST_DATA_DIR_FOR_MODEL, EMOTION_LABEL_MAP
)

# Importa la definición del modelo CNN
from cnn_modelo import Deep_Emotion

# Importa los data loaders adaptados
from data_loaders import get_dataloaders 

def test_model():
    """
    Evalúa el modelo Deep_Emotion entrenado en el conjunto de test.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")

    # 1. Carga el modelo
    model = Deep_Emotion()
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Error: Pesos del modelo no encontrados en {MODEL_SAVE_PATH}. Entrena el modelo primero.")
        return

    try:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        model.to(device)
        model.eval() # Modo evaluación
        print(f"Modelo cargado correctamente desde {MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"Error cargando el modelo desde {MODEL_SAVE_PATH}: {e}")
        return

    # 2. Carga los datos de test (usando la ruta de validación para consistencia)
    print("\nCargando conjunto de test...")
    _, test_loader = get_dataloaders(
        train_data_dir=TEST_DATA_DIR_FOR_MODEL, 
        val_data_dir=TEST_DATA_DIR_FOR_MODEL, # Este será nuestro test 
        batch_size=BATCH_SIZE
    )
    print(f"Número de muestras de test: {len(test_loader.dataset)}")
    print(f"Número de batches de test: {len(test_loader)}")

    # 3. Evalúa el modelo
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    print("\nIniciando evaluación...")
    with torch.no_grad(): # Desactiva gradientes
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1) # Índice de la probabilidad máxima
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    print(f"\nPrecisión en test: {accuracy:.2f}%")

    # Genera reporte de clasificación y matriz de confusión
    print("\nReporte de clasificación:")
    print(classification_report(all_labels, all_predictions, target_names=EMOTION_LABELS))

    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS)
    plt.xlabel('Etiqueta predicha')
    plt.ylabel('Etiqueta real')
    plt.title('Matriz de confusión')
    plt.show()

if __name__ == "__main__":
    test_model()
