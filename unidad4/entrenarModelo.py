
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time

# Importa configuraciones
from config import (
    NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, MODEL_SAVE_PATH,
    TRAIN_DATA_DIR_FOR_MODEL, VALIDATION_DATA_DIR_FOR_MODEL, EMOTION_LABEL_MAP
)

# Importa la definición del modelo CNN
from cnn_modelo import Deep_Emotion

# Importa los data loaders adaptados
from data_loaders import get_dataloaders 

def train_model():
    """
    Configurando y ejecutando el proceso de entrenamiento para el modelo Deep_Emotion.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")

    # 1. Carga de datos
    print("\nCargando datasets...")
    train_loader, val_loader = get_dataloaders(
        train_data_dir=TRAIN_DATA_DIR_FOR_MODEL,
        val_data_dir=VALIDATION_DATA_DIR_FOR_MODEL,
        batch_size=BATCH_SIZE
    )
    print(f"Número de batches de entrenamiento: {len(train_loader)}")
    print(f"Número de batches de validación: {len(val_loader)}")

    # 2. Inicializa modelo, optimizador y función de pérdida
    model = Deep_Emotion().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss() # Para clasificación multiclase

    # Asegura que el directorio para guardar el modelo exista
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    best_val_accuracy = 0.0

    print(f"\nIniciando entrenamiento por {NUM_EPOCHS} épocas...")

    for epoch in range(NUM_EPOCHS):
        model.train() # Modo entrenamiento
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        start_time = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad() # Reinicia gradientes
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward() # Backpropagation
            optimizer.step() # Actualiza pesos

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            if (batch_idx + 1) % 100 == 0: # Muestra progreso cada 100 batches
                print(f"Época [{epoch+1}/{NUM_EPOCHS}], Paso [{batch_idx+1}/{len(train_loader)}], "
                      f"Pérdida: {loss.item():.4f}")

        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_accuracy = 100 * correct_train / total_train

        # Fase de validación
        model.eval() # Modo evaluación
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad(): # Desactiva gradientes para validación
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_accuracy = 100 * correct_val / total_val
        end_time = time.time()
        epoch_time = end_time - start_time

        print(f"\nÉpoca {epoch+1}/{NUM_EPOCHS} completada en {epoch_time:.2f}s:")
        print(f"  Pérdida entrenamiento: {epoch_train_loss:.4f}, Precisión entrenamiento: {epoch_train_accuracy:.2f}%")
        print(f"  Pérdida validación: {epoch_val_loss:.4f}, Precisión validación: {epoch_val_accuracy:.2f}%")

        # Guarda el mejor modelo según la precisión de validación
        if epoch_val_accuracy > best_val_accuracy:
            best_val_accuracy = epoch_val_accuracy
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  Modelo guardado en {MODEL_SAVE_PATH} con mejor precisión de validación: {best_val_accuracy:.2f}%")

    print("\n¡Entrenamiento completo!")
    print(f"Mejor precisión de validación alcanzada: {best_val_accuracy:.2f}%")

if __name__ == "__main__":
    train_model()
