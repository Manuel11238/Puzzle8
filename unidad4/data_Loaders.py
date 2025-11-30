
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import numpy as np

# Importa configuraciones
from config import IMG_HEIGHT, IMG_WIDTH, EMOTION_LABELS, LABEL_TO_INT

class EmotionDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Llena image_paths y labels según la estructura de carpeta
        for label_name in EMOTION_LABELS:
            label_int = LABEL_TO_INT[label_name]
            emotion_folder = os.path.join(data_dir, label_name)
            if not os.path.exists(emotion_folder):
                print(f"Advertencia: Carpeta de emoción '{emotion_folder}' no encontrada. Se omite.")
                continue
            
            for img_name in os.listdir(emotion_folder):
                if img_name.endswith(('.png', '.jpg', '.jpeg')): 
                    img_path = os.path.join(emotion_folder, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(label_int)

        print(f"Cargadas {len(self.image_paths)} muestras desde {data_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Carga la imagen en escala de grises
        image = Image.open(img_path).convert('L') # 'L' para escala de grises

        if self.transform:
            image = self.transform(image)
        
        return image, label

# Función para obtener los DataLoaders de entrenamiento y validación
def get_dataloaders(train_data_dir, val_data_dir, batch_size):
    train_transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)), # Asegura tamaño consistente
        transforms.Grayscale(num_output_channels=1), # Asegura 1 canal si no lo está
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]) # Normaliza imágenes en escala de grises
    ])

    # Transforms de validación/test (sin aumento)
    val_test_transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = EmotionDataset(train_data_dir, transform=train_transform)
    val_dataset = EmotionDataset(val_data_dir, transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count() // 2 or 1)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count() // 2 or 1)

    return train_loader, val_loader

if __name__ == '__main__':
    # Ejemplo de uso:
    from config import TRAIN_DATA_DIR_FOR_MODEL, VALIDATION_DATA_DIR_FOR_MODEL, BATCH_SIZE
    train_loader, val_loader = get_dataloaders(
        TRAIN_DATA_DIR_FOR_MODEL,
        VALIDATION_DATA_DIR_FOR_MODEL,
        BATCH_SIZE
    )

    print(f"\nEjemplo: Primer batch del DataLoader de entrenamiento:")
    images, labels = next(iter(train_loader))
    print(f"Forma del batch de imágenes: {images.shape}") # Esperado: [batch_size, 1, 48, 48]
    print(f"Forma del batch de etiquetas: {labels.shape}") # Esperado: [batch_size]
    print(f"Etiquetas: {labels.numpy()}")

    print(f"\nNúmero de batches en train_loader: {len(train_loader)}")
    print(f"Número de batches en val_loader: {len(val_loader)}")
