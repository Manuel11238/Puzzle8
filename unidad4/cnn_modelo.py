
import torch
import torch.nn as nn
import torch.nn.functional as F

class Deep_Emotion(nn.Module):
    def __init__(self):
        super(Deep_Emotion, self).__init__()
        # El modelo está diseñado para recibir imágenes de 1 canal (escala de grises) de 48x48 píxeles.
        # Procesa la imagen a través de varias capas convolucionales, de pooling y una capa totalmente conectada.

        # Primer bloque convolucional
        # Entrada: 1 canal, 48x48
        # Salida después de conv1: (batch_size, 10, 46, 46) -> (48 - 3 + 1 = 46)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3)
        # Salida después de conv2: (batch_size, 10, 44, 44)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3)
        # Salida después de pool1: (batch_size, 10, 22, 22) -> (44 / 2 = 22)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25) # Dropout para evitar overfitting

        # Segundo bloque convolucional
        # Entrada: 10 canales, 22x22
        # Salida después de conv3: (batch_size, 20, 20, 20) -> (22 - 3 + 1 = 20)
        self.conv3 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3)
        # Salida después de conv4: (batch_size, 20, 18, 18)
        self.conv4 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=3)
        # Salida después de pool2: (batch_size, 20, 9, 9) -> (18 / 2 = 9)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.25) # Dropout para evitar overfitting

        # Capa totalmente conectada
        # Después de dos capas de pooling, el tamaño del mapa de características es 9x9.
        # Número de características antes de aplanar: 20 canales * 9 alto * 9 ancho = 1620
        # Salida: 4 (para las 4 clases de emociones de FER2013)
        self.fc1 = nn.Linear(in_features=20 * 9 * 9, out_features=4)

    def forward(self, x):
        # Primer bloque convolucional con ReLU y pooling
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.dropout1(x)

        # Segundo bloque convolucional con ReLU y pooling
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.dropout2(x)

        # Aplana el mapa de características para la capa totalmente conectada
        # El '-1' permite a PyTorch inferir el tamaño del batch
        x = x.view(-1, 20 * 9 * 9)

        # Aplica la capa totalmente conectada final
        x = self.fc1(x)
        return x

if __name__ == "__main__":
    # Ejemplo de uso:
    model = Deep_Emotion()
    print("Arquitectura del modelo Deep_Emotion CNN:")
    print(model)

    # Prueba con un input dummy (batch size 1, 1 canal, imagen 48x48)
    dummy_input = torch.randn(1, 1, 48, 48)
    print(f"\nForma del input dummy: {dummy_input.shape}")

    output = model(dummy_input)
    print(f"Forma de la salida del modelo para el input dummy: {output.shape}") # Debe ser (1, 7)

    # Asegura que la salida tenga la forma esperada
    assert output.shape == (1, 7), "¡La forma de la salida del modelo es incorrecta! Se esperaba (1, 7)."
    print("Prueba de forward exitosa. La salida coincide con las 7 clases de emociones esperadas.")
