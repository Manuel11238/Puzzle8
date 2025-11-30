
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import time
import os

# Importa configuraciones desde config.py
from config import (
    IMG_HEIGHT, IMG_WIDTH, MODEL_SAVE_PATH, FACE_CASCADE_PATH, EMOTION_LABELS
)

# Importa la definición del modelo CNN
from cnn_modelo import Deep_Emotion

# Función para cargar el modelo entrenado
def load_trained_model(model_path, device):
    """
    Carga el modelo Deep_Emotion entrenado desde la ruta especificada.
    """
    model = Deep_Emotion()
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Pesos del modelo no encontrados en {model_path}. Entrena el modelo primero.")

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval() # Modo evaluación
        print(f"Modelo cargado correctamente desde {model_path} en {device}")
    except Exception as e:
        raise RuntimeError(f"Error cargando el modelo desde {model_path}: {e}")
    return model

# Función para preprocesar la región del rostro detectado
def preprocess_face_roi(face_image_bgr, transform):
    """
    Preprocesa una región de imagen de rostro (BGR) para la inferencia del modelo.
    """
    # Convierte de BGR de OpenCV a imagen PIL RGB 
    pil_image = Image.fromarray(cv2.cvtColor(face_image_bgr, cv2.COLOR_BGR2RGB))
    # Aplica las transformaciones definidas 
    processed_image = transform(pil_image)
    # Añade dimensión de batch: (C, H, W) -> (1, C, H, W)
    processed_image = processed_image.unsqueeze(0)
    return processed_image

# Función para predecir la emoción a partir de la imagen procesada
def predict_emotion(model, face_tensor, device):
    """
    Predice la emoción a partir de un tensor de imagen de rostro preprocesado.
    """
    with torch.no_grad():
        face_tensor = face_tensor.to(device)
        outputs = model(face_tensor)
        probabilities = F.softmax(outputs, dim=1)
        _, predicted_class_idx = torch.max(probabilities, 1)
        confidence = probabilities[0, predicted_class_idx.item()].item()
        return predicted_class_idx.item(), confidence

# Función principal para ejecutar la inferencia en tiempo real con la webcam
def run_webcam_inference():
    """
    Ejecuta la predicción de emociones en tiempo real usando la webcam.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")

    # Carga el modelo entrenado
    try:
        model = load_trained_model(MODEL_SAVE_PATH, device)
    except (FileNotFoundError, RuntimeError) as e:
        print(e)
        return

    # Define las transformaciones de preprocesamiento para cada frame capturado
    inference_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), # Asegura 1 canal
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)), # Redimensiona a 48x48
        transforms.ToTensor(), # Convierte a tensor de PyTorch
        transforms.Normalize(mean=[0.5], std=[0.5]) # Normaliza
    ])

    # Carga el clasificador Haar Cascade para detección de rostros
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    if face_cascade.empty():
        print(f"Error: No se pudo cargar el clasificador de rostros desde {FACE_CASCADE_PATH}.")
        print("Asegúrate de que 'haarcascade_frontalface_default.xml' esté en la ruta indicada.")
        return

    # Abre la webcam
    cap = cv2.VideoCapture(0) # 0 para la webcam por defecto
    if not cap.isOpened():
        print("Error: No se pudo abrir la webcam. Verifica que la cámara esté conectada y no esté en uso.")
        return

    print("\nFeed de webcam iniciado. Presiona 'q' para salir.")

    prev_frame_time = 0
    new_frame_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo capturar el frame de la cámara.")
            break

        # Convierte el frame a escala de grises para la detección de rostros
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detecta rostros en el frame en escala de grises
        faces = face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:
            # Extrae la región de interés (ROI) del rostro de la imagen original en color
            face_roi_bgr = frame[y:y+h, x:x+w]
            
            # Preprocesa la ROI del rostro para la inferencia del modelo
            processed_face_tensor = preprocess_face_roi(face_roi_bgr, inference_transform)
            
            # Predice la emoción
            predicted_emotion_idx, confidence = predict_emotion(model, processed_face_tensor, device)
            predicted_emotion_label = EMOTION_LABELS[predicted_emotion_idx]

            # Dibuja un rectángulo y la etiqueta sobre el rostro detectado
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # Rectángulo azul
            text = f"{predicted_emotion_label} ({confidence:.2f})"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2) # Texto azul

        # Calcula y muestra los FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) # Texto verde

        # Muestra el frame en una ventana
        cv2.imshow('Emotion Detection', frame)

        # Rompe el bucle si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libera los recursos de la cámara y cierra las ventanas
    cap.release()
    cv2.destroyAllWindows()
    print("Inferencia con webcam finalizada.")

if __name__ == "__main__":
    run_webcam_inference()
