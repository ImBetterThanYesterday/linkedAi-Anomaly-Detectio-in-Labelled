import os
import json
import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from io import BytesIO
import shutil
from tempfile import TemporaryDirectory
import gdown
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image as kimage

# Cargar las variables de entorno
from dotenv import load_dotenv
load_dotenv()

# Rutas de los archivos
MODEL_PATH = os.getenv('MODEL_PATH', 'model_classifier.h5')
DRIVE_MODEL_URL = os.getenv('DRIVE_MODEL_URL', 'https://drive.google.com/uc?id=1AZ-3d6Bs60vPKYh5HRNTdNHwU1lFGrwn')

# Definir las etiquetas de las clases (ajusta según tu caso)
class_labels = ['hammer', 'plier', 'screw', 'screwdriver', 'wrench']

# Crear la instancia de la aplicación FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todos los orígenes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def download_model_from_drive(drive_url, output_path):
    """Descargar el modelo desde Google Drive."""
    try:
        gdown.download(drive_url, output_path, quiet=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al descargar el modelo desde Google Drive: {e}")

def load_model(model_path):
    """Cargar el modelo de clasificación guardado."""
    if not os.path.exists(model_path):
        download_model_from_drive(DRIVE_MODEL_URL, model_path)
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al cargar el modelo: {e}")

def preprocess_image(image: Image.Image):
    """Preprocesar la imagen para el modelo VGG16."""
    img = image.resize((224, 224))
    img_array = kimage.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def extract_embeddings(image_array):
    """Extraer los embeddings de la imagen usando VGG16."""
    model = VGG16(weights='imagenet', include_top=False)
    embeddings = model.predict(image_array)
    embeddings = embeddings.flatten()
    return embeddings

def process_annotations_and_images(annotations, images_dir, output_dir):
    """Procesar anotaciones y recortar objetos de las imágenes."""
    for entry in annotations:
        image_filename = entry['image']
        image_path = os.path.join(images_dir, image_filename)
        
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue

        # Extract the base name without extension
        base_name = os.path.splitext(image_filename)[0]

        # Iterate over each tagged object in the image
        for tag in entry['tags']:
            class_name = tag['name']
            x, y, w, h = int(tag['pos']['x']), int(tag['pos']['y']), int(tag['pos']['w']), int(tag['pos']['h'])

            # Create directory for the class if it doesn't exist
            class_dir = os.path.join(output_dir, class_name)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)

            # Crop the object from the image
            cropped_object = image[y:y+h, x:x+w]

            # Create a filename for the cropped object
            object_filename = f"{base_name}.png"
            object_path = os.path.join(class_dir, object_filename)

            # Save the cropped object
            cv2.imwrite(object_path, cropped_object)

@app.on_event("startup")
def startup_event():
    """Cargar el modelo al iniciar la aplicación."""
    global classifier_model
    classifier_model = load_model(MODEL_PATH)

@app.post("/predict/")
async def predict_class(class_labelled_by_person: str = Form(...), file: UploadFile = File(...)):
    """Endpoint para predecir la clase de una imagen."""
    try:
        image = Image.open(BytesIO(await file.read()))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer la imagen: {e}")

    try:
        image_array = preprocess_image(image)
        embeddings = extract_embeddings(image_array)
        embeddings = np.expand_dims(embeddings, axis=0)
        
        predictions = classifier_model.predict(embeddings)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_name = class_labels[predicted_class_index]  # Obtener el nombre de la clase
        print("tes")
        anomaly = predicted_class_name != class_labelled_by_person
        
        return {
            "predicted_class": predicted_class_name,
            "actual_class": class_labelled_by_person,
            "Anomaly": anomaly
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al predecir la clase: {e}")

@app.post("/process/")
async def process_files(annotation_file: UploadFile = File(...), images: list[UploadFile] = File(...)):
    """Endpoint para procesar anotaciones e imágenes y realizar predicciones."""
    try:
        # Crear directorio temporal para imágenes y recortes
        with TemporaryDirectory() as temp_dir:
            images_dir = os.path.join(temp_dir, "images")
            output_dir = os.path.join(temp_dir, "cropped_objects")
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
            
            # Guardar imágenes subidas en el directorio temporal
            for img in images:
                img_path = os.path.join(images_dir, img.filename)
                with open(img_path, "wb") as f:
                    f.write(await img.read())

            # Leer archivo de anotaciones
            annotations = json.loads(await annotation_file.read())
            
            # Procesar anotaciones y recortar objetos
            process_annotations_and_images(annotations, images_dir, output_dir)
            
            # Predecir clases de los objetos recortados
            predictions = []
            for class_name in os.listdir(output_dir):
                class_dir = os.path.join(output_dir, class_name)
                for cropped_image_name in os.listdir(class_dir):
                    cropped_image_path = os.path.join(class_dir, cropped_image_name)
                    image = Image.open(cropped_image_path)
                    image_array = preprocess_image(image)
                    embeddings = extract_embeddings(image_array)
                    embeddings = np.expand_dims(embeddings, axis=0)
                    
                    prediction = classifier_model.predict(embeddings)
                    predicted_class_index = np.argmax(prediction, axis=1)[0]
                    predicted_class_name = class_labels[predicted_class_index]
                    
                    anomaly = predicted_class_name != class_name
                    predictions.append({
                        "cropped_image": cropped_image_name,
                        "predicted_class": predicted_class_name,
                        "actual_class": class_name,
                        "anomaly": anomaly
                    })

            return {"predictions": predictions}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar los archivos: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
