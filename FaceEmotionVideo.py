from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import imutils
import numpy as np
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import glob
import requests

app = Flask(__name__)
CORS(app)

classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Cargamos el modelo de detección de rostros
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Carga el modelo de clasificación de emociones
emotionModel = load_model("modelFEC.h5")

@app.route("/predict_emotion", methods=["POST"])
def process_image():
    # Obtener la URL de la imagen del cuerpo de la solicitud POST
    image_url = request.json.get("imageUrl")

    # Validar si se proporcionó la URL de la imagen
    if not image_url:
        return jsonify({"error": "No se proporcionó la URL de la imagen"}), 400

    # Descargar la imagen desde la URL
    response = requests.get(image_url)
    if response.status_code != 200:
        return jsonify({"error": "Error al descargar la imagen"}), 500

    # Leer la imagen utilizando OpenCV
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Detección de rostros en la imagen utilizando el modelo faceNet
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    results = []

    # Iterar sobre las detecciones de rostros
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filtrar las detecciones con confianza suficiente
        if confidence > 0.5:
            # Obtener las coordenadas de la caja delimitadora del rostro
            box = detections[0, 0, i, 3:7] * np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")

            # Obtener la región de interés (ROI) del rostro detectado
            face = img[startY:endY, startX:endX]

            # Preprocesamiento del rostro para la clasificación de emociones
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (48, 48))
            face = np.expand_dims(face, axis=0)
            face = np.expand_dims(face, axis=3)

            # Clasificación de emociones en el rostro
            preds = emotionModel.predict(face)
            emotion_class = classes[np.argmax(preds)]
            emotion_prob = preds[0, np.argmax(preds)]

            # Agregar el resultado al array de resultados
            result = {
                "emotion_class": emotion_class,
                "emotion_prob": float(emotion_prob),
                "bounding_box": {"startX": int(startX), "startY": int(startY), "endX": int(endX), "endY": int(endY)}
            }
            results.append(result)

    return jsonify({"results": results})

if __name__ == "__main__":
    app.run()
