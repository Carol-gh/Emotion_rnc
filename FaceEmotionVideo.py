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
#import ssl
app = Flask(__name__)
CORS(app)

classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

emotionModel = load_model("modelFEC.h5")

@app.route("/predict_emotion", methods=["POST"])
def process_image():
    image_url = request.json.get("imageUrl")

    if not image_url:
        return jsonify({"error": "No se proporciono la URL de la imagen"}), 400

    response = requests.get(image_url)
    if response.status_code != 200:
        return jsonify({"error": "Error al descargar la imagen"}), 500

    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    results = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")

            face = img[startY:endY, startX:endX]

            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (48, 48))
            face = np.expand_dims(face, axis=0)
            face = np.expand_dims(face, axis=3)

            preds = emotionModel.predict(face)
            emotion_class = classes[np.argmax(preds)]
            emotion_prob = preds[0, np.argmax(preds)]

            result = {
                "emotion_class": emotion_class,
                "emotion_prob": float(emotion_prob),
                "bounding_box": {"startX": int(startX), "startY": int(startY), "endX": int(endX), "endY": int(endY)}
            }
            results.append(result)

    return jsonify({"results": results})

if __name__ == "__main__":
    app.run()
# en el servidor necesitaras habilitar lo de abajo y desactivar app.run() 
#context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
    #context.load_cert_chain('/etc/letsencrypt/live/emocion.online/fullchain.pem', >

    #app.run(host='0.0.0.0', port=443, ssl_context=context)