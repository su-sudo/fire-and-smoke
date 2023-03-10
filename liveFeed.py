import cv2
from keras.models import  load_model
import numpy as np

print("[INFO] loading model...")
model = load_model('output/fire_detection.h5')
classes = ['Fire','Non_Fire'  ]
count =0
images = []
labels = []

vidcap = cv2.VideoCapture(0)

if vidcap.isOpened():
    ret, frame = vidcap.read()
    if ret:
        while (True):
            ret, img = vidcap.read()
            img = cv2.resize(img, (128, 128))
            img = img.astype('float32') / 256
            pred = model.predict(np.expand_dims(img, axis=0))[0]
            result = classes[np.argmax(pred)]
            img = cv2.resize(img, (480, 500))

            cv2.putText(img, result, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 3)
            cv2.imshow("Frame", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        print("Error : Failed to capture frame")
else:
    print("Cannot open camera")