
from keras.models import  load_model
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
print("[INFO] loading model...")
model = load_model('output/fire_detection.h5')

classes = ['Fire','Non_Fire',]
count =0
images = []
labels = []
for c in classes:
    #try:
        for img in os.listdir('Image Dataset/' + c):
            if count ==2687:
                break
            img = cv2.imread('Image Dataset/' + c + '/' + img)

            count+=1
            print("directory accessed",count,c)
            img = cv2.resize(img, (128, 128))
            images.append(img)
            labels.append([0, 1][c == 'Fire'])

labels = np.array(labels)
labels = np_utils.to_categorical(labels,num_classes=2)
print("to category")
d = {}
classTotals = labels.sum(axis=0)
classWeight = classTotals.max() / classTotals
d[0] = classWeight[0]
d[1] = classWeight[1]
print("test train split")
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.25, shuffle=True, random_state=42)



for i in range(50):
    print("loop")
    random_index = np.random.randint(0,len(X_test))
    org_img = X_test[random_index]*255
    img = org_img.copy()
    img = cv2.resize(img,(128,128))
    img = img.astype('float32')/256
    pred = model.predict(np.expand_dims(img,axis=0))[0]
    result = classes[np.argmax(pred)]
    cv2.putText(org_img, result, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 3)
    cv2.imwrite('output/testing/{}.png'.format(i), org_img)
