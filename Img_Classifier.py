import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
import os
import numpy as np
import shutil as sh

os.chdir(r"C:\Users\v\Desktop\Code\GM_Remover\ ")

new_model = load_model('img_classifier.h5')

data_dir = 'c:\\Users\\v\\Desktop\\Code\\GM_Remover\\WA Images'
target1 = 'c:\\Users\\v\\Desktop\\Code\\GM_Remover\\Relevent'
target2 = 'c:\\Users\\v\\Desktop\\Code\\GM_Remover\\Irrelevent'

for image in os.listdir(data_dir):     
    image_path = os.path.join(data_dir, image)
    try: 
        img = cv2.imread(image_path)
        resize = tf.image.resize(img, (256,256))
        yhat = new_model.predict(np.expand_dims(resize/255, 0))
        if yhat > 0.5: 
            print('Relevent {}'.format(image_path))
            sh.copy(image_path, target1)
        else:
            print('Irrelevent {}'.format(image_path))
            sh.copy(image_path, target2)
    except Exception as e: 
        print('Issue with image {}'.format(image_path))