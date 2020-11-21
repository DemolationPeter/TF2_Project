import tensorflow as tf
import os
import numpy as np
import cv2


saved_model_dir = '/home/marinfrede/Dokumente/Studienarbeit/Projekt/workspace/exported-models/my_model/saved_model'
images_path = '/home/marinfrede/Dokumente/Studienarbeit/Projekt/workspace/images/new'


def rep_data():
  for f_name in os.listdir(images_path):
                 file_path = os.path.normpath(os.path.join(images_path, f_name))
                 img = cv2.imread(file_path)
                 img = cv2.resize(img, (320, 320))
                 img = img / 255.0
                 img = np.reshape(img, (1, 320, 320, 3))
                 img = img.astype(np.float32)
                 test = tf.dtypes.as_dtype(img)
                 print(test)
                 print(img)
                 yield[img]


converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_data
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8  
converter.inference_output_type = tf.uint8 
tflite_quant_model_full_int = converter.convert()

with open('model_quant_int.tflite', 'wb') as f:
  f.write(tflite_quant_model_full_int)



















