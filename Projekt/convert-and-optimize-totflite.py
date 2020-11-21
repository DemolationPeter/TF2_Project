import logging
logging.getLogger("tensorflow").setLevel(logging.DEBUG)


import tensorflow as tf
import os
import numpy as np
import cv2


#https://www.tensorflow.org/lite/performance/post_training_quantization



saved_model_dir = '/home/marinfrede/Dokumente/Studienarbeit/Projekt/workspace/exported-models/my_model/saved_model'
train_images_path_new = '/home/marinfrede/Dokumente/Studienarbeit/Projekt/workspace/images/new'
train_images_path = '/home/marinfrede/Dokumente/Studienarbeit/Projekt/workspace/images/train'
tflite_path = '/home/marinfrede/Dokumente/Studienarbeit/Projekt'


# nur in tflite umgewandelt, keine Ahnung wieso größer als ursprüngliches Modell
#converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
#tflite_model_lite = converter.convert()

#with open("converted_model_lite.tflite", "wb") as f:
#   f.write(tflite_model_lite)

###############################################################################################################################################

# Dynamic range quantization ( statically quantizes only the weights from floating point to integer)

#converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
#converter.optimizations = [tf.lite.Optimize.DEFAULT]		# ohne diesen Zusatz kein Lite Modell, steht so nicht in der Anleitung der Weg funktionierte nicht, Hiermit Dynamic range quantization, d.h weights from fp to int 8-bits, ohne diesen Zusatz ist das neue Modell größer als das ursprüngliche -> kann nicht sein!! damikeine Vereinfachung
#tflite_model_dynamic_range = converter.convert()

#with open("converted_model_quant_dynamic_range.tflite", "wb") as f:
#   f.write(tflite_model_dynamic_range)


#interpreter = tf.lite.Interpreter(model_content=tflite_model_dynamic_range)
#input_type = interpreter.get_input_details()[0]['dtype']
#print('input: ', input_type)
#output_type = interpreter.get_output_details()[0]['dtype']
#print('output: ', output_type)

##########################################################################################################################################################

NORM_H = 320
NORM_W = 320
BATCH_SIZE = 1


# https://www.tensorflow.org/lite/performance/post_training_quantization    
# https://www.tensorflow.org/lite/guide/inference


#  weights to float16
#converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
#converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.target_spec.supported_types = [tf.float16]
#tflite_quant_model_16float = converter.convert()

#with open("converted_model_quant_16float.tflite", "wb") as f:
#   f.write(tflite_quant_model_16float)

# Ergebniss ist größer als dynamic range -> macht Sinn, da Gewichte nur float16 sind und nicht int8

#interpreter = tf.lite.Interpreter(model_content=tflite_quant_model_16float)
#input_type = interpreter.get_input_details()[0]['dtype']
#print('input: ', input_type)
#output_type = interpreter.get_output_details()[0]['dtype']
#print('output: ', output_type)


############################################################################################################################################################

# Full integer quantization
#input_value = os.listdir(train_images_path_new)
#def representative_data_gen():
#  for input_value in tf.data.Dataset.from_tensor_slices(train_images_path_new).batch(1).take(4):
#    yield [input_value]

a = []
for f_name in os.listdir(train_images_path_new):
                 file_name = f_name
                 #print(file_name)
                 file_path = os.path.normpath(os.path.join(train_images_path_new, file_name))
                 a.append(file_path)

def representative_data_gen():
 dataset_list = os.listdir(train_images_path_new)
 for i in range(4):
   image = next(iter(dataset_list))
   imgae = tf.io.read_file(image)
   image = tf.io.decode_jpeg(image, channels=3)
   image = tf.image.resize(image, [NORM_H, NORM_W])
   image = tf.cast(image / 255., tf.float32)
   image = tf.expand_dims(image, 0)
   print(image)
   yield [image]


def rep_data():
  for f_name in os.listdir(train_images_path_new):
                 file_name = f_name
                 #print(file_name)
                 file_path = os.path.normpath(os.path.join(train_images_path_new, file_name))
                 #print(file_path)
                 img = cv2.imread(file_path)
                 img = cv2.resize(img, (NORM_H, NORM_W))
                 img = img / 255.0
                 img = img.reshape(1, 320, 320, 3)
                 img = img.astype(np.float32)
                 #print(f_name)  #f_name sind die Datei Namen im Verzeichnis os.listdir(...)
                 print(img)
                 yield[img]


converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_data
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.target_spec.supported_types = [tf.int8]
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8
tflite_quant_model_full_int = converter.convert()




#########################################################################################################################################################

def rep_data_gen():
  a = []
  for f_name in os.listdir(train_images_path_new):
            if f_name.endswith('.jpg'):
                 file_name = f_name
                 #print(file_name)
                 file_path = os.path.normpath(os.path.join(train_images_path, file_name))
                 #print(file_path)
                 img = cv2.imread(file_path)
                 img = cv2.resize(img, (NORM_H, NORM_W))
                 img = img / 255.0
                 img = img.astype(np.float32)
                 a.append(img)
                 print(f_name)
  #a = np.array(a)
  #print(a.shape) # a is np array of 160 3D images
  #img = tf.data.Dataset.from_tensor_slices(a).batch(1)
  #for i in img.take(BATCH_SIZE):
  #  print(i)
  #  yield[i]








#def representative_data_gen1():
#  file_name_a = []
#  for f_name in os.listdir(train_images_path_new):
#    file_path = os.path.normpath(os.path.join(train_images_path_new, f_name))
#    file_name_a.append(file_path)

#  print(file_name_a)


 # dataset_list = tf.data.Dataset.list_files(file_name_a)
 # for i in range(5):
 #    image = next(iter(dataset_list))
 #    image = tf.io.read_file(image)
 #    image = tf.io.decode_jpeg(image, channels=3)
 #    image = tf.image.resize(image, [320, 320])
 #    image = tf.cast(image / 255., tf.float32)
 #    image = tf.expand_dims(image, axis=0)
 #    yield[image]



#converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
#converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.representative_dataset = representative_data_gen
#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
#converter.inference_input_type = tf.uint8
#converter.inference_output_type = tf.uint8
#tflite_model_quant = converter.convert()


#with open("converted_model_quant.tflite", "wb") as f:
#   f.write(tflite_model_quant)














