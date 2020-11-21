# TF2_Project

This is the full project folder. The models folder from the Object Detection API is not included

The code in "convert-and-optimize-totflite_mini_exam.py" ist the minimal example from the following post (https://stackoverflow.com/questions/64621991/tensorflow-tf2-quantization-to-full-integer-error-with-tfliteconverter-runtime?noredirect=1#comment114631842_64621991)

Also there are three to .tflite converted models.
I converted these models with the script "convert-and-optimize-totflite.py". The script is a little bit messy. I think it worked one month a go but not the part with the INT8 quantization. 


In the folder /workspace are several folders.
/exported-models -> the graph after using the export_inference_graph.py
/models the checkpoints from the training process
also the files I used to train and export the data


Since mid October I'm not working on these files any more. I moved back to TF1.15.
