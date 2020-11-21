# copy script into /train folder with all the images and .txt files
# script writes file path into the test.txt
# for other image format change .jpg into other format


import os
import numpy

cwd = os.getcwd()

image_path = []
imagetxt_path = [0]

counter = 0
#print(cwd)

for f_name in os.listdir(cwd):
    if f_name.endswith('.jpg'):
        file_path = os.path.normpath(os.path.join(cwd, f_name))
       # print(file_path)
        image_path.append(file_path)
        counter = counter + 1
    else:
        file_path = os.path.normpath(os.path.join(cwd, f_name))
        imagetxt_path.append(file_path)

print(counter)


f = open('/home/marinfrede/Dokumente/Studienarbeit/yolo_detection/darknet/training/model/train.txt', 'w')
f.close()

with open('/home/marinfrede/Dokumente/Studienarbeit/yolo_detection/darknet/training/model/train.txt', 'w') as f:
    for x in image_path:
        f.write("%s\n" % x)
f.close()


for f_name in os.listdir(cwd):

    if f_name.endswith('.txt'):
      
       print(f_name)   

       with open(f_name, 'r') as f:
            filedata = f.read()
            filedata = filedata.replace('Pen', '0')
             
       with open(f_name, 'w') as f:
            write_file = f.write(filedata)

#for x in image_path:
#  print(x)

#for x in imagetxt_path:
#  print(x)
