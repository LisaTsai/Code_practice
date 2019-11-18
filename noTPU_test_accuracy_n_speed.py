############################
# Created by Lisa Tsai on 2019/09/17
# Purpose : Test the accuracy and speed of different embedded system
# Hardware : Raspberry Pi 3/4 w/ or w/o TPU, Jetson nano
# Model : Tensorflow Lite
############################

import os
import time
import argparse
from tflite_runtime.interpreter import Interpreter
from PIL import Image
import numpy as np


def load_labels(path):
  with open(path, 'r') as f:
    return {i: line.strip() for i, line in enumerate(f.readlines())}


def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def classify_image(interpreter, image, top_k=1):
  """Returns a sorted array of classification results."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = np.squeeze(interpreter.get_tensor(output_details['index']))

  # If the model is quantized (uint8 data), then dequantize the results
  if output_details['dtype'] == np.uint8:
    scale, zero_point = output_details['quantization']
    output = scale * (output - zero_point)

  ordered = np.argpartition(-output, top_k)
  return [(i, output[i]) for i in ordered[:top_k]]

if __name__ == '__main__':
    
    start_time=time.time()

    # Load model
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--model', help='File path of Tflite model.',
      default='/home/pi/Desktop/AUTO_ML/models_edge_ICN6216886327266610278_2019-08-26_07-02-41-723_tflite_model.tflite', required=False)
    parser.add_argument('--label', help='File path of label file.',
                      default = '/home/pi/Desktop/AUTO_ML/models_edge_ICN6216886327266610278_2019-08-26_07-02-41-723_tflite_dict.txt',required=False)
    #parser.add_argument(
    #    '--image', help='File path of the image to be tested.', required=False)
    args = parser.parse_args()

    labels = load_labels(args.label)
    interpreter = Interpreter(args.model)
    interpreter.allocate_tensors()
    _, height, width, _ = interpreter.get_input_details()[0]['shape']

    print("load model costs %s sec"%(time.time()-start_time))
    
    dir_path = '/home/pi/Desktop/test-dataset/'
    os.chdir(dir_path)
    subdir_list=next(os.walk('.'))[1]
    print(subdir_list)

    acc=[]
    ave_time=[]
    for subdir in subdir_list:
        files=[]
        count=0
        sub_p = os.path.join(dir_path,subdir)
        for r,d,f in os.walk(sub_p):
            for file in f:
                if file[:2]!="._" and file[-3:]=="jpg":
                    files.append(os.path.join(r,file))
        #print(files[1:10])
        print('There are %s images in subdirectory %s'%(len(files),subdir))
        start_time=time.time()
        for i_path in files:
            img = Image.open(i_path)
            img = img.resize((224,224),Image.ANTIALIAS)
            results = classify_image(interpreter, img)
            label_id, prob = results[0]
            if labels[label_id]==subdir:
                count+=1
            #print(labels[label_id],count)
        acc.append(count)
        delta_t = time.time()-start_time
        ave_time.append(delta_t)
        print("recognition of %s images costs %s sec"%(len(files),delta_t))
    for x in range(len(acc)):
        print(subdir_list[x],acc[x],ave_time[x])
