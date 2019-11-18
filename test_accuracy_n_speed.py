############################
# Created by Lisa Tsai on 2019/09/17
# Purpose : Test the accuracy and speed of different embedded system
# Hardware : Raspberry Pi 3/4 w/ or w/o TPU, Jetson nano
# Model : Tensorflow Lite
############################
import os
import time
import argparse
from edgetpu.classification.engine import ClassificationEngine
from edgetpu.utils import dataset_utils
from PIL import Image


  
  
  # Run inference.
  #img = Image.open(args.image)
  #for result in engine.ClassifyWithImage(img, top_k=3):
  #  print('---------------------------')
  #  print(labels[result[0]])
  #  print('Score : ', result[1])


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
    # Prepare labels.
    labels = dataset_utils.ReadLabelFile(args.label)
    # Initialize engine
    engine = ClassificationEngine(args.model)
    print("load model costs %s sec"%(time.time()-start_time))
    
    dir_path = '/home/pi/Desktop/test-dataset/'
    os.chdir(dir_path)
    subdir_list=next(os.walk('.'))[1]
    print(subdir_list)

    acc=[]
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
            for result in engine.ClassifyWithImage(img, top_k=1):
                if labels[result[0]]==subdir:
                    count+=1
                    #print(labels[result[0]],count)
        acc.append(count)
        print("recognition of %s images costs %s sec"%(len(files),time.time()-start_time))
    for x in range(len(acc)):
        print(subdir_list[x],acc[x])
