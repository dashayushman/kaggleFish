import glob
import numpy as np
from PIL import Image
import tensorflow as tf

dir_path_ALB = "/home/anubhav/PycharmProjects/TensorFlow/Fisheries/train/ALB/*.jpg"
dir_path_BET = "/home/anubhav/PycharmProjects/TensorFlow/Fisheries/train/BET/*.jpg"
dir_path_DOL = "/home/anubhav/PycharmProjects/TensorFlow/Fisheries/train/DOL/*.jpg"
dir_path_LAG = "/home/anubhav/PycharmProjects/TensorFlow/Fisheries/train/LAG/*.jpg"
dir_path_NoF = "/home/anubhav/PycharmProjects/TensorFlow/Fisheries/train/NoF/*.jpg"
dir_path_Other = "/home/anubhav/PycharmProjects/TensorFlow/Fisheries/train/OTHER/*.jpg"
dir_path_SHARK = "/home/anubhav/PycharmProjects/TensorFlow/Fisheries/train/SHARK/*.jpg"
dir_path_YFT = "/home/anubhav/PycharmProjects/TensorFlow/Fisheries/train/YFT/*.jpg"

filenames_ALB = glob.glob(dir_path_ALB)
filenames_BET = glob.glob(dir_path_BET)
filenames_DOL = glob.glob(dir_path_DOL)
filenames_LAG = glob.glob(dir_path_LAG)
filenames_NoF = glob.glob(dir_path_NoF)
filenames_OTHER = glob.glob(dir_path_Other)
filenames_SHARK = glob.glob(dir_path_SHARK)
filenames_YFT = glob.glob(dir_path_YFT)

print filenames_ALB[0:5]
print len(filenames_BET)
print len(filenames_ALB)

fish_id = {
    "ALB" : 0,
    "BET" : 1,
    "DOL" : 2,
    "LAG" : 3,
    "NoF" : 4,
    "OTHER" : 5,
    "SHARK" : 6,
    "YFT" : 7
}

id_fish = {v:k for k,v in fish_id.iteritems()}

print id_fish


def gen_batches(batch_size):
    images = []
    labels = []
    #Each fish_type has to have equal representation in the batch
    num_classes = 8
    num_images_per_class = batch_size/8
    idx_ALB = np.random.randint(0,len(filenames_ALB),num_images_per_class)
    idx_BET = np.random.randint(0,len(filenames_BET),num_images_per_class)
    idx_DOL = np.random.randint(0,len(filenames_DOL),num_images_per_class)
    idx_LAG = np.random.randint(0,len(filenames_LAG),num_images_per_class)
    idx_NoF = np.random.randint(0,len(filenames_NoF),num_images_per_class)
    idx_oth = np.random.randint(0,len(filenames_OTHER),num_images_per_class)
    idx_SHA = np.random.randint(0,len(filenames_SHARK),num_images_per_class)
    idx_YFT = np.random.randint(0,len(filenames_YFT),num_images_per_class)

    #Extract images of classes

    for i in idx_ALB:
        img = Image.open(filenames_ALB[i])
        img.load()
        image_arr = np.asarray(img,dtype=np.float64)
        images.append(image_arr)
    for i in idx_BET:
        img = Image.open(filenames_BET[i])
        img.load()
        image_arr = np.asarray(img,dtype=np.float64)
        images.append(image_arr)
    for i in idx_DOL:
        img = Image.open(filenames_DOL[i])
        img.load()
        image_arr = np.asarray(img, dtype=np.float64)
        images.append(image_arr)
    for i in idx_LAG:
        img = Image.open(filenames_DOL[i])
        img.load()
        image_arr = np.asarray(img, dtype=np.float64)
        images.append(image_arr)
    for i in idx_NoF:
        img = Image.open(filenames_NoF[i])
        img.load()
        image_arr = np.asarray(img, dtype=np.float64)
        images.append(image_arr)
    for i in idx_oth:
        img = Image.open(filenames_OTHER[i])
        img.load()
        image_arr = np.asarray(img, dtype=np.float64)
        images.append(image_arr)
    for i in idx_SHA:
        img = Image.open(filenames_SHARK[i])
        img.load()
        image_arr = np.asarray(img, dtype=np.float64)
        images.append(image_arr)
    for i in idx_YFT:
        img = Image.open(filenames_YFT[i])
        img.load()
        image_arr = np.asarray(img, dtype=np.float64)
        images.append(image_arr)

    images = np.array(images)


    labels = []



    for k in range(batch_size):
        y = np.zeros(8, dtype=np.float64)
        y[k/num_images_per_class] = 1.0
        labels.append(y)


    labels = np.array(labels)

    return images,labels



def get_weights(shape,name):
    return tf.get_variable(name,shape,tf.float64,tf.contrib.layers.xavier_initializer())

def get_biases(shape):
    return tf.zeros(shape=shape,dtype=tf.float64)

num_classes = 8

x = tf.placeholder([None,720,1200,3])
y = tf.placeholder([None,num_classes])