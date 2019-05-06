import keras_segmentation
from keras.models import load_model
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


import keras.backend.tensorflow_backend as KTF


import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

def get_session(gpu_fraction=0.3):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    print("num_threads:",num_threads)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    print("gpu_options:", gpu_options)
    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
KTF.set_session(get_session())

#注意寻找好模型 U-Net FCN32 等。。。。
#应为GPU 内存不足无法训练，只能去修改  input_height=192, input_width=224  输入参数（x % 32 ==0 ）

model = keras_segmentation.models.unet.unet(n_classes=51 ,
                                            input_height=192,
                                            input_width=224 )

#===============训练====================
model.train(
    train_images =  "dataset1/images_prepped_train/",
    train_annotations = "dataset1/annotations_prepped_train/",
    checkpoints_path = "e_5_unet_1",
    epochs = 5,  #一批训练次数
    batch_size = 2  # 分割数据
)

#===============预测====================
import cv2
import time

model.load_weights("e_5_unet_1.4")

a = time.time()
frame=cv2.imread('2.png')
out = model.predict_segmentation(
    inp=frame,
    out_fname = "out_unet_e5_2.png"
)

print("s:",(time.time() - a)/1000)


