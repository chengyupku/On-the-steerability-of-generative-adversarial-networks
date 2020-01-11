




output_dir = 'notebooks/models/biggan_linear_zoom'

lr = 0.1
num_samples = 200

# make output directory
import os
os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'output'), exist_ok=True)

module_path = '/home/chengyu/gan_steerability_copy/notebooks'

import io
import numpy as np
import PIL.Image
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import time


tf.device('/gpu:2')
tf.reset_default_graph()
print('Loading BigGAN module from:', module_path)
module = hub.Module(module_path)

inputs = {k: tf.placeholder(v.dtype, v.get_shape().as_list(), k)
          for k, v in module.get_input_info_dict().items()}
output = module(inputs)

print('Inputs:\n', '\n'.join('  {}: {}'.format(*kv) for kv in inputs.items()))
print('Output:', output)

input_z = inputs['z']
input_y = inputs['y']
input_trunc = inputs['truncation']
dim_z = input_z.shape.as_list()[1]
vocab_size = input_y.shape.as_list()[1]

# input placeholders
Nsliders = 1
z = tf.placeholder(tf.float32, shape=(None, dim_z))
y = tf.placeholder(tf.float32, shape=(None, vocab_size))
truncation = tf.placeholder(tf.float32, shape=None)

# original output
inputs_orig = {u'y': y,
               u'z': z,
               u'truncation': truncation}
outputs_orig = module(inputs_orig)

img_size = outputs_orig.shape[1].value
num_channels = outputs_orig.shape[-1].value

# output placeholders
target = tf.placeholder(tf.float32, shape=(None, img_size, img_size, num_channels))
mask = tf.placeholder(tf.float32, shape=(None, img_size, img_size, num_channels))

# set walk parameters
alpha = tf.placeholder(tf.float32, shape=(None, Nsliders))
w = tf.Variable(np.random.normal(0.0, 0.1, [1, z.shape[1], Nsliders]), name='walk', dtype=np.float32)

# transform the output
z_new = z
for i in range(Nsliders):
    z_new = z_new+tf.expand_dims(alpha[:,i], axis=1)*w[:,:,i]
transformed_inputs = {u'y': y,
                      u'z': z_new,
                      u'truncation': truncation}
transformed_output = module(transformed_inputs)

# losses
loss = tf.losses.compute_weighted_loss(tf.square(
    transformed_output-target), weights=mask)

# train op 
# change to loss to loss_lpips to optimize lpips loss
train_step = tf.train.AdamOptimizer(lr).minimize(
    loss, var_list=tf.trainable_variables(scope=None), name='AdamOpter')

initializer = tf.global_variables_initializer()
config = tf.ConfigProto(log_device_placement=False)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(initializer)
saver = tf.train.Saver(tf.trainable_variables(scope=None))

def get_target_np(outputs_zs, alpha):
    
    mask_fn = np.ones(outputs_zs.shape)
    
    if alpha == 1:
        return outputs_zs, mask_fn
    
    new_size = int(alpha*img_size)

    ## crop
    if alpha < 1:
        output_cropped = outputs_zs[:,img_size//2-new_size//2:img_size//2+new_size//2, img_size//2-new_size//2:img_size//2+new_size//2,:]
        mask_cropped = mask_fn
    
    ## padding
    else:
        output_cropped = np.zeros((outputs_zs.shape[0], new_size, new_size, outputs_zs.shape[3]))
        mask_cropped = np.zeros((outputs_zs.shape[0], new_size, new_size, outputs_zs.shape[3]))
        output_cropped[:, new_size//2-img_size//2:new_size//2+img_size//2, new_size//2-img_size//2:new_size//2+img_size//2,:] = outputs_zs 
        mask_cropped[:, new_size//2-img_size//2:new_size//2+img_size//2, new_size//2-img_size//2:new_size//2+img_size//2,:] = mask_fn
    
    ## Resize
    target_fn = np.zeros(outputs_zs.shape)
    mask_out = np.zeros(outputs_zs.shape)
    for i in range(outputs_zs.shape[0]):
        target_fn[i,:,:,:] = cv2.resize(output_cropped[i,:,:,:], (img_size, img_size), interpolation = cv2.INTER_LINEAR)
        mask_out[i,:,:,:] = cv2.resize(mask_cropped[i,:,:,:], (img_size, img_size), interpolation = cv2.INTER_LINEAR)
        
    mask_out[np.nonzero(mask_out)] = 1.
    assert(np.setdiff1d(mask_out, [0., 1.]).size == 0)

    return target_fn, mask_out

    # define sampling operations
from graph_util import *

# This can be train.py

import logging
import sys
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.FileHandler("{0}/{1}.log".format(output_dir, 'train')),
        logging.StreamHandler(sys.stdout)
    ])
logger = logging.getLogger()

loss_vals = []


# train
def train(saver):
    trunc=1.0
    noise_seed=0
    zs = truncated_z_sample(num_samples, trunc, noise_seed)
    ys = np.random.randint(0,vocab_size,size=zs.shape[0])
    ys = one_hot_if_needed(ys, vocab_size)

    Loss_sum = 0
    n_epoch = 1
    Loss_sum_iter = 0
    optim_iter = 0
    batch_size = 4
    for epoch in range(n_epoch):
        for batch_start in range(0, num_samples, batch_size):
            start_time = time.time()

            coin = np.random.uniform(0, 1)
            if coin <= 0.5:
                alpha_val = np.random.uniform(0.25, 1.) 
            else:
                alpha_val = np.random.uniform(1., 4.) 

            s = slice(batch_start, min(num_samples, batch_start + batch_size))

            feed_dict_out = {z: zs[s], y: ys[s], truncation: trunc}
            out_zs = sess.run(outputs_orig, feed_dict_out)

            target_fn, mask_out = get_target_np(out_zs, alpha_val)
            

            alpha_val_for_graph = np.ones((zs[s].shape[0], Nsliders)) * np.log(alpha_val)
            feed_dict = {z: zs[s], y: ys[s], truncation: trunc, alpha: alpha_val_for_graph, target: target_fn, mask: mask_out}
            
            curr_loss, _ = sess.run([loss, train_step], feed_dict=feed_dict)
            Loss_sum = Loss_sum + curr_loss
            Loss_sum_iter = Loss_sum_iter + curr_loss
            
            elapsed_time = time.time() - start_time

            logger.info('T, epc, bst, lss, a: {}, {}, {}, {}, {}'.format(elapsed_time, epoch, batch_start, curr_loss, alpha_val))


            if (optim_iter % 4 == 0) and (optim_iter > 0):
                saver.save(sess, './{}/{}/model_{}.ckpt'.format(output_dir, 'output', optim_iter*batch_size), write_meta_graph=False, write_state=False)

            if (optim_iter % 4 == 0) and (optim_iter > 0):
                loss_vals.append(Loss_sum_iter/(4*batch_size))
                Loss_sum_iter = 0
                # print('Loss:', loss_vals)
                
            optim_iter = optim_iter+1
            
    if optim_iter > 0:
        print('average loss with this metric: ', Loss_sum/(optim_iter*batch_size))
    saver.save(sess, "./{}/{}/model_{}.ckpt".format(output_dir, 'output', optim_iter*batch_size), write_meta_graph=False, write_state=False)

## WAIT ######################
best_w = w.eval(sess)
# print('best_w before restore:', best_w)
print(best_w.shape)


train(saver)

from image import imgrid

# To restore previous w:
saver.restore(sess, "./{}/{}/model_{}.ckpt".format(output_dir, 'output', 16))
best_w = w.eval(sess)
# print('best_w at restore:', best_w)
print(best_w.shape)

# test: show imgs 
# this can be test.py

category = 207

a = np.array([8, 4, 2, 1, 0.5, 0.25, 0.125])

trunc = 0.5
noise_seed= 0  
num_samples_vis = 6
batch_size = 1

import matplotlib.pyplot as plt

zs = truncated_z_sample(num_samples_vis, trunc, noise_seed)
ys = np.array([category] * zs.shape[0])
ys = one_hot_if_needed(ys, vocab_size)

for batch_num, batch_start in enumerate(range(0, num_samples_vis, batch_size)):

    ims = []
    targets = []

    s = slice(batch_start, min(num_samples, batch_start + batch_size))

    input_test = {y: ys[s],
                  z: zs[s],
                  truncation: trunc}

    out_input_test = sess.run(outputs_orig, input_test)

    for i in range(a.shape[0]):
        target_fn, mask_out = get_target_np(out_input_test, a[i])
        
        alpha_val_for_graph = np.ones((zs[s].shape[0], Nsliders)) * np.log(a[i])
        
        best_inputs = {z: zs[s], y: ys[s], truncation: trunc, alpha: alpha_val_for_graph, target: target_fn, mask: mask_out}
        best_im_out = sess.run(transformed_output, best_inputs)
       
        # collect images
        ims.append(np.uint8(np.clip(((best_im_out + 1) / 2.0) * 256, 0, 255)))
        targets.append(np.uint8(np.clip(((target_fn + 1) / 2.0) * 256, 0, 255)))
        
    im_stack = np.concatenate(targets + ims).astype(np.uint8)
    plt.imshow(imgrid(im_stack, cols = len(a)))

    # plot losses 


plt.plot(loss_vals)
plt.xlabel('num samples, lr{}'.format(lr))
plt.ylabel('Loss')
plt.show()
