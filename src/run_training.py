# setup
import argparse
import copy
import os
import sys

import dnnlib
from dnnlib import EasyDict
import numpy as np
from metrics.metric_defaults import metric_defaults


ganformer=True
duplex=False
attention_discriminator=False
#tfrecords_dataset = 'FFHQ'
tfrecords_dataset = 'Cartoon'
img_resolution = 64 if tfrecords_dataset == 'Cartoon' else 128 #Resolution of the Image default 64 for Cartoon 128 for FFHQ



base_2_log = int(np.log2(img_resolution))
dataset = 'custom'
# data_dir = '/content/datasets/' if tfrecords_dataset == 'Cartoon' else '/content/TFRecords_FFHQ/'

# train with mnist dataset
data_dir = '/content/gansformer-reproducibility-challenge/src'
img_resolution = 32

num_gpus = 1
total_kimg = 300
mirror_augment = True
metrics = ['fid50k', 'is50k','pr50k3'] #' 
metrics_10k = ['fid10k']
gamma = None
tick_size=16
result_dir = '/content/drive/MyDrive/model'
if ganformer:
    result_dir = '/content/drive/MyDrive/GANFORMER_Duplex/' if duplex else '/content/drive/MyDrive/GANFORMER_Simplex/'
else:
    result_dir = '/content/drive/MyDrive/STYLEGAN2/'
#----------------------------------------------------------------------------

train     = EasyDict(run_func_name='training.training_loop.training_loop') # Options for training loop.
sched     = EasyDict()                                                     # Options for TrainingSchedule.
grid      = EasyDict(size='1080p', layout='random')                           # Options for setup_snapshot_image_grid().
sc        = dnnlib.SubmitConfig()                                          # Options for dnnlib.submit_run().
tf_config = {'rnd.np_random_seed': 1000}                                   # Options for tflib.init_tf().


if ganformer:
  G = EasyDict(func_name='training.networks_GANFormer.G_GANformer', truncation_psi = 0.65, 
               architecture = 'resnet', latent_size = 32, 
               dlatent_size = 32, components_num = 16, 
               mapping_resnet = True, style = True, 
               fused_modconv = True, local_noise = True, 
               transformer = True, norm = 'layer', 
               integration = 'mul', kmeans = duplex, 
               kmeans_iters = 1, mapping_ltnt2ltnt = True, 
               use_pos = True, num_heads = 2, 
               pos_init = 'uniform', pos_directions_num = 2, 
               merge_layer = -1, start_res = 0, 
               end_res = base_2_log, img2img = 0, 
               style_mixing = 0.9, component_mixing = 0.0, 
               component_dropout = 0.0)       # Options for generator network.

  if attention_discriminator:
    func_name='training.networks_GANFormer.D_GANformer'
  else:
    func_name='training.networks_GANFormer.D_Stylegan'
    
  D = EasyDict(func_name=func_name, latent_size = 32,
               components_num = 16, mbstd_group_size = 4, 
               use_pos = True, num_heads = 2, 
               pos_init = 'uniform', pos_directions_num = 2, 
               start_res = 0, end_res = base_2_log, img2img = 0)  # Options for discriminator network.
  G_loss = EasyDict(func_name='training.loss.G_logistic_ns_pathreg')      # Options for generator loss.
  D_loss = EasyDict(func_name='training.loss.D_logistic_r1')              # Options for discriminator loss.
  # G_opt = EasyDict(beta1=0.9, beta2=0.999, epsilon=1e-3)                  # Options for generator optimizer.
  # D_opt = EasyDict(beta1=0.9, beta2=0.999, epsilon=1e-3)                  # Options for discriminator optimizer.
  G_opt = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)                  # Options for generator optimizer.
  D_opt = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)                  # Options for discriminator optimizer.
  desc = 'GANFormer'
else:
  G = EasyDict(func_name='training.networks_stylegan2.G_main')       # Options for generator network.
  D = EasyDict(func_name='training.networks_stylegan2.D_stylegan2')  # Options for discriminator network.
  G_loss = EasyDict(func_name='training.loss_stylegan2.G_logistic_ns_pathreg')      # Options for generator loss.
  D_loss = EasyDict(func_name='training.loss_stylegan2.D_logistic_r1')              # Options for discriminator loss.
  G_opt = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)                  # Options for generator optimizer.
  D_opt = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)                  # Options for discriminator optimizer.
  desc = 'stylegan2'



train.data_dir = data_dir
train.total_kimg = total_kimg
train.mirror_augment = mirror_augment
train.image_snapshot_ticks = train.network_snapshot_ticks = 1
sched.G_lrate_base = sched.D_lrate_base = 0.002
sched.minibatch_size_base = 24
sched.minibatch_gpu_base = 12
D_loss.gamma = 10
metrics = [metric_defaults[x] for x in metrics]
metrics_10k = [metric_defaults[x] for x in metrics_10k]


desc += '-' + dataset
dataset_args = EasyDict(tfrecord_dir=dataset, resolution=img_resolution)

assert num_gpus in [1, 2, 4, 8]
sc.num_gpus = num_gpus
desc += '-%dgpu' % num_gpus

if gamma is not None:
    D_loss.gamma = gamma

sc.submit_target = dnnlib.SubmitTarget.LOCAL
sc.local.do_not_copy_source_files = True

#----------------------------------------------------------------------------
kwargs = EasyDict(train)
kwargs.update(G_args=G, D_args=D, G_opt_args=G_opt, D_opt_args=D_opt, G_loss_args=G_loss, D_loss_args=D_loss)
kwargs.update(dataset_args=dataset_args, sched_args=sched, grid_args=grid, metric_arg_list=metrics, metrics_10k_arg_list=metrics_10k, tf_config=tf_config,tick_size=tick_size, ganformer=ganformer,resume_pkl=None,resume_kimg =300)
kwargs.submit_config = copy.deepcopy(sc)
kwargs.submit_config.run_dir_root = result_dir
kwargs.submit_config.run_desc = desc
dnnlib.submit_run(**kwargs)