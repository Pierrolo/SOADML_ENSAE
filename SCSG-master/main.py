# -*- coding: utf-8 -*-
"""
Created on Wed May  1 15:53:20 2019

@author: woill
"""
import warnings
warnings.filterwarnings("ignore")



import tensorflow as tf
import numpy as np
tf.set_random_seed(seed = 1234)
np.random.seed(seed = 1234)
import pandas as pd
from tensorflow.examples.tutorials.mnist import input_data
from experiments.run_mnist_experiment import run_single_experiment
from run_kc_hous_price_expreiment import run_single_experiment_reg, kc_house_set

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

# Set the font dictionaries (for plot title and axis titles)
title_font = {'fontname':'Arial', 'size':'16', 'color':'black', 'weight':'bold', 'verticalalignment':'bottom'} # Bottom vertical alignment for more space
axis_font = {'fontname':'Arial', 'size':'14'}

# Set the font properties (for use in legend)   
font_path = 'C:\Windows\Fonts\Arial.ttf'
font_prop = font_manager.FontProperties(fname=font_path, size=14)


# =============================================================================
# =============================================================================
# # CLASSIFICATION
# =============================================================================
# =============================================================================
batch_size = 1024
num_iterations = 150
learning_rate = 0.01
ratio = 32
#fix_batch = True
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

results = {}

for fix_batch in [True, False]:
    for model in  ['cnn', 'fc'] :
        for method in ['adam', 'scsg_v2', 'scsg', 'sgd']:
            tf.reset_default_graph()
            print("\n\nModel :",model," || Method :", method," || Batch Size Fixed :", fix_batch)
            results[str(model) + "_" + str(method) + "_" + str(fix_batch)] = run_single_experiment(method = method, 
                                                                                                    model = model, 
                                                                                            		batch_size = batch_size,
                                                                                            		learning_rate = learning_rate, 
                                                                                            		ratio = ratio,
                                                                                            		num_iterations = num_iterations,
                                                                                            		fix_batch = fix_batch,
                                                                                                    data = mnist
                                                                                            		) 




# =============================================================================
# SCSG V1 vs V2
# =============================================================================
fixed_batch = True
plt.figure(figsize=(20,12))

plt.subplot(221)
plt.plot(results["fc_scsg_"+str(fixed_batch)].val_acc, label ="SCSG")
plt.plot(results["fc_scsg_v2_"+str(fixed_batch)].val_acc, label ="SCSG V2")
plt.title("FC", **title_font)
plt.xlabel("Iterations", **axis_font); plt.ylabel("Acuracy", **axis_font); plt.legend()

plt.subplot(222)
plt.plot(results["cnn_scsg_"+str(fixed_batch)].val_acc, label ="SCSG")
plt.plot(results["cnn_scsg_v2_"+str(fixed_batch)].val_acc, label ="SCSG V2")
plt.title("CNN", **title_font)
plt.xlabel("Iterations", **axis_font); plt.ylabel("Acuracy", **axis_font); plt.legend()

plt.subplot(223)
plt.plot(results["fc_scsg_"+str(fixed_batch)].val_loss, label ="SCSG")
plt.plot(results["fc_scsg_v2_"+str(fixed_batch)].val_loss, label ="SCSG V2")
plt.title("FC", **title_font)
plt.xlabel("Iterations", **axis_font); plt.ylabel("Loss", **axis_font); plt.legend()

plt.subplot(224)
plt.plot(results["cnn_scsg_"+str(fixed_batch)].val_loss, label ="SCSG")
plt.plot(results["cnn_scsg_v2_"+str(fixed_batch)].val_loss, label ="SCSG V2")
plt.title("CNN", **title_font)
plt.xlabel("Iterations", **axis_font); plt.ylabel("Loss", **axis_font); plt.legend()

plt.show()


# =============================================================================
# SCSG vs SGD vs ADAM Fixed Batch Size
# =============================================================================
plt.figure(figsize=(20,18))

## Validation Accuracy
plt.subplot(321)
for name in list(results.keys()):
    if name[-4:] == 'True' and name[:3] == 'fc_':
        plt.plot(results[name].val_acc, label =name[3:-5])
plt.title("FC", **title_font)
plt.xlabel("Iterations", **axis_font); plt.ylabel("Val Accuracy", **axis_font); plt.legend()

plt.subplot(322)
for name in list(results.keys()):
    if name[-4:] == 'True' and name[:3] == 'cnn':
        plt.plot(results[name].val_acc, label =name[4:-5])
plt.title("CNN", **title_font)
plt.xlabel("Iterations", **axis_font); plt.ylabel("Val Accuracy", **axis_font); plt.legend()

## Validation Loss
plt.subplot(323)
for name in list(results.keys()):
    if name[-4:] == 'True' and name[:3] == 'fc_':
        plt.plot(np.log(results[name].val_loss), label =name[3:-5])
plt.title("FC", **title_font)
plt.xlabel("Iterations", **axis_font); plt.ylabel("$\log($Val Loss$)$", **axis_font); plt.legend()

plt.subplot(324)
for name in list(results.keys()):
    if name[-4:] == 'True' and name[:3] == 'cnn':
        plt.plot(np.log(results[name].val_loss), label =name[4:-5])
plt.title("CNN", **title_font)
plt.xlabel("Iterations", **axis_font); plt.ylabel("$\log($Val Loss$)$", **axis_font); plt.legend()

## Train Loss
plt.subplot(325)
for name in list(results.keys()):
    if name[-4:] == 'True' and name[:3] == 'fc_':
        plt.plot(np.log(results[name].training_loss), label =name[3:-5])
plt.title("FC", **title_font)
plt.xlabel("Iterations", **axis_font); plt.ylabel("$\log($Train Loss$)$", **axis_font); plt.legend()

plt.subplot(326)
for name in list(results.keys()):
    if name[-4:] == 'True' and name[:3] == 'cnn':
        plt.plot(np.log(results[name].training_loss), label =name[4:-5])
plt.title("CNN", **title_font)
plt.xlabel("Iterations", **axis_font); plt.ylabel("$\log($Train Loss$)$", **axis_font); plt.legend()

plt.show()



## TIME COMPARISON
plt.figure(figsize=(20,5))

## Validation Accuracy
plt.subplot(121)
for name in list(results.keys()):
    if name[-4:] == 'True' and name[:3] == 'fc_':
        plt.plot(results[name].update_time.cumsum(), results[name].val_acc, label =name[3:-5])
plt.title("FC", **title_font)
plt.xlabel("Wall Clock Time (in sec)", **axis_font); plt.ylabel("Val Accuracy", **axis_font); plt.legend()

plt.subplot(122)
for name in list(results.keys()):
    if name[-4:] == 'True' and name[:3] == 'cnn':
        plt.plot(results[name].update_time.cumsum(), results[name].val_acc, label =name[4:-5])
plt.title("CNN", **title_font)
plt.xlabel("Wall Clock Time (in sec)", **axis_font); plt.ylabel("Val Accuracy", **axis_font); plt.legend()

plt.show()


# =============================================================================
# Fixed VS Varying Batch Size
# =============================================================================
plt.figure(figsize=(25,5))
name = 'cnn_scsg_v2_False'
plt.subplot(131)
plt.plot(results[name].batch_size, label = "Varying")
plt.plot(1024 * np.ones(shape = (len(results[name].batch_size))), label = "Fixed")
plt.xlabel("Iteration", **axis_font); plt.ylabel("Batch Size", **axis_font); plt.legend()
plt.title("Evolution of Batch Size in training procedure", **title_font)


plt.subplot(132)
plt.plot(results['cnn_scsg_v2_False'].num_used_data, label = "Varying")
plt.plot(results['cnn_scsg_v2_True'].num_used_data, label = "Fixed")
plt.xlabel("Iteration", **axis_font); plt.ylabel("Number of data used", **axis_font); plt.legend()
plt.title("Evolution of data used in training procedure", **title_font)

plt.subplot(133)
learning_rates = [1.0/(j+1) for j in range(len(results['cnn_scsg_v2_False'].num_used_data))]
plt.plot(learning_rates, label = "Varying")
plt.plot(0.01 * np.ones(shape = (len(results[name].batch_size))), label = "Fixed")
plt.xlabel("Iteration", **axis_font); plt.ylabel("$\eta_j$", **axis_font); plt.legend()
plt.title("Evolution of learning rate in training procedure", **title_font)


plt.show()


plt.figure(figsize=(20,18))

## val acc
plt.subplot(321)
for name in list(results.keys()):
    if name[:3] == 'fc_' and 'scsg' in name and 'v2' in name:
        size_name = " Fixed Size" if name[-4:] == 'True' else " Varying Size"
        plt.plot(results[name].val_acc, label = size_name)
plt.title("FC", **title_font)
plt.xlabel("Iteration", **axis_font); plt.ylabel("Val Accuracy", **axis_font); plt.legend()

plt.subplot(322)
for name in list(results.keys()):
    if name[:3] == 'cnn' and 'scsg' in name and 'v2' in name:
        size_name = " Fixed Size" if name[-4:] == 'True' else " Varying Size"
        plt.plot(results[name].val_acc, label = size_name)
plt.title("CNN", **title_font)
plt.xlabel("Iteration", **axis_font); plt.ylabel("Val Accuracy", **axis_font); plt.legend()


## val loss
plt.subplot(323)
for name in list(results.keys()):
    if name[:3] == 'fc_' and 'scsg' in name and 'v2' in name:
        size_name = " Fixed Size" if name[-4:] == 'True' else " Varying Size"
        plt.plot(np.log(results[name].val_loss), label = size_name)
plt.title("FC", **title_font)
plt.xlabel("Iteration", **axis_font); plt.ylabel("$\log($Val Loss$)$", **axis_font); plt.legend()

plt.subplot(324)
for name in list(results.keys()):
    if name[:3] == 'cnn' and 'scsg' in name and 'v2' in name:
        size_name = " Fixed Size" if name[-4:] == 'True' else " Varying Size"
        plt.plot(np.log(results[name].val_loss), label = size_name)
plt.title("CNN", **title_font)
plt.xlabel("Iteration", **axis_font); plt.ylabel("$\log($Val Loss$)$", **axis_font); plt.legend()


## val acc
plt.subplot(325)
for name in list(results.keys()):
    if name[:3] == 'fc_' and 'scsg' in name and 'v2' in name:
        size_name = " Fixed Size" if name[-4:] == 'True' else " Varying Size"
        plt.plot(results[name].num_used_data, results[name].val_acc, label = size_name)
plt.title("FC", **title_font)
plt.xlabel("Total number of used observations", **axis_font); plt.ylabel("Val Accuracy", **axis_font); plt.legend()

plt.subplot(326)
for name in list(results.keys()):
    if name[:3] == 'cnn' and 'scsg' in name and 'v2' in name:
        size_name = " Fixed Size" if name[-4:] == 'True' else " Varying Size"
        plt.plot(results[name].num_used_data, results[name].val_acc, label = size_name)
plt.title("CNN", **title_font)
plt.xlabel("Total number of used observations", **axis_font); plt.ylabel("Val Accuracy", **axis_font); plt.legend()



plt.show()



# =============================================================================
# =============================================================================
# # REGRESSION
# =============================================================================
# =============================================================================


data = kc_house_set()
results_reg = {}

batch_size = 1024
num_iterations = 150
learning_rate = 0.01
ratio = 32

for fix_batch in [True, False]:
    for method in ['adam', 'scsg_v2', 'scsg', 'sgd']:
        tf.reset_default_graph()
        print("\n\n Method :", method," || Batch Size Fixed :", fix_batch)
        results_reg[str(method) + "_" + str(fix_batch)] = run_single_experiment_reg(method = method, 
                                                                            		batch_size = batch_size,
                                                                            		learning_rate = learning_rate, 
                                                                            		ratio = ratio,
                                                                            		num_iterations = num_iterations,
                                                                            		fix_batch = fix_batch,
                                                                                    data = data
                                                                            		) 


## TIME COMPARISON
plt.figure(figsize=(20,12))

## LOSS by iteration
plt.subplot(221)
for name in list(results_reg.keys()):
    if name[-4:] == 'True':
        plt.plot(np.log(results_reg[name].val_loss), label =name[:-5])
plt.title("Validation Set", **title_font)
plt.xlabel("Iteration", **axis_font); plt.ylabel("$\log($Loss$)$", **axis_font); plt.legend()

plt.subplot(222)
for name in list(results_reg.keys()):
    if name[-4:] == 'True':
        plt.plot(np.log(results_reg[name].training_loss), label =name[:-5])
plt.title("Training Set", **title_font)
plt.xlabel("Iteration", **axis_font); plt.ylabel("$\log($Loss$)$", **axis_font); plt.legend()


## LOSS by TIME
plt.subplot(223)
for name in list(results_reg.keys()):
    if name[-4:] == 'True':
        plt.plot(results_reg[name].update_time.cumsum(), np.log(results_reg[name].val_loss), label =name[:-5])
plt.title("Validation Set", **title_font)
plt.xlabel("Wall clock (in sec)", **axis_font); plt.ylabel("$\log($Loss$)$", **axis_font); plt.legend()

plt.subplot(224)
for name in list(results_reg.keys()):
    if name[-4:] == 'True':
        plt.plot(results_reg[name].update_time.cumsum(), np.log(results_reg[name].training_loss), label =name[:-5])
plt.title("Training Set", **title_font)
plt.xlabel("Wall clock (in sec)", **axis_font); plt.ylabel("$\log($Loss$)$", **axis_font); plt.legend()


plt.show()



# =============================================================================
# Fixed Vs Varying
# =============================================================================
plt.figure(figsize=(20,12))

## LOSS by iteration
plt.subplot(221)
for name in list(results_reg.keys()):
    if 'scsg_v2' in name:
        size_name = " Fixed Size" if name[-4:] == 'True' else " Varying Size"
        plt.plot(np.log(results_reg[name].val_loss), label =size_name)
plt.title("Validation Set", **title_font)
plt.xlabel("Iteration", **axis_font); plt.ylabel("$\log($Loss$)$", **axis_font); plt.legend()

plt.subplot(222)
for name in list(results_reg.keys()):
    if 'scsg_v2' in name:
        size_name = " Fixed Size" if name[-4:] == 'True' else " Varying Size"
        plt.plot(np.log(results_reg[name].training_loss), label =size_name)
plt.title("Training Set", **title_font)
plt.xlabel("Iteration", **axis_font); plt.ylabel("$\log($Loss$)$", **axis_font); plt.legend()

plt.subplot(223)
for name in list(results_reg.keys()):
    if 'scsg_v2' in name:
        size_name = " Fixed Size" if name[-4:] == 'True' else " Varying Size"
        plt.plot(results_reg[name].num_used_data, np.log(results_reg[name].val_loss), label =size_name)
plt.title("Validation Set", **title_font)
plt.xlabel("Iteration", **axis_font); plt.ylabel("Total number of used data", **axis_font); plt.legend()

plt.subplot(224)
for name in list(results_reg.keys()):
    if 'scsg_v2' in name:
        size_name = " Fixed Size" if name[-4:] == 'True' else " Varying Size"
        plt.plot(results_reg[name].num_used_data,np.log(results_reg[name].training_loss), label =size_name)
plt.title("Training Set", **title_font)
plt.xlabel("Iteration", **axis_font); plt.ylabel("Total number of used data", **axis_font); plt.legend()


plt.show()

















