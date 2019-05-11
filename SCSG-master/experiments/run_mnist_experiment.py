"""
This script runs the experiments for comparing SCSG and SGD on MNIST.
"""
from __future__ import absolute_import, division, print_function

import warnings
warnings.filterwarnings("ignore")

import math,argparse,time

import numpy as np 
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
import sys
sys.path.append('core') 
from core.scsg import SCSGOptimizer
from core.scsg_v2 import SCSGOptimizer as SCSGOptimizer_v2
from experiments.model import build_network, loss, training, training_ADAM, evaluation

from tqdm import tqdm


def run_single_experiment(method='sgd', model = 'cnn', batch_size = 1000, learning_rate = 0.001, ratio = 10, num_iterations = 400,  fix_batch = False, data = input_data.read_data_sets('MNIST_data', one_hot=True)):
    """
	Carries out a single experiment of training MNIST. It saves the  
	Args:
	method: str. optimization method to use: sgd or scsg.
	model: str. use cnn or fully connected network.
	batch_size: int. batch size for training. (Only used for fixed batchsize experiments.)
	learning rate: float. learning rate for training. 
	ratio: the ratio of batch size and mini-batch size. (Only for SCSG)
	num_iterations: int. Number of iterations for training.
	fix_batch: bool. If the batchsize is fixed or increasing according to the scheme in paper.  
	Return:
	None
    """
    tf.reset_default_graph()
    #mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    
    ## Create the Dataframe that we outputs to plot some results
    columns = ["iteration", "num_used_data", "batch_size", "update_time", "training_loss", "val_loss", "val_acc"]
    export_data = pd.DataFrame(data = np.zeros(shape =(num_iterations, len(columns))), columns =columns)
    
    mnist = data
	# Build placeholders and networks.
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    logits = build_network(x, model = model)
    batch_loss = loss(logits, y_)
    loss_op = tf.reduce_mean(batch_loss) 
    acc_op = evaluation(logits, y_)
    
    if method == 'sgd':
        train_op = training(loss_op, learning_rate)
    elif method == 'adam':
        train_op = training_ADAM(loss_op, learning_rate)
    elif method == 'scsg':
        optimizer = SCSGOptimizer(loss_op, learning_rate)
    elif method == 'scsg_v2':
        optimizer = SCSGOptimizer_v2(loss_op, learning_rate)
        
    num_used_data = 0

    with tf.Session() as sess:
        # sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        ttqqddmm = tqdm(range(num_iterations))
        
        for j in ttqqddmm:
            
            export_data.loc[j,"iteration"] = j
            # j=0
            if not fix_batch: 
                batch_size = int((j+1) ** 1.5) 

            mini_batchsize = max(1,int(batch_size / float(ratio))) # mini-batch size. 
            batch = mnist.train.next_batch(batch_size, shuffle=True)
            feed_dict = {x:batch[0], y_:batch[1]}
            
            t_start_j = time.time()
            if method == 'scsg' or method == 'scsg_v2':
                optimizer.batch_update(sess, feed_dict, batch_size, mini_batchsize, lr = 1.0/(j+1) if not fix_batch else None) 
            elif method == 'sgd' or method == 'adam':
                sess.run(train_op, feed_dict = feed_dict) 
            t_stop_j = time.time()
            
            num_used_data += batch_size
            export_data.loc[j,"num_used_data"] = num_used_data
            export_data.loc[j,"batch_size"] = batch_size
            export_data.loc[j,"update_time"] = t_stop_j - t_start_j
            
            samples = np.random.choice(range(mnist.train._num_examples), size = 10000, replace = False)
            train_loss_val = sess.run(loss_op, feed_dict = {x: mnist.train.images[samples],  y_: mnist.train.labels[samples]})
            val_loss_val, acc_val = sess.run([loss_op,acc_op], feed_dict = {x: mnist.validation.images,  y_: mnist.validation.labels}) 
            
            export_data.loc[j,"training_loss"] = train_loss_val
            export_data.loc[j,"val_loss"] = val_loss_val
            export_data.loc[j,"val_acc"] = acc_val
    
            train_loss_val, val_loss_val, acc_val = round(train_loss_val,3), round(val_loss_val,3), round(acc_val,3)
            
        
            ttqqddmm.set_description('# data used: {} Train loss: {} Val loss: {} Val Accuracy: {}'.format(num_used_data, train_loss_val, val_loss_val, acc_val))
            if j == num_iterations //2 :
                print("\n")
    return export_data



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',type = str,default = 'fc', choices = ['cnn', 'fc']) 
    parser.add_argument('--method',type = str,default = 'scsg', choices = ['scsg', 'sgd']) 
    parser.add_argument('--batch_size',type=int, default = 1024)
    parser.add_argument('--num_iterations',type=int, default = 400) 
    parser.add_argument('--learning_rate',type=float, default = 0.001)
    parser.add_argument('--ratio',type=int,default=32) 
    parser.add_argument('--fix_batch', action='store_true')
    args = parser.parse_args() 

    data = run_single_experiment(method = 'scsg_v2',#args.method, 
                        		model = args.model, 
                        		batch_size = args.batch_size,
                        		learning_rate = 0.01, # args.learning_rate, 
                        		ratio = args.ratio,
                        		num_iterations = args.num_iterations,
                        		fix_batch = args.fix_batch,
                        		) 

if __name__ == '__main__': 
    main()
