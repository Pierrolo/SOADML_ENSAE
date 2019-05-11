# -*- coding: utf-8 -*-
"""
Created on Fri May 10 22:06:07 2019

@author: woill
"""
from __future__ import absolute_import, division, print_function

import warnings
warnings.filterwarnings("ignore")

import time

import numpy as np 
import pandas as pd
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import sys
sys.path.append('core') 
from core.scsg import SCSGOptimizer
from core.scsg_v2 import SCSGOptimizer as SCSGOptimizer_v2
from experiments.model import build_network_reg, reg_loss, training, training_ADAM, evaluation_reg

from tqdm import tqdm

import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)


from scipy import stats
from scipy.stats import norm, skew #for some statistics
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points




class kc_house_set():
    """
    self = kc_house_set()
    """
    def __init__(self):
        house_price = pd.read_csv('./data/kc_house_data.csv')
        house_price = house_price[['price', 'bedrooms', 'bathrooms', 'sqft_living',
                                   'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
                                   'sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15']]
        self.x_col = ['bedrooms', 'bathrooms', 'sqft_living',
                      'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
                      'sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15']
        self.y_col = ['price']
        house_price.shape
        #We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
        house_price["price"] = np.log1p(house_price["price"])
        
        def norm_col(col_name):
            return house_price[col_name]/house_price[col_name].max()
        
        for col_name in self.x_col:
            house_price[col_name] = norm_col(col_name)
            
        idxes = np.random.choice(a = range(house_price.shape[0]), size= house_price.shape[0], replace = False)
        self.idxes_train = idxes[:int(house_price.shape[0]*(1-0.2))]
        self.idxes_test = idxes[int(house_price.shape[0]*(1-0.2)):]
        
        self.train_set = house_price.iloc[self.idxes_train]
        self.train_set.reset_index(drop = True)
        self.train_num_examples = self.train_set.shape[0]
        self.validation_set = house_price.iloc[self.idxes_test]
        self.validation_set.reset_index(drop = True)
        self.validation_num_examples = self.validation_set.shape[0]
        
        self.train_x = self.train_set[self.x_col]
        self.train_y = self.train_set[self.y_col]
        
        self.validation_x = self.validation_set[self.x_col]
        self.validation_y = self.validation_set[self.y_col]
        
        self.input_shape = self.train_x.shape[1]
        
    def next_batch_train(self, batch_size= 1024):
        idx = np.random.choice(a = range(self.train_num_examples), size= batch_size, replace = False)
        to_return = [self.train_set.iloc[idx][self.x_col],
                     self.train_set.iloc[idx][self.y_col]]
        return to_return

data = kc_house_set()

def run_single_experiment_reg(method='sgd', batch_size = 1024, learning_rate = 0.001, ratio = 32, num_iterations = 400,  fix_batch = False, data = input_data.read_data_sets('MNIST_data', one_hot=True)):
    """
	Carries out a single experiment of training MNIST. It saves the  
	Args:
	method: str. optimization method to use: sgd or scsg.
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
    columns = ["iteration", "num_used_data", "batch_size", "update_time", "training_loss", "val_loss"]
    export_data = pd.DataFrame(data = np.zeros(shape =(num_iterations, len(columns))), columns =columns)
    
	# Build placeholders and networks.
    x = tf.placeholder(tf.float32, shape=[None, data.input_shape])
    y_ = tf.placeholder(tf.float32, shape=[None, 1])
    logits = build_network_reg(x)
    batch_loss = reg_loss(logits, y_)
    loss_op = tf.reduce_mean(batch_loss) 
    
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
            batch = data.next_batch_train(batch_size)
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
            
            samples = np.random.choice(range(data.train_num_examples), size = 10000, replace = False)
            train_loss_val = sess.run(loss_op, feed_dict = {x: data.train_x.iloc[samples],  y_: data.train_y.iloc[samples]})
            val_loss_val = sess.run(loss_op, feed_dict = {x: data.validation_x,  y_: data.validation_y}) 
            
            export_data.loc[j,"training_loss"] = train_loss_val
            export_data.loc[j,"val_loss"] = val_loss_val
    
            train_loss_val, val_loss_val = round(train_loss_val,3), round(val_loss_val,3)
            
        
            ttqqddmm.set_description('# data used: {} Train loss: {} Val loss: {}'.format(num_used_data, train_loss_val, val_loss_val))
            if j == num_iterations //2 :
                print("\n")
    return export_data


if __name__ == '__main__': 
    data = kc_house_set()

    results = run_single_experiment_reg(method='scsg_v2',
                                          batch_size = 1024, 
                                          learning_rate = 0.0001, 
                                          ratio = 32, 
                                          num_iterations = 50,  
                                          fix_batch = True, 
                                          data = data)



    
    # =============================================================================
    # DATA Preparation
    # =============================================================================
    
    house_price = pd.read_csv('./data/kc_house_data.csv')
    house_price = house_price[['price', 'bedrooms', 'bathrooms', 'sqft_living',
                           'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
                           'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'sqft_living15', 'sqft_lot15']]
    house_price.shape
    #We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
    house_price["price"] = np.log1p(house_price["price"])
    #Check the new distribution 
    sns.distplot(house_price['price'] , fit=norm);
    # Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(house_price['price'])
    print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
    #Now plot the distribution
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
    plt.ylabel('Frequency')
    plt.title('price distribution')
    #Get also the QQ-plot
    fig = plt.figure()
    res = stats.probplot(house_price['price'], plot=plt)
    plt.show()
    
    
    all_data_na = (house_price.isnull().sum() / len(house_price)) * 100
    all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
    missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
    missing_data.head(20)
    ## No missing data, it's good
    
    idxes = np.random.choice(a = range(house_price.shape[0]), size= house_price.shape[0], replace = False)
    idxes_train = idxes[:int(house_price.shape[0]*(1-0.2))]
    idxes_test = idxes[int(house_price.shape[0]*(1-0.2)):]




