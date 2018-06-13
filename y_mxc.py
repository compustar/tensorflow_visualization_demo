import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--addnoise', dest='add_noise', action='store_true')
parser.add_argument('--m', type=int, default=10)
parser.add_argument('--c',  type=int, default=-15)
parser.add_argument('--learning_rate',  type=float, default=0.01)
parser.add_argument('--steps',  type=int, default=500)

parsed_args = parser.parse_args()

tf.set_random_seed(1)
np.random.seed(1)

train_x = np.array(range(-4, 4))
noise = np.zeros(train_x.shape)
if parsed_args.add_noise: noise = np.random.normal(-2, 2, size=train_x.shape)
train_m = parsed_args.m
train_c = parsed_args.c
truth_y = train_x * train_m + train_c
train_y = train_x * train_m + train_c + noise

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
m = tf.Variable([0.])
c = tf.Variable([0.])
model = x * m + c

loss = tf.losses.mean_squared_error(y, model)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=parsed_args.learning_rate)

train_op = optimizer.minimize(loss)

plt.figure(figsize=(12, 4))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    steps = []
    train_losses = []
    for step in range(parsed_args.steps):
        # train
        _, l, pred_y, pred_m, pred_c = sess.run([train_op, loss, model, m, c], {x: train_x, y: train_y})
        
        steps.append(step)
        train_losses.append(l)        
        if step % 10 == 9 or step == 0:
            # visualize
            plt.clf()
            plt.subplot(1, 2, 1)
            plt.scatter(train_x, train_y)
            plt.plot(train_x, truth_y, 'b-', lw=5)
            plt.plot(train_x, pred_y, 'r-', lw=3)
            max_y = max(train_y)
            min_y = min(train_y)
            line_space = abs(max_y - min_y) / 10
            min_x = min(train_x)
            plt.text(min_x, max_y, f'train_c: {train_c}, pred_c={pred_c[0]:3f}', fontdict={'size': 10, 'color': 'black'})
            plt.text(min_x, max_y - line_space * 1, f'train_m: {train_m}, pred_m={pred_m[0]:3f}', fontdict={'size': 10, 'color': 'black'})

            plt.subplot(1, 2, 2)
            plt.plot(steps, train_losses, 'g-')
            max_y = max(train_losses)
            plt.text(0, max_y, f'step: {step + 1}', fontdict={'size': 10, 'color': 'black'})
            plt.text(0, max_y * 0.9, f'train_loss={l:10f}', fontdict={'size': 10, 'color': 'black'})
            plt.pause(0.01)

plt.show()
