import numpy as np
import tensorflow as tf
import gym


def discount_reward(r):
    discount_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(r.size)):
        running_add = running_add * gamma + r[t]
        discount_r[t] += running_add
    return discount_r


env = gym.make("CartPole-v1")
env.reset()
reward_sum = 0
random_episodes = 0
H = 50
batch_size = 25
learning_rate = 1e-1
D = 4
gamma = 0.99
observations = tf.placeholder(tf.float32, [None, D], name="input_x")
w1 = tf.get_variable(name="w1", shape=[D, H])
layer1 = tf.nn.relu(tf.matmul(observations, w1))
w2 = tf.get_variable(name="w2", shape=[H, 1])
score = tf.matmul(layer1, w2)
probability = tf.nn.sigmoid(score)

adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
W1Grad = tf.placeholder(tf.float32, name="batch_grad1")
W2Grad = tf.placeholder(tf.float32, name="batch_grad2")
batchGrad = [W1Grad, W2Grad]
tvars = tf.trainable_variables()
updateGrads = adam.apply_gradients(zip(batchGrad, tvars))
input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
advantages = tf.placeholder(tf.float32, name="reward_signal")
loglik = tf.log(input_y * (input_y - probability) + (1 - input_y) * probability)
loss = -tf.reduce_mean(loglik * advantages)
newGrads = tf.gradients(loss, tvars)
xs, ys, drs = [], [], []
reward_sum = 0
episode_number = 1
total_episodes = 100000

with tf.Session() as sess:
    rendering = False;
    init = tf.global_variables_initializer()
    sess.run(init)
    observation = env.reset()
    gradBuffer = sess.run(tvars)
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0


    while episode_number < total_episodes:

        if reward_sum / batch_size > 10000 or rendering == True:
            env.render()
            rendering = True
        x = np.reshape(observation, [1, D])
        tfprob = sess.run(probability, feed_dict={observations: x})
        action = 1 if np.random.uniform() < tfprob else 0
        xs.append(x)
        y = 1 - action
        ys.append(y)
        observation, reward, done, _ = env.step(action)
        reward_sum += reward
        drs.append(reward)

        if done:
            episode_number += 1
            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epdrs = np.vstack(drs)
            xs, ys, drs = [], [], []
            discounted_epr = discount_reward(epdrs)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)
            tGrad = sess.run(newGrads, feed_dict={observations: epx, input_y: epy, advantages: discounted_epr})
            for ix, grad in enumerate(tGrad):
                gradBuffer[ix] += grad

            if episode_number % batch_size == 0:
                sess.run(updateGrads, feed_dict={W1Grad: gradBuffer[0], W2Grad: gradBuffer[1]})
                print('Average reward for episode %d : %f.' % (episode_number, reward_sum / batch_size))
                for ix, grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0


            if reward_sum / batch_size > 100000:
                print("task solved in ", episode_number, 'episodes')
                break
                reward_sum = 0

            observation = env.reset()
