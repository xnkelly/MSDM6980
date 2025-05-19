import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()                  # 关掉 TF2 行为，回到 TF1.x
tf.reset_default_graph()
tf.disable_eager_execution()

# for reproducible
np.random.seed(1)
tf.set_random_seed(1)

from tensorflow.keras.layers import Dense

class PolicyGradient:
    def __init__(self,
                 n_features,
                 n_actions,
                 learning_rate=0.001,
                 reward_decay=0.99,
                 output_graph=False):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self._build_net()
        self.sess = tf.Session()
        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(
                tf.float32, [None, self.n_features], name="observations")
            self.tf_acts = tf.placeholder(
                tf.int32,   [None,],              name="actions_num")
            self.tf_vt   = tf.placeholder(
                tf.float32, [None,],              name="actions_value")

        # --- 用 Keras Dense 替代 tf.compat.v1.layers.dense ---
        layer = Dense(
            units=20,
            activation=tf.nn.tanh,
            name='fc1'
        )(self.tf_obs)

        all_act = Dense(
            units=self.n_actions,
            activation=None,
            name='fc2'
        )(layer)

        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')

        with tf.name_scope('loss'):
            neg_log_prob = tf.reduce_sum(
                -tf.log(self.all_act_prob) *
                tf.one_hot(self.tf_acts, self.n_actions),
                axis=1
            )
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def pro_choose_action(self, observation):
        prob_weights = self.sess.run(
            self.all_act_prob,
            feed_dict={self.tf_obs: observation})
        action = np.random.choice(
            range(prob_weights.shape[1]),
            p=prob_weights.ravel())
        return action

    def fix_choose_action(self, observation):
        prob_weights = self.sess.run(
            self.all_act_prob,
            feed_dict={self.tf_obs: observation})
        return np.argmax(prob_weights.ravel())

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        discounted_ep_rs_norm = self._discount_rewards()
        self.sess.run(
            self.train_op,
            feed_dict={
                self.tf_obs: np.vstack(self.ep_obs),
                self.tf_acts: np.array(self.ep_as),
                self.tf_vt:   np.array(discounted_ep_rs_norm)
            })
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        return discounted_ep_rs_norm

    def _discount_rewards(self):
        discounted = np.zeros_like(self.ep_rs, dtype=float)
        running_add = 0
        for t in reversed(range(len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted[t] = running_add
        discounted -= np.mean(discounted)
        discounted /= np.std(discounted) + 1e-8
        return discounted

    def save(self, checkpoint):
        saver = tf.train.Saver()
        saver.save(self.sess, checkpoint)

    def load(self, checkpoint):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint)
        if ckpt and ckpt.model_checkpoint_path:
            print('training from last checkpoint', checkpoint)
            saver.restore(self.sess, ckpt.model_checkpoint_path)

        reader = tf.train.NewCheckpointReader(checkpoint+'trained_model.ckpt')
        #Print tensor name and values
#        var_to_shape_map = reader.get_variable_to_shape_map()
#        for key in var_to_shape_map:
#            print("tensor_name: ", key)
#            print(reader.get_tensor(key))
        self.bias_1 = reader.get_tensor('fc1/bias')
        self.kernel_1 = reader.get_tensor('fc1/kernel')
        self.bias_2 = reader.get_tensor('fc2/bias')
        self.kernel_2 = reader.get_tensor('fc2/kernel')
    def softmax(self, x):
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def quick_time_action(self, observation):
        # 需在 load() 后拿到 kernel_/bias_ 才能用
        l1 = self.tanh(np.dot(observation, self.kernel_1) + self.bias_1)
        pro = self.softmax(np.dot(l1, self.kernel_2) + self.bias_2)
        return np.argmax(pro)
