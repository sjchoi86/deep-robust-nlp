import tensorflow as tf
import tensorflow.contrib.slim as slim
from sklearn.utils import shuffle
from util import load_directory_data,load_dataset,download_and_load_datasets,mixup,gpusession,print_n_txt

class mlp_cls_class(object):
    def __init__(self,_name='mlp_cls',_x_dim=128,_t_dim=2,_h_dims=[64,64],
                _actv=tf.nn.relu,_bn=slim.batch_norm,_l2_reg_coef=1e-5,_momentum = 0.5,
                _USE_SGD=False,_USE_MIXUP=False,_GPU_ID=0,_VERBOSE=True):
        self.name = _name 
        self.x_dim = _x_dim
        self.t_dim = _t_dim
        self.h_dims = _h_dims
        self.actv = _actv
        self.bn = _bn
        self.l2_reg_coef = _l2_reg_coef
        self.momentum = _momentum
        self.USE_SGD = _USE_SGD
        self.USE_MIXUP = _USE_MIXUP
        self.GPU_ID = _GPU_ID
        self.VERBOSE = _VERBOSE
        if self.GPU_ID < 0:
            # Build model
            self.build_model()
            # Build graph
            self.build_graph()
            # Check parameters
            self.check_params()
        else:
            with tf.device('/device:GPU:%d'%(self.GPU_ID)):
                # Build model
                self.build_model()
                # Build graph
                self.build_graph()
                # Check parameters
                self.check_params()
        
    def build_model(self):
        self.x = tf.placeholder(dtype=tf.float32,shape=[None,self.x_dim]) # Input [N x xdim]
        self.t = tf.placeholder(dtype=tf.float32,shape=[None,self.t_dim]) # Output [N x D]
        self.kp = tf.placeholder(dtype=tf.float32,shape=[]) # []
        self.is_training = tf.placeholder(dtype=tf.bool,shape=[]) # []
        self.lr = tf.placeholder(dtype=tf.float32,shape=[]) # []
        self.fully_init  = tf.random_normal_initializer(stddev=0.01)
        self.bias_init   = tf.constant_initializer(0.)
        self.bn_init     = {'beta': tf.constant_initializer(0.),
                           'gamma': tf.random_normal_initializer(1., 0.01)}
        self.bn_params   = {'is_training':self.is_training,'decay':0.9,'epsilon':1e-5,
                           'param_initializers':self.bn_init,'updates_collections':None}
        with tf.variable_scope(self.name,reuse=False) as scope:
            with slim.arg_scope([slim.fully_connected],activation_fn=self.actv,
                                weights_initializer=self.fully_init,biases_initializer=self.bias_init,
                                normalizer_fn=self.bn,normalizer_params=self.bn_params,
                                weights_regularizer=None):            
                # List of features
                _net = self.x 
                for h_idx,h_dim in enumerate(self.h_dims):
                    _net = slim.fully_connected(inputs=_net,num_outputs=h_dim,scope='lin'+str(h_idx))
                self.feat = _net
                _net = slim.dropout(self.feat, keep_prob=self.kp,is_training=self.is_training,scope='dropout')  
                _out = slim.fully_connected(_net,self.t_dim,activation_fn=None,normalizer_fn=None, scope='out')# [N x D]
                self.out = _out
    
    def build_graph(self):
        # Cross-entropy loss
        self._loss_ce = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.t,logits=self.out) # [N]
        self.loss_ce = tf.reduce_mean(self._loss_ce) # []
        # Weight decay regularizer
        _g_vars = tf.global_variables()
        _c_vars = [var for var in _g_vars if '%s/'%(self.name) in var.name]
        self.l2_reg = self.l2_reg_coef*tf.reduce_sum(tf.stack([tf.nn.l2_loss(v) for v in _c_vars])) # []
        # Total loss
        self.loss_total = self.loss_ce + self.l2_reg
        if self.USE_SGD:
            # self.optm = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.loss_total)
            self.optm = tf.train.MomentumOptimizer(learning_rate=self.lr,momentum=self.momentum).minimize(self.loss_total)
        else:
            self.optm = tf.train.AdamOptimizer(learning_rate=self.lr
                                               ,beta1=0.9,beta2=0.999,epsilon=1e-6).minimize(self.loss_total)
        # Accuracy
        _corr = tf.equal(tf.argmax(self.out, 1), tf.argmax(self.t, 1))    
        self.accr = tf.reduce_mean(tf.cast(_corr,tf.float32)) 
    
    # Check parameters
    def check_params(self):
        _g_vars = tf.global_variables()
        self.g_vars = [var for var in _g_vars if '%s/'%(self.name) in var.name]
        if self.VERBOSE:
            print ("==== Global Variables ====")
        for i in range(len(self.g_vars)):
            w_name  = self.g_vars[i].name 
            w_shape = self.g_vars[i].get_shape().as_list()
            if self.VERBOSE:
                print (" [%02d] Name:[%s] Shape:[%s]" % (i,w_name,w_shape))
        
    # Train 
    def train(self,_sess,_x_train,_t_train,_x_test,_t_test,
              _max_epoch=10,_batch_size=256,_lr=1e-3,_kp=0.9,
              _LR_SCHEDULE=False,_PRINT_EVERY=10,_VERBOSE_TRAIN=True):
        tf.set_random_seed(0)
        n_train,n_test = _x_train.shape[0],_x_test.shape[0]
        txtName = ('res/res_%s.txt'%(self.name))
        f = open(txtName,'w') # Open txt file
        print_n_txt(_f=f,_chars='Text name: '+txtName)
        print_period = max(1,_max_epoch//_PRINT_EVERY)
        max_iter,max_test_accr = max(n_train//_batch_size,1),0.0
        for epoch in range(_max_epoch+1): # For every epoch 
            _x_train,_t_train = shuffle(_x_train,_t_train) 
            for iter in range(max_iter): # For every iteration in one epoch
                start,end = iter*_batch_size,(iter+1)*_batch_size
                # Learning rate scheduling
                if _LR_SCHEDULE:
                    if epoch < 0.5*_max_epoch:
                        _lr_use = _lr
                    elif epoch < 0.75*_max_epoch:
                        _lr_use = _lr/10.0
                    else:
                        _lr_use = _lr/100.0
                else:
                    _lr_use = _lr
                if self.USE_MIXUP:
                    x_batch = _x_train[start:end,:]
                    t_batch = _t_train[start:end,:]
                    x_batch,t_batch = mixup(x_batch,t_batch,32)
                else:
                    x_batch = _x_train[start:end,:]
                    t_batch = _t_train[start:end,:]
                feeds = {self.x:x_batch,self.t:t_batch
                         ,self.kp:_kp,self.lr:_lr_use,self.is_training:True}
                _sess.run(self.optm,feed_dict=feeds)
            # Print training losses, training accuracy, validation accuracy, and test accuracy
            if (epoch%print_period)==0 or (epoch==(_max_epoch)):
                batch_size4print = 512 
                # Compute train loss and accuracy
                max_iter4print = max(n_train//batch_size4print,1)
                train_loss,train_accr,n_temp = 0,0,0
                for iter in range(max_iter4print):
                    start,end = iter*batch_size4print,(iter+1)*batch_size4print
                    feeds_train = {self.x:_x_train[start:end,:],self.t:_t_train[start:end,:]
                             ,self.kp:1.0,self.is_training:False}
                    _train_loss,_train_accr = _sess.run([self.loss_total,self.accr],feed_dict=feeds_train) 
                    _n_temp = end-start; n_temp+=_n_temp
                    train_loss+=(_n_temp*_train_loss); train_accr+=(_n_temp*_train_accr)
                train_loss/=n_temp;train_accr/=n_temp
                # Compute test loss and accuracy
                max_iter4print = max(n_test//batch_size4print,1)
                test_loss,test_accr,n_temp = 0,0,0
                for iter in range(max_iter4print):
                    start,end = iter*batch_size4print,(iter+1)*batch_size4print
                    feeds_test = {self.x:_x_test[start:end,:],self.t:_t_test[start:end,:]
                             ,self.kp:1.0,self.is_training:False}
                    _test_loss,_test_accr = _sess.run([self.loss_total,self.accr],feed_dict=feeds_test) 
                    _n_temp = end-start; n_temp+=_n_temp
                    test_loss+=(_n_temp*_test_loss); test_accr+=(_n_temp*_test_accr)
                test_loss/=n_temp;test_accr/=n_temp
                # Compute max val accr
                if test_accr > max_test_accr:
                    max_test_accr = test_accr
                strTemp = (("[%02d/%d] [Loss] train:%.3f test:%.3f"
                            +" [Accr] train:%.1f%% test:%.1f%% maxTest:%.1f%%")
                       %(epoch,_max_epoch,train_loss,test_loss
                         ,train_accr*100,test_accr*100,max_test_accr*100))
                print_n_txt(_f=f,_chars=strTemp,_DO_PRINT=_VERBOSE_TRAIN)
        # Done 
        print ("Training finished.")
