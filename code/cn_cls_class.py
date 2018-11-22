import tensorflow as tf
import tensorflow.contrib.slim as slim
from sklearn.utils import shuffle
from util import load_directory_data,load_dataset,download_and_load_datasets,mixup,gpusession,print_n_txt,create_gradient_clipping
import numpy as np

class cn_cls_class(object):
    def __init__(self,_name='mlp_cls',_x_dim=128,_t_dim=2,_h_dims=[64,64],
                 _k_mix=5,_rho_ref_train=0.95,_tau_inv=1e-4,_pi1_bias=0.0,
                 _log_sigma_z_val=-2,_logsumexp_coef=0.1,_kl_reg_coef=0.1,
                 _actv=tf.nn.relu,_bn=slim.batch_norm,_l2_reg_coef=1e-5,_momentum = 0.5,
                 _USE_SGD=False,_USE_MIXUP=False,_GPU_ID=0,_VERBOSE=True):
        self.name = _name 
        self.x_dim = _x_dim
        self.t_dim = _t_dim
        self.h_dims = _h_dims
        
        self.k_mix = _k_mix
        self.rho_ref_train = _rho_ref_train
        self.tau_inv = _tau_inv
        self.pi1_bias = _pi1_bias
        self.log_sigma_z_val = _log_sigma_z_val
        self.logsumexp_coef = _logsumexp_coef
        self.kl_reg_coef = _kl_reg_coef
        
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
        self.rho_ref = tf.placeholder(dtype=tf.float32,shape=[],name='rho_ref') 
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
                
                # Feature to K rhos
                _rho_raw = slim.fully_connected(self.feat,self.k_mix,scope='rho_raw')
                self.rho_temp = tf.nn.sigmoid(_rho_raw) # [N x K] # Classification
                self.rho = tf.concat([self.rho_temp[:,0:1]*0.0+self.rho_ref,self.rho_temp[:,1:]]
                                     ,axis=1) # [N x K]
                # Sampler variables
                _Q = self.feat.get_shape().as_list()[1] # Feature dimension
                self.Q = _Q
                self.muW = tf.get_variable(name='muW',shape=[_Q,self.t_dim],
                                          initializer=tf.random_normal_initializer(stddev=0.1)
                                           ,dtype=tf.float32) # [Q x D]
                self.logSigmaW = tf.get_variable(name='logSigmaW'
                                        ,shape=[_Q,self.t_dim]
                                        ,initializer=tf.constant_initializer(-3.0)
                                        ,dtype=tf.float32) # [Q x D]
                self.muZ = tf.constant(np.zeros((_Q,self.t_dim))
                                        ,name='muZ',dtype=tf.float32) # [Q x D]
                self.logSigmaZ = tf.constant(self.log_sigma_z_val*np.ones((_Q,self.t_dim)) # -2.0 <== Important Heuristics
                                        ,name='logSigmaZ',dtype=tf.float32) # [Q x D]
                # Make sampler
                _N = tf.shape(self.x)[0]
                _muW_tile = tf.tile(self.muW[tf.newaxis,:,:]
                                    ,multiples=[_N,1,1]) # [N x Q x D]
                _sigmaW_tile = tf.exp(tf.tile(self.logSigmaW[tf.newaxis,:,:]
                                              ,multiples=[_N,1,1])) # [N x Q x D]
                _muZ_tile = tf.tile(self.muZ[tf.newaxis,:,:]
                                    ,multiples=[_N,1,1]) # [N x Q x D]
                _sigmaZ_tile = tf.exp(tf.tile(self.logSigmaZ[tf.newaxis,:,:]
                                              ,multiples=[_N,1,1])) # [N x Q x D]
                samplerList = []
                for jIdx in range(self.k_mix): # For all K mixtures
                    _rho_j = self.rho[:,jIdx:jIdx+1] # [N x 1] 
                    _rho_tile = tf.tile(_rho_j[:,:,tf.newaxis]
                                        ,multiples=[1,_Q,self.t_dim]) # [N x Q x D]
                    _epsW = tf.random_normal(shape=[_N,_Q,self.t_dim],mean=0,stddev=1
                                             ,dtype=tf.float32) # [N x Q x D]
                    _W = _muW_tile + tf.sqrt(_sigmaW_tile)*_epsW # [N x Q x D]
                    _epsZ = tf.random_normal(shape=[_N,_Q,self.t_dim]
                                             ,mean=0,stddev=1,dtype=tf.float32) # [N x Q x D]
                    _Z = _muZ_tile + tf.sqrt(_sigmaZ_tile)*_epsZ # [N x Q x D]
                    # Append to list
                    _Y = _rho_tile*_muW_tile + (1.0-_rho_tile**2) \
                        *(_rho_tile*tf.sqrt(_sigmaZ_tile)/tf.sqrt(_sigmaW_tile) \
                              *(_W-_muW_tile)+tf.sqrt(1-_rho_tile**2)*_Z)
                    samplerList.append(_Y) # Append 
                # Make list to tensor
                WlistConcat = tf.convert_to_tensor(samplerList) # K*[N x Q x D] => [K x N x Q x D]
                self.wSample = tf.transpose(WlistConcat,perm=[1,3,0,2]) # [N x D x K x Q]
                # K mean mixtures [N x D x K]
                _wTemp = tf.reshape(self.wSample
                                ,shape=[_N,self.k_mix*self.t_dim,_Q]) # [N x KD x Q]
                _featRsh = tf.reshape(self.feat,shape=[_N,_Q,1]) # [N x Q x 1]
                _mu = tf.matmul(_wTemp,_featRsh) # [N x KD x Q] x [N x Q x 1] => [N x KD x 1]
                self.mu = tf.reshape(_mu,shape=[_N,self.t_dim,self.k_mix]) # [N x D x K]
                # K var mixtures [N x D x K]
                _logvar_raw = slim.fully_connected(self.feat,self.t_dim,scope='var_raw') # [N x D]
                _var_raw = tf.exp(_logvar_raw) # [N x D]
                _var_tile = tf.tile(_var_raw[:,:,tf.newaxis]
                                    ,multiples=[1,1,self.k_mix]) # [N x D x K]
                _rho_tile = tf.tile(self.rho[:,tf.newaxis,:]
                                    ,multiples=[1,self.t_dim,1]) # [N x D x K]
                _tau_inv = self.tau_inv
                self.var = (1.0-_rho_tile**2)*_var_tile + _tau_inv # [N x D x K]
                # Weight allocation probability pi [N x K]
                _pi_logits = slim.fully_connected(self.feat,self.k_mix
                                                  ,scope='pi_logits') # [N x K]
                self.pi_temp = tf.nn.softmax(_pi_logits,axis=1) # [N x K]
                # Some heuristics to ensure that pi_1(x) is high enough
                self.pi_temp = tf.concat([self.pi_temp[:,0:1]+self.pi1_bias
                                          ,self.pi_temp[:,1:]],axis=1) # [N x K]
                self.pi = tf.nn.softmax(self.pi_temp,axis=1) # [N x K]
                
                # _net = slim.dropout(self.feat, keep_prob=self.kp,is_training=self.is_training,scope='dropout')  
                # _out = slim.fully_connected(_net,self.t_dim,activation_fn=None,normalizer_fn=None, scope='out')# [N x D]
                # self.out = _out
    
    def build_graph(self):
        # MDN loss
        _N = tf.shape(self.x)[0]
        t,mu,var = self.t,self.mu,self.var
        pi = self.pi # [N x K]
        yhat = mu + tf.sqrt(var)*tf.random_normal(shape=[_N,self.t_dim,self.k_mix]) # Sampled y [N x D x K]
        tTile = tf.tile(t[:,:,tf.newaxis],[1,1,self.k_mix]) # Target [N x D x K]
        piTile = tf.tile(pi[:,tf.newaxis,:],[1,self.t_dim,1]) # piTile: [N x D x K]
        
        self.yhat_normalized = tf.nn.softmax(yhat,axis=1) # [N x D x K]
        self._loss_fit = tf.reduce_sum(-piTile*self.yhat_normalized*tTile,axis=[1,2]) # [N]
        self.loss_fit = tf.reduce_mean(self._loss_fit) # [1]

        self._loss_reg = pi*tf.reduce_logsumexp(yhat,axis=[1]) # [N x K]
        self.__loss_reg = tf.reduce_sum(self._loss_reg,axis=[1]) # [N]
        self.loss_reg = self.logsumexp_coef*tf.reduce_mean(self.__loss_reg) # [1] 
        # KL-divergence regularizer 
        _eps = 1e-8
        self._kl_reg = self.kl_reg_coef*tf.reduce_sum(-self.rho
                        *(tf.log(self.pi+_eps)-tf.log(self.rho+_eps)),axis=1) # (N)
        self.kl_reg = tf.reduce_mean(self._kl_reg) # (1)   
        # Weight decay regularizer
        _g_vars = tf.global_variables()
        _c_vars = [var for var in _g_vars if '%s/'%(self.name) in var.name]
        self.l2_reg = self.l2_reg_coef*tf.reduce_sum(tf.stack([tf.nn.l2_loss(v) for v in _c_vars])) # []
        # Total loss
        self.loss_total = tf.reduce_mean(self.loss_fit+self.loss_reg+self.kl_reg+self.l2_reg) # [1]
        if self.USE_SGD:
            # _optm = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
            _optm = tf.train.MomentumOptimizer(learning_rate=self.lr,momentum=self.momentum)
        else:
            _optm = tf.train.AdamOptimizer(learning_rate=self.lr
                                           ,beta1=0.9,beta2=0.999,epsilon=1e-6)
        self.optm = create_gradient_clipping(self.loss_total
                                        ,_optm,tf.trainable_variables(),clipVal=1.0)
        # Accuracy
        # Compute accuray 
        maxIdx = tf.argmax(input=pi,axis=1, output_type=tf.int32) # Argmax Index [N]
        maxIdx = 0*tf.ones_like(maxIdx)
        coords = tf.stack([tf.transpose(gv) for gv in tf.meshgrid(tf.range(_N),tf.range(self.t_dim))] + 
                          [tf.reshape(tf.tile(maxIdx[:,tf.newaxis],[1,self.t_dim]),shape=(_N,self.t_dim))]
                          ,axis=2) # [N x D x 3]
        mu_bar = tf.gather_nd(mu,coords) # [N x D]
        _corr = tf.equal(tf.argmax(mu_bar, 1), tf.argmax(self.t, 1))    
        self.accr = tf.reduce_mean(tf.cast(_corr,tf.float32)) # Accuracy
    
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
                feeds = {self.x:x_batch,self.t:t_batch,self.rho_ref:self.rho_ref_train,
                         self.kp:_kp,self.lr:_lr_use,self.is_training:True}
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
                                 ,self.rho_ref:1.0,self.kp:1.0,self.is_training:False}
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
                                 ,self.rho_ref:1.0,self.kp:1.0,self.is_training:False}
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
