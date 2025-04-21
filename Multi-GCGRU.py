#encoding: utf8
'''
Tensorflow version=2.1
'''
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# =================================================================================
# 1) CHEMINS VERS TES FICHIERS  (adapte si besoin)
# =================================================================================
XL_CONCEPT  = "data/raw data/CSI500 concepts.xlsx"      # colonnes : Source | Target | …
XL_INDUSTRY = "data/raw data/CSI500 industry.xlsx"      # colonnes : symbol | sector
XL_HOLDER   = "data/raw data/CSI500 shareholders.xlsx"  # colonnes : Ticker | Stockholder

# ────────────────────────────────────────────────────────────────
# 1) Lire les feuilles
# ────────────────────────────────────────────────────────────────
concept_df  = pd.read_excel(XL_CONCEPT,  usecols=["concept_name","ts_code"])
industry_df = pd.read_excel(XL_INDUSTRY, usecols=["ts_code","industry"])
holder_df   = pd.read_excel(XL_HOLDER,   usecols=["ts_code","holder_name"])

# ────────────────────────────────────────────────────────────────
# 2) Liste commune de tickers et mapping -> indice
# ────────────────────────────────────────────────────────────────
tickers = sorted(set(concept_df["ts_code"]) |
                 set(industry_df["ts_code"]) |
                 set(holder_df["ts_code"]))
idx = {t:i for i,t in enumerate(tickers)}
N   = len(tickers)

# helper pour initialiser matrice
def empty_mat(): return np.zeros((N,N), dtype=np.float32)

# ────────────────────────────────────────────────────────────────
# 3‑A) Matrice **T** : titres qui partagent ≥1 concept
# ────────────────────────────────────────────────────────────────
T = empty_mat()
for concept, group in concept_df.groupby("concept_name"):
    ids = [idx[t] for t in group["ts_code"] if t in idx]
    for i in ids:
        T[i, ids] += 1.0      # +1 par concept partagé
np.fill_diagonal(T, 0)

# ────────────────────────────────────────────────────────────────
# 3‑B) Matrice **I** : titres dans la même industrie
# ────────────────────────────────────────────────────────────────
I = empty_mat()
for indus, group in industry_df.groupby("industry"):
    ids = [idx[t] for t in group["ts_code"] if t in idx]
    for i in ids:
        I[i, ids] = 1.0
np.fill_diagonal(I, 0)

# ────────────────────────────────────────────────────────────────
# 3‑C) Matrice **S** : actionnaires communs
# ────────────────────────────────────────────────────────────────
S = empty_mat()
for holder, group in holder_df.groupby("holder_name"):
    ids = [idx[t] for t in group["ts_code"] if t in idx]
    for i in ids:
        S[i, ids] += 1.0      # +1 par actionnaire commun
np.fill_diagonal(S, 0)

print("T:", T.shape, "I:", I.shape, "S:", S.shape)

# ────────────────────────────────────────────────────────────────
# 4) Normalisation GCN
# ────────────────────────────────────────────────────────────────
def gcn_norm(A):
    A = A + np.eye(A.shape[0], dtype=np.float32)
    d = A.sum(1)
    D = np.diag(np.power(d, -0.5, where=d>0))
    return D @ A @ D

Fixed_Matrices = [
    tf.constant(gcn_norm(S), dtype=tf.float32),
    tf.constant(gcn_norm(I), dtype=tf.float32),
    tf.constant(gcn_norm(T), dtype=tf.float32),
]

'''
samples: the input data
lables: the labels of input data
M: the number of samples
P: the historical length
N: the number of related stocks
F: the number of stock features
We use random data as an example and our reader can use their own data
'''
M = 1000
P = 10
N = T.shape[0]
F = 8
samples = np.random.normal(size=(M, P, N, F))
labels = np.random.normal(size=(M, N))

'''
train set = 70%, test set=20%, val set=10%
'''
TEST_SIZE = 0.3 # the test and val size of dataset
RANDOM_STATE = 5
x_train, x_test, y_train, y_test = train_test_split(samples, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.33, random_state=RANDOM_STATE)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
print(x_val.shape, y_val.shape)

# numpy to tensor
x_train = tf.constant(x_train,dtype=tf.float32)
y_train = tf.constant(y_train,dtype=tf.float32)
x_test = tf.constant(x_test,dtype=tf.float32)
y_test = tf.constant(y_test,dtype=tf.float32)
x_val = tf.constant(x_val,dtype=tf.float32)
y_val= tf.constant(y_val,dtype=tf.float32)


### Model
'''
Units_GCN: the number of units of gcn layers, for example, [16,32]
Units_GRU: the number of units of gru layers, for example, [16,32,G]
Units_FC:the number of units of gcn layers, for example, [1]
Matrix_Weights: the weights of all fixed matrices, for example, [1,1,1] refers that all fixed matrices are utilized
Fixed_Matrices: the set of all fixed matrices, shape = [(N,N),(N,N),(N,N)]
Is_Dyn: True=model with dynamic graph, False=model with fixed graph
'''
class GCGRU(tf.keras.Model):
    def __init__(self, N, F, Units_GCN, Units_GRU, Units_FC,
                 Fixed_Matrices, Matrix_Weights, Is_Dyn,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros'):
        super(GCGRU, self).__init__()
        # Number of Nodes
        self.N = N
        # the number of input features
        self.F = F
        # pre-defined matrices: shape=[(N,N),(N,N),(N,N)]
        self.mat = Fixed_Matrices
        # Dynamic Matrix:shape=(N,N)
        self.dyn = self.add_weight(name='w_Dynamic', shape=(self.N, self.N),
                                 initializer=tf.keras.initializers.get(kernel_initializer),
                                 trainable=True)

        
        if Is_Dyn:
            self.mats = self.dyn
        else:
            self.mats = self.add_weight(name='w_Matrices', shape=(self.N, self.N),
                                        initializer=tf.keras.initializers.get(kernel_initializer),
                                        trainable=True)
            for i in range(len(Matrix_Weights)):
                coe = tf.Variable(1.0,trainable=True)
                self.mats = self.mats + Matrix_Weights[i]*self.mat[i]*coe
        
        # GCN_Weights
        self.units_gcn = Units_GCN
        self.w_gcn = []
        self.b_gcn = []
        pre = self.F
        for i in range(len(self.units_gcn)):
            aft = self.units_gcn[i]
            w = self.add_weight(name='w_GCN', shape=(pre, aft),
                                initializer=tf.keras.initializers.get(kernel_initializer),
                                trainable=True)
            self.w_gcn.append(w)
            b = self.add_weight(name='b_GCN',shape=(aft,),
                                initializer=tf.keras.initializers.get(bias_initializer),
                                trainable=True)
            self.b_gcn.append(b)
            pre = aft
        # GRU_Weights
        self.units_gru = Units_GRU
        self.w_gru = []
        self.b_gru = []
        # the number of output features of Multi-GCN
        C = self.units_gcn[-1]
        F = self.F
        for i in range(len(self.units_gru)-1):
            H = self.units_gru[i]
            pre = F+C+H
            aft = H
            for j in range(3):
                w = self.add_weight(name='w_GRU', shape=(pre,aft),
                                    initializer=tf.keras.initializers.get(kernel_initializer),
                                    trainable=True)
                self.w_gru.append(w)
                b = self.add_weight(name='b_GRU', shape=(aft,),
                                    initializer=tf.keras.initializers.get(bias_initializer),
                                    trainable=True)
                self.b_gru.append(b)
            F = aft
        # the last layer weights
        H = self.units_gru[-2]
        G = self.units_gru[-1]
        w = self.add_weight(name='w_GRU', shape=(H,G),
                            initializer=tf.keras.initializers.get(kernel_initializer),
                            trainable=True)
        
        self.w_gru.append(w)
        b = self.add_weight(name='b_GRU', shape=(G,),
                                 initializer=tf.keras.initializers.get(bias_initializer),
                                 trainable=True)
        self.b_gru.append(b)
        
        # FC_weights
        self.units_fc = Units_FC
        self.w_fc = []
        self.b_fc = []
        pre = G
        for i in range(len(self.units_fc)):
            aft = self.units_fc[i]
            w = self.add_weight(name='w_FC', shape=(pre,aft),
                                initializer=tf.keras.initializers.get(kernel_initializer),
                                trainable=True)
            self.w_fc.append(w)
            b = self.add_weight(name='b_FC', shape=(aft,),
                                initializer=tf.keras.initializers.get(bias_initializer),
                                trainable=True)
            self.b_fc.append(b)
            pre = aft
    
    def Multi_GCN(self, inputs):
        '''
        inputs:shape=(None,P,N,F)
        x_gcn:shape=(None,P,N,C)
        '''
        P = inputs.shape[1]
        x_gcn = []
        for t in range(P):
            # (None, P, N, F) =>(None, N, F)
            xt_gcn = inputs[:,t,:,:]
            
            # (N,N)*(None, N, F)*(F,C)=> (None, N, C)
            for i in range(len(self.units_gcn)):
                xt_gcn = self.mats @ xt_gcn @ self.w_gcn[i] + self.b_gcn[i]
                xt_gcn = tf.nn.tanh(xt_gcn)
            x_gcn.append(xt_gcn)
        # (None,P,N,C)
        x_gcn = tf.stack(x_gcn, axis=1)
        return x_gcn
    
    def GRU(self, x, x_gcn):
        '''
        x:shape=(None,P,N,F)
        x_gcn:shape=(None,N,C)
        x_gru:shape=(None,N,G)
        '''
        # initialize the hidden state in each gru layer
        h_gru = []
        # (None, P, N, F)=>(None, N, F)*(F,H)=> (None, N, H)
        for i in range(len(self.units_gru)-1):
            H = self.units_gru[i]
            h = tf.zeros_like(x[:,0,:,:], dtype=tf.float32) @ tf.zeros([F, H])
            h_gru.append(h)
        
        # all gru layers at each time step
        for t in range(P):
            # (None, P, N, C) =>(None, N, C)
            xt_gcn = x_gcn[:,t,:,:]
            # (None, P, N, F) =>(None, N, F)
            xt = x[:,t,:,:]

            # the i_th layer
            for i in range(len(h_gru)):
                #(None, N, H)
                ht_1 = h_gru[i]
                #(None, N, F)+(None, N, C)+(None, N, H)=> (None, N, C+F+H)
                x_tgh = tf.concat([xt,xt_gcn,ht_1], axis=2)
                #(None, N, C+F+H)=> (None, N, H)
                ut = tf.nn.sigmoid(x_tgh @ self.w_gru[3*i+0] + self.b_gru[3*i+0])
                rt =  tf.nn.sigmoid(x_tgh @ self.w_gru[3*i+1] + self.b_gru[3*i+1])
                
                # (None, N, C+F+H)
                x_tghr = tf.concat([xt, xt_gcn, tf.multiply(rt, ht_1)], axis=2)
                
                # (None, N, H)
                ct = tf.nn.tanh(x_tghr @ self.w_gru[3*i+2] + self.b_gru[3*i+2])
                # (None, N, H)
                ht = tf.multiply(ut, ht_1) + tf.multiply((1-ut), ct)
                # (None, N, H)
                xt = ht
                # (None, N, H)
                h_gru[i]=ht
        # the last layer
        x_gru = tf.nn.sigmoid(ht @ self.w_gru[-1] + self.b_gru[-1])
        return x_gru
    
    def FC(self, x_gru):
        '''
        x_gru:shape=(None, N, G)
        outputs:shape=(None,N,1)
        '''
        x = x_gru
        for i in range(len(self.w_fc)):
            x = x @ self.w_fc[i] + self.b_fc[i]
            x = tf.nn.sigmoid(x)
        # (None, N)
        x_fc = tf.squeeze(x, axis=-1)
        return x_fc

    def build(self, input_shape):
        super().build(input_shape)
        
    def call(self, inputs):
        '''
        inputs:shape=(None,P,N,F)
        x_fc:shape=(None,N)
        '''
        # (None,P,N,C)
        x_gcn = self.Multi_GCN(inputs)
        # (None,N,G)
        x_gru = self.GRU(inputs, x_gcn)
        # (None,N)
        x_fc = self.FC(x_gru)
        return x_fc

# Build model
Units_GCN = [16,32]
Units_GRU = [16,32]
Units_FC = [1]
Matrix_Weights = [1,1,1]
Is_Dyn=False
    
model = GCGRU(N, F, Units_GCN, Units_GRU, Units_FC, Fixed_Matrices, Matrix_Weights,Is_Dyn)
model.build(input_shape=(None, P, N, F))
model.summary()
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=[tf.keras.metrics.BinaryAccuracy()])
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath = './h5/500-GCGRU.weights.h5',
    monitor ='val_binary_accuracy',
    save_weights_only = True,
    save_best_only = True)


Epochs = 20
Batch_size =32
History = model.fit(x_train, y_train, batch_size=Batch_size, epochs=Epochs, callbacks=[model_checkpoint], validation_data=(x_val, y_val))

# Overfitting Observation
loss = History.history['loss']
val_loss = History.history['val_loss']
E = [i for i in range(Epochs)]
plt.plot(E, loss,'b-',label='loss')
plt.plot(E, val_loss,'r-',label='val_loss')
plt.legend()
plt.title('Loss VS Val_loss')
plt.show()

# Prediction
model.load_weights('./h5/500-GCN.weights.h5')
model.load_weights('./h5/500-GCGRU.weights.h5')

result = model.evaluate(x_test, y_test)
print(result)
