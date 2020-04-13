from numpy import *
import sys  

from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers.merge import Multiply
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, AveragePooling1D
from keras.optimizers import Adam

import pandas as pd
import scipy.stats as st

#command: python .\DeepCpf1Train.py .\HT1_1.txt .\HT1_2.txt 

def PREPROCESS(lines):
    data_n = len(lines) - 1
    SEQ = zeros((data_n, 34, 4), dtype=int)
    Indel_fre_BS = []
    
    for l in range(1, data_n+1):
        data = lines[l].split()
        seq = data[1]
        Indel_fre_BS.append(data[10])
        for i in range(34):
            if seq[i] in "Aa":
                SEQ[l-1, i, 0] = 1
            elif seq[i] in "Cc":
                SEQ[l-1, i, 1] = 1
            elif seq[i] in "Gg":
                SEQ[l-1, i, 2] = 1
            elif seq[i] in "Tt":
                SEQ[l-1, i, 3] = 1

    return SEQ,Indel_fre_BS

print("Loading training data.")
FILE = open(sys.argv[1], "r")
data = FILE.readlines()
SEQ,Indel_fre_BS = PREPROCESS(data)
FILE.close()

print("Loading test data.")
test_FILE = open(sys.argv[2], "r")
test_data = test_FILE.readlines()
test_SEQ,test_Indel_fre_BS = PREPROCESS(test_data)
test_FILE.close()

print("Build the model.")
Seq_deepCpf1_Input_SEQ = Input(shape=(34,4))    #?使用了onehot
Seq_deepCpf1_C1 = Convolution1D(80, 5, activation='relu')(Seq_deepCpf1_Input_SEQ) #80:filters
Seq_deepCpf1_P1 = AveragePooling1D(2)(Seq_deepCpf1_C1)
Seq_deepCpf1_F = Flatten()(Seq_deepCpf1_P1)
Seq_deepCpf1_DO1= Dropout(0.3)(Seq_deepCpf1_F)
Seq_deepCpf1_D1 = Dense(80, activation='relu')(Seq_deepCpf1_DO1)
Seq_deepCpf1_DO2= Dropout(0.3)(Seq_deepCpf1_D1)
Seq_deepCpf1_D2 = Dense(40, activation='relu')(Seq_deepCpf1_DO2)
Seq_deepCpf1_DO3= Dropout(0.3)(Seq_deepCpf1_D2)
Seq_deepCpf1_D3 = Dense(40, activation='relu')(Seq_deepCpf1_DO3)
Seq_deepCpf1_DO4= Dropout(0.3)(Seq_deepCpf1_D3)
Seq_deepCpf1_Output = Dense(1, activation='linear')(Seq_deepCpf1_DO4)
Seq_deepCpf1 = Model(inputs=[Seq_deepCpf1_Input_SEQ], outputs=[Seq_deepCpf1_Output])

print("Compile the model.")
Seq_deepCpf1.compile(
    optimizer=Adam(), #optimizer='adam',
    loss='mean_squared_error',
)

print("Train the model.")
test_Indel_fre_BS_V = list(map(float, test_Indel_fre_BS))
Seq_deepCpf1.fit(
    x = array(SEQ),
    y = array(Indel_fre_BS),
    epochs=500,
    batch_size=64, #15000/100=150;太小，loss会很小，但是test data loss不再收敛
    validation_data=(test_SEQ, test_Indel_fre_BS_V)
)

print("Test the model.")
test_loss=Seq_deepCpf1.evaluate(
   x = array(test_SEQ),
   y = array(test_Indel_fre_BS),
)
#print(test_loss)

print("save weights")
Seq_deepCpf1.save_weights('Seq_deepCpf1.h5')

print("predict test data")
test_predict = Seq_deepCpf1.predict(
    x = test_SEQ)

print("Spearman correlation  #0.2760428")
test_predict_F = list(test_predict.flatten())
test_Indel_fre_BS = list(map(float, test_Indel_fre_BS))
print(st.spearmanr(test_predict_F, test_Indel_fre_BS))


