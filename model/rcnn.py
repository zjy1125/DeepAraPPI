import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import keras
from keras.layers import Input,CuDNNGRU 
from keras.layers.merge import concatenate,multiply 
from keras.layers import Dense,Bidirectional
from keras.models import Model 
from keras.optimizers import Adam
from keras.layers.convolutional import Conv1D 
from keras.layers.pooling import MaxPooling1D, GlobalAveragePooling1D
import numpy as np

def get_sample_label(file):
    samples = []
    labels = []
    with open(file) as f:
        for line in f:
            (pro1,pro2,t_label) = line.strip().split('\t')
            sample = (pro1,pro2)
            samples.append(sample)
            t_label = int(t_label)
            if t_label == 1:
                f_label = 0
                label = [t_label, f_label]
                labels.append(label)
            else:
                f_label = 1
                label = [t_label, f_label]
                labels.append(label)
    samples = np.array(samples)
    labels = np.array(labels)
    return samples,labels
sample_file = 'data/ara_ppi_sample.txt'
all_sample,all_label = get_sample_label(sample_file)

def get_pro_embedding(pro_file,embedding_file):
    all_pro_seq = {}
    for line in open(pro_file):
        line = line.strip().split('\t')
        all_pro_seq[line[0]] = line[1]
    
    aa_vec = {} 
    for line in open(embedding_file):
        line = line.strip().split('\t')
        aa = line[0]
        vec = np.array([float(x) for x in line[1].split()])
        embed_dim = len(vec)
        aa_vec[aa] = vec
    
    from tqdm import tqdm
    pros_vec = {}
    for pro,seq in tqdm(all_pro_seq.items()):
        seq_list = list(seq) 
        seq_vec = [] 
        for x in seq_list: 
            vec = aa_vec.get(x) 
            seq_vec.append(vec)
        if len(seq_vec) > 2000:
            seq_vec = seq_vec[:2000]
            seq_vec = np.array(seq_vec)
        elif len(seq_vec) < 2000:
            seq_vec = np.concatenate((seq_vec,np.zeros((2000 - len(seq_vec), 32))))
        pros_vec[pro] = seq_vec
    return pros_vec
seq_file = 'data/ara_sequence.txt'
emb_file = 'data/epoch_5_sg_0_1_3_word2vec.txt' 
all_pro_vec = get_pro_embedding(seq_file, emb_files)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(all_sample,all_label,test_size=0.2,random_state=63)

X_train_array_1 = []
X_train_array_2 = []
for train_sample in X_train:
    X_train_array_1.append(all_pro_vec[train_sample[0]])
    X_train_array_2.append(all_pro_vec[train_sample[1]])
X_train_array_1 = np.array(X_train_array_1)
X_train_array_2 = np.array(X_train_array_2)

X_test_array_1 = []
X_test_array_2 = []
for test_sample in X_test:
    X_test_array_1.append(all_pro_vec[test_sample[0]])
    X_test_array_2.append(all_pro_vec[test_sample[1]])
X_test_array_1 = np.array(X_test_array_1)
X_test_array_2 = np.array(X_test_array_2)

seq_size = 2000
dim = 32
hidden_dim = 50
def build_model():
    seq_input1 = Input(shape=(seq_size, dim), name='seq1')
    seq_input2 = Input(shape=(seq_size, dim), name='seq2')
    l1=Conv1D(hidden_dim, 3)
    r1=Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))
    l2=Conv1D(hidden_dim, 3)
    r2=Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))
    l3=Conv1D(hidden_dim, 3)
    r3=Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))
    l4=Conv1D(hidden_dim, 3)
    r4=Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))
    l5=Conv1D(hidden_dim, 3)
    r5=Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))
    l6=Conv1D(hidden_dim, 3)
    

    s1=MaxPooling1D(3)(l1(seq_input1))
    s1=concatenate([r1(s1), s1])
    s1=MaxPooling1D(3)(l2(s1))
    s1=concatenate([r2(s1), s1])
    s1=MaxPooling1D(3)(l3(s1))
    s1=concatenate([r3(s1), s1])
    s1=MaxPooling1D(3)(l4(s1))
    s1=concatenate([r4(s1), s1])
    s1=MaxPooling1D(3)(l5(s1))
    s1=concatenate([r5(s1), s1])
   
    s1=l6(s1)
    
    s1=GlobalAveragePooling1D()(s1)
    s2=MaxPooling1D(3)(l1(seq_input2))
    s2=concatenate([r1(s2), s2])
    s2=MaxPooling1D(3)(l2(s2))
    s2=concatenate([r2(s2), s2])
    s2=MaxPooling1D(3)(l3(s2))
    s2=concatenate([r3(s2), s2])
    s2=MaxPooling1D(3)(l4(s2))
    s2=concatenate([r4(s2), s2])
    s2=MaxPooling1D(3)(l5(s2))
    s2=concatenate([r5(s2), s2])
    

    s2=l6(s2)
    s2=GlobalAveragePooling1D()(s2)
    merge_text = multiply([s1, s2])
    x = Dense(100, activation='linear')(merge_text)
    x = keras.layers.LeakyReLU(alpha=0.3)(x)
    x = Dense(int((hidden_dim+7)/2), activation='linear')(x)
    x = keras.layers.LeakyReLU(alpha=0.3)(x)
    main_output = Dense(2, activation='softmax')(x)
    merge_model = Model(inputs=[seq_input1, seq_input2], outputs=[main_output])
    return merge_model

from sklearn.model_selection import KFold, ShuffleSplit

kf = KFold(n_splits=5,random_state=63, shuffle=True)
train_test = []
for train, test in kf.split(y_train):
    train_test.append((train, test))

from keras.models import load_model
k = 1
batch_size1 = 32
n_epochs=50
for train, test in train_test:
    merge_model = None
    merge_model = build_model()
    adam = Adam(lr=0.001,beta_1=0.9,beta_2=0.999,amsgrad=True, epsilon=1e-8)
    merge_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    merge_model.fit([X_train_array_1[train],X_train_array_2[train]],y_train[train], batch_size=batch_size1, epochs=n_epochs,
                    validation_data=([X_train_array_1[test],X_train_array_2[test]],y_train[test]))
   
    merge_model.save('rcnn_model'+str(k)+'.h5')
    pred = merge_model.predict([X_train_array_1[test],X_train_array_2[test]])
    np.savetxt('rcnn_val_result'+str(k)+'.txt',pred,fmt='%s',delimiter='\t')
    test_pred = merge_model.predict([X_test_array_1, X_test_array_2])
    np.savetxt('rcnn_test_reslt_'+str(k)+'.txt',test_pred,fmt='%s',delimiter='\t')
    k+=1
