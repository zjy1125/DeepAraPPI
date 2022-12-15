import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import keras
from keras.layers import Input
from keras.layers.merge import concatenate
from keras.layers import Dense
from keras.models import Model 
from keras.optimizers import Adam
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
  
sample_file = 'ara_data/ara_ppi_sample.txt'
all_sample,all_label = get_sample_label(sample_file)

def get_embeddings(embedding_file):
    all_pro_vec = {}
    with open(embedding_file) as f:
        for line in f:
            if line.startswith('>'):
                line = line.strip()
                pro = line[1:]
            else:
                vec = line.strip().split('\t',127)
                vec = list(float(i) for i in vec)
                vec = np.array([vec])
                all_pro_vec[pro] = vec   
all_pro_go2vec = get_embeddings('ara_data/ara_go2vec_embeddings.txt')
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(all_sample,all_label,test_size=0.2,random_state=63)

X_train_array_1 = []
X_train_array_2 = []
for train_sample in X_train:
    X_train_array_1.append(all_pro_go2vec[train_sample[0]])
    X_train_array_2.append(all_pro_go2vec[train_sample[1]])
X_train_array_1 = np.array(X_train_array_1)
X_train_array_2 = np.array(X_train_array_2)

X_test_array_1 = []
X_test_array_2 = []
for test_sample in X_test:
    X_test_array_1.append(all_pro_go2vec[test_sample[0]])
    X_test_array_2.append(all_pro_go2vec[test_sample[1]])
X_test_array_1 = np.array(X_test_array_1)
X_test_array_2 = np.array(X_test_array_2)

def build_model():
    go2vec_1 = Input(shape=(1,128),name='go2vec1')
    go2vec_2 = Input(shape=(1,128),name='go2vec2')
    s1 = keras.layers.Flatten()(go2vec_1)
    s2 = keras.layers.Flatten()(go2vec_2)
    add_merge_text = concatenate([s1,s2]) 
    x = Dense(64, activation='linear')(add_merge_text)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)
    x = Dense(16, activation='linear')(x)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)
    main_output = Dense(2, activation='softmax')(x)
    merge_model = Model(inputs=[go2vec_1, go2vec_2], outputs=[main_output])
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
    adam = Adam(lr=0.01,beta_1=0.9,beta_2=0.999,amsgrad=True, epsilon=1e-8)
    merge_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    
    merge_model.fit([X_train_array_1[train],X_train_array_2[train]],y_train[train], batch_size=batch_size1, epochs=n_epochs,
                    validation_data=([X_train_array_1[test],X_train_array_2[test]],y_train[test]))
   
    merge_model.save('go2vec_model'+str(k)+'.h5')
    pred = merge_model.predict([X_train_array_1[test],X_train_array_2[test]])
    np.savetxt('go2vec_val'+str(k)+'.txt',pred,fmt='%s',delimiter='\t')
    test_pred = merge_model.predict([X_test_array_1, X_test_array_2])
    np.savetxt('go2vec_test'+str(k)+'.txt',test_pred,fmt='%s',delimiter='\t')
    k+=1
