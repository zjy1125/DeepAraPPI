# DeepAraPPI
DeepAraPPI: an integrated deep learning framework that uses sequence, 
domain and Gene Ontology information to predict PPIs of Arabidopsis thaliana. 
DeepAraPPI currently consists of three basic models: 
(i) a Siamese recurrent convolutional neural network (RCNN) model based on word2vec encoding, 
(ii) a multiple layer perceptron model based on Domain2vec encoding, and 
(iii) an multiple layer perceptron model based on GO2vec encoding. 
Finally, DeepAraPPI combines the results of three individual predictors using a logistic regression model. 
DeepAraPPI shows superior performance compared to existing state-of-the-art Arabidopsis PPI prediction methods.
# Requirements
python                    (==3.6)<br>
tensorflow-gpu            (==1.9.0)<br>
keras                     (==2.2.0)<br>
cudatoolkit               (==9.0)<br>
cudnn                     (==7.6.5)<br>
# Datasets
We provided datasets for two species, 
including Arabidopsis thaliana and Oryza sativa. 
For the prediction of Arabidopsis PPIs, 
three tasks were designed to test the performance of the model for unknown proteins.<br>

ara_data/ contains all the datasets required for Arabidopsis PPIs prediction:<br>

[ara_ppi_sample.txt](https://github.com/zjy1125/DeepAraPPI/blob/main/ara_data/ara_ppi_sample.txt) can be used to execute Task1, randomly dividing the dataset into 
80% training set and 20% independent test set, and 
perform 5-fold cross-validation on the training set.<br>
[c1_ppi_sample.txt](https://github.com/zjy1125/DeepAraPPI/blob/main/ara_data/c1_ppi_sample.txt) 
is used as the training set to train the model for Task2,3.<br>
[c2_ppi_sample.txt](https://github.com/zjy1125/DeepAraPPI/blob/main/ara_data/c2_ppi_sample.txt)
 is used as the independent test set for Task2.<br>
[c3_ppi_sample.txt](https://github.com/zjy1125/DeepAraPPI/blob/main/ara_data/c3_ppi_sample.txt)
 is used as the independent test set for Task3.
# Acknowledgments
We would like to thank the [PIPR](https://github.com/muhaochen/seq_ppi) team for the source code of RCNN model part.