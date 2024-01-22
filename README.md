# Precipitation Prediction using ML
### Precipitation prediction using ML by creating various models and checking their accuracies and efficiencies.
### Aim: 
To create a **machine learning model** that can predict prediction.

### Dataset:
- The precipitation column in the given dataset was our target feature in this project.
- Changed the positive data values for prediction to 1 to indicate the occurrence of prediction
  - This helps in the analysis of the data further as we only need to check for 1 (precipitation occurs) and 0 (no precipitation).
- Dropped the columns with an excessive number of null values as they do not contribute to the learning of the model.
- Replaced the null values of the remaining columns with the modes of those columns.
  - Due to this, the variance of the data is reduced.
- Dropped the columns with an excessive number of null values as they do not contribute to the learning of the model.

### Exploratory Data Analysis (EDA):
- Plotted **distplots** of various columns in the data frame to visualize the univariate distribution of data.
- Plotted **boxplots** next to **identify outliers** in the various feature columns.

### Data Preprocessing:
- **Removed the outliers** of some feature columns.
  - The outliers are to be removed because they lie outside of the range of the usual expected results.
  - But one needs to be careful here as to not remove excessive data points while removing outliers, as some of these points may be rare but possible values of the outcomes that can be useful in training the model.
- Plotted a **correlation matrix** indicating the pairwise correlation between the various feature columns and also between the features and target column.


### SMOTE (Synthetic Minority Oversampling Technique):
- Trained the model to **handle the class imbalance** using the SMOTE algorithm, as the negative values were in excess as opposed to the positive values.
- The **minority class is oversampled** by creating new examples from the existing examples.
- Used **Random Forest Classifier** for training the model by importing the classifier and smote function directly from the scikit-learn library.
- Generated a **confusion matrix**. This helps in defining the performance of the classification algorithm used. 
  - The positive and negative samples were seen to be roughly equal after removing the class imbalance.
- Calculated the **Recall Score** of this model. 
  - Recall score helps in determining how many of the actual needed (or positive) outcomes were correctly predicted.

### Chi-square Test:
- Performed **chi-square test** for **feature selection** using the chi2 function from the scikit-learn library.
  - Plotted a bar plot to determine p-values.
  - **P-values** give a measure of the **independence of the feature** from the response.
  - **Removed the feature column** which was **highly independent** of the **response** as this would not help in training the model and could hinder its performance in predicting the correct outcome quickly by learning unnecessary data.
- **Normalized the dataset** to bring the features to a common scale which makes it easier to compare and analyze the data.

### Logistic Regression Classifier:
- Used logistic regression classifier from the scikit-learn library to see if an instance belonged to the class where prediction occurs or not.
- Trained the model using the **train_test_split function** and passed the test size and random state variables.
- This algorithm is generally used for **binary classification** and hence was used to train the model.
- Calculated the **accuracy score** of this model to see the percentage of labels that the model successfully predicted.
-Calculated the **ROC-AUC score** which predicts the efficiency of the model in distinguishing between positive and negative classes.
- Generated a **classification report** that contains various parameters like **precision, recall, and f1-score** which are useful in predicting the **model’s efficiency**.
- Created a **confusion matrix** for this model.
- Plotted regplots between various feature columns and target labels to visualize the logistic regression dependency.

### Decision Tree Classifier:
- Used decision tree classifier from the scikit-learn library to train the model.
- It creates a tree that has the feature labels as nodes and each branch descending from that node corresponds to one of the possible values of that attribute, here we only had 1 and 0 as outputs.
- Calculated the **accuracy score** and **ROC-AUC score** as done previously for the logistic regression model.
- Generated the **classification report** and **confusion matrix** for this model.

### Neural Network Model:
- Imported the **TensorFlow** library to directly train the model using its inbuilt functions.
- Trained the model first.
- Created a **Sequential model** which has a series of neural layers.
  - The **layers** created were of **Dense** type and the number of units (neurons) were controlled by passing the unit variable value to the **Dense function**.
- The **activation functions** are used to create **non-linearity** in the neural network models which allows the building of efficient, optimized, and accurate models.
  - **Relu activation** was used for hidden layers and **sigmoid activation** for the output layer.
- Compiled the model using the **Binary Crossentropy loss function** which measures the difference between the predicted binary outcomes and actual binary labels.
- Used the **Adam optimizer**, which is an iterative optimization algorithm. It minimizes the loss function white training the neural network.
- Calculated the **loss, accuracy score, and ROC-AUC score** for this model.

### Comparison of Models:
- Compared the models based on their accuracy and ROC-Auc scores.
- The **Neural networks model** was found to be the **most efficient** out of the three models based on both scores.
