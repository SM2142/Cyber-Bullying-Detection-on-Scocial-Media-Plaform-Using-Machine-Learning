# -*- coding: utf-8 -*-
"""Cyber bulling Dettection.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xNWqYo-ADEjogUdlnIVrUsuM6ERBU4Cn
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

from nltk.stem.porter import PorterStemmer
import nltk
import re, string
from nltk.corpus import stopwords

file_url = 'https://raw.githubusercontent.com/deepu099cse/Multi-Labeled-Bengali-Toxic-Comments-Classification/main/Multi_labeled_toxic_comments.csv'

df = pd.read_csv(file_url)

df.head()

def check_for_bullying(row):
    if row['vulgar'] == 1 or row['hate'] == 1 or row['religious'] == 1 or row['threat'] == 1 or row['troll'] == 1 or row['Insult'] == 1:
        return 1
    else:
        return 0
df['bullying'] = df.apply(check_for_bullying, axis=1)

# Display the DataFrame with the new 'bullying' column
print(df.head())

df.drop(['hate'],axis = 1,inplace = True)
df.drop(['religious'],axis = 1,inplace = True)
df.drop(['troll'],axis = 1,inplace = True)
df.drop(['Insult'],axis = 1,inplace = True)
df.drop(['vulgar'], axis=1, inplace = True)
df.drop(['threat'], axis=1, inplace = True)
df.head()

df['bullying'].value_counts()

df['bullying'].value_counts().plot(kind='pie' ,autopct='%.2f')

df['bullying'].value_counts().plot(kind='bar')

print("PosiNon cyber trollingtive: ", df.bullying.value_counts()[0]/len(df.bullying)*100,"%")
print("Cybertrolling: ", df.bullying.value_counts()[1]/len(df.bullying)*100,"%")

nltk.download('stopwords')
stop = set(stopwords.words('bengali'))

regex = re.compile('[%s]' % re.escape(string.punctuation))

def test_re(s):
    return regex.sub('', s)

df ['content_without_stopwords'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
df ['content_without_puncs'] = df['content_without_stopwords'].apply(lambda x: regex.sub('',x))
del df['content_without_stopwords']
del df['text']
df

#Stemming
porter_stemmer = PorterStemmer()
#punctuations
nltk.download('punkt')
tok_list = []
size = df.shape[0]

for i in range(size):
  word_data = df['content_without_puncs'][i]
  nltk_tokens = nltk.word_tokenize(word_data)
  final = ''
  for w in nltk_tokens:
    final = final + ' ' + porter_stemmer.stem(w)
  tok_list.append(final)

df['content_tokenize'] = tok_list
del df['content_without_puncs']
df

noNums = []
for i in range(len(df)):
  noNums.append(''.join([i for i in df['content_tokenize'][i] if not i.isdigit()]))

df['text'] = noNums
df

tfIdfVectorizer=TfidfVectorizer(use_idf=True, sublinear_tf=True)
tfIdf = tfIdfVectorizer.fit_transform(df.text.tolist())

print(tfIdf.shape) # means total rows  20001 with 14783 features

df2 = pd.DataFrame(tfIdf[2].T.todense(), index=tfIdfVectorizer.get_feature_names_out(), columns=["TF-IDF"])
df2 = df2.sort_values('TF-IDF', ascending=False)
print(df2.head(10))

dfx = pd.DataFrame(tfIdf.toarray(), columns = tfIdfVectorizer.get_feature_names_out())
print(dfx)

def display_scores(vectorizer, tfidf_result):
    scores = zip(vectorizer.get_feature_names_out(),
                 np.asarray(tfidf_result.sum(axis=0)).ravel())
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    i=0
    for item in sorted_scores:
        print ("{0:50} Score: {1}".format(item[0], item[1]))
        i = i+1
        if (i > 25):
          break

#top 25 words
display_scores(tfIdfVectorizer, tfIdf)

X=tfIdf.toarray()
y = np.array(df.bullying.tolist())
#Spltting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

#Training data biasness
unique_elements, counts_elements = np.unique(y_train, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))

#Test Data
unique_elements, counts_elements = np.unique(y_test, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))

def getStatsFromModel(model):
  print(classification_report(y_test, y_pred))

  logit_roc_auc = roc_auc_score(y_test, model.predict(X_test))
  fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
  plt.figure()
  plt.plot(fpr, tpr, label='(area = %0.2f)' % logit_roc_auc)
  plt.plot([0, 1], [0, 1],'r--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver operating characteristic')
  plt.legend(loc="lower right")
  plt.savefig('Log_ROC')
  plt.show()

def getStatsFromModel(model):
  print(classification_report(y_test, y_pred))

  logit_roc_auc = roc_auc_score(y_test, model.predict(X_test))
  fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
  plt.figure()
  plt.plot(fpr, tpr, label='(area = %0.2f)' % logit_roc_auc)
  plt.plot([0, 1], [0, 1],'r--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver operating characteristic')
  plt.legend(loc="lower right")
  plt.savefig('Log_ROC')
  plt.show()

df.isnull().mean()*100

X = df['text']
y = df['bullying']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)
x_val=vectorizer.transform(X)

model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

y_pred = model.predict(X_test_vectorized)

# Evaluate the model
accuracy1 = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"accuracy : {accuracy}")

num_data_points_y_test = len(y_test)
num_data_points_y_pred = len(y_pred)
print(f"Number of data points in y_test: {num_data_points_y_test}")
print(f"Number of data points in y_pred: {num_data_points_y_pred}")

#print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)

conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix as a heatmap
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

accuracy = (conf_matrix[0, 0] + conf_matrix[1, 1]) / sum(sum(conf_matrix))

print(f'Accuracy: {accuracy}')

# Calculate precision from the confusion matrix
precision = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])

print(f'Precision: {precision}')

# Calculate recall from the confusion matrix
recall = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])

print(f'Recall: {recall}')

true_positive_rate = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])

# Calculate True Negative Rate (Specificity)
true_negative_rate = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])

print(f'True Positive Rate (Sensitivity): {true_positive_rate}')
print(f'True Negative Rate (Specificity): {true_negative_rate}')

labels = ['Precision', 'Accuracy', 'Recall']
values = [precision, accuracy, recall]

plt.bar(labels, values, color=['blue', 'green', 'orange'])
plt.ylabel('Score')
plt.title('Model Performance Metrics')
plt.show()

"""# cross validation

"""

cross_val_scores = cross_val_score(model, x_val, y, cv=5)
print("Cross-validation scores:", cross_val_scores)
print("Mean cross-validation score:", cross_val_scores.mean())

plt.plot(range(1, 6), cross_val_scores, marker='o')
plt.xlabel('Fold')
plt.ylabel('Accuracy Score')
plt.title('Cross-Validation Scores')
plt.show()

"""##Data visualization"""

! pip install pandas-profiling==2.9.0

import os
os._exit(00)

import pandas as pd
import pandas_profiling
from pandas_profiling import ProfileReport

profile = ProfileReport(df, title='Pandas Profiling Report', explorative=True, minimal=True, plot={'categorical': False})

profile.to_file("your_report.html")

"""#ID3"""

model1 = DecisionTreeClassifier(criterion='entropy')  # ID3 algorithm uses entropy
model1.fit(X_train_vectorized, y_train)

# Predictions on the test set
y_pred = model1.predict(X_test_vectorized)

# Evaluate the model
accuracy2 = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Display results
print("Accuracy:", accuracy2)
print("Confusion Matrix:\n", conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Not Bullying', 'Bullying'], yticklabels=['Not Bullying', 'Bullying'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Calculate precision from the confusion matrix
precision = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])

print(f'Precision: {precision}')
# Calculate recall from the confusion matrix
recall = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])

print(f'Recall: {recall}')



"""#CV"""

cross_val_scores = cross_val_score(model1, x_val, y, cv=5)
print("Cross-validation scores:", cross_val_scores)
print("Mean cross-validation score:", cross_val_scores.mean())

"""SVM"""

model2 = SVC(kernel='linear')
model2.fit(X_train_vectorized, y_train)

# Predictions on the test set
y_pred = model2.predict(X_test_vectorized)

# Evaluate the model
accuracy3 = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Display results
print("Accuracy:", accuracy3)
print("Confusion Matrix:\n", conf_matrix)
# Calculate precision from the confusion matrix
precision = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])

print(f'Precision: {precision}')
# Calculate recall from the confusion matrix
recall = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])

print(f'Recall: {recall}')

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Not Bullying', 'Bullying'], yticklabels=['Not Bullying', 'Bullying'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

cross_val_scores = cross_val_score(model2, x_val, y, cv=5)
print("Cross-validation scores:", cross_val_scores)
print("Mean cross-validation score:", cross_val_scores.mean())

"""Random Forest"""

model3 = RandomForestClassifier()
model3.fit(X_train_vectorized, y_train)

from sklearn.metrics import precision_score, recall_score
# Predictions on the test set
y_pred = model3.predict(X_test_vectorized)

# Evaluate the model
accuracy4 = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Display results
print("Accuracy:", accuracy4)
print("Confusion Matrix:\n", conf_matrix)

precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

# Display results
print("Precision:", precision)
print("Recall:", recall)



# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Not Bullying', 'Bullying'], yticklabels=['Not Bullying', 'Bullying'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

cross_val_scores = cross_val_score(model3, x_val, y, cv=5)
print("Cross-validation scores:", cross_val_scores)
print("Mean cross-validation score:", cross_val_scores.mean())

"""Naive Bayes"""

model4 = MultinomialNB()
model4.fit(X_train_vectorized, y_train)

# Predictions on the test set
y_pred = model4.predict(X_test_vectorized)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Display results
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Not Bullying', 'Bullying'], yticklabels=['Not Bullying', 'Bullying'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

cross_val_scores = cross_val_score(model4, x_val, y, cv=21)
print("Cross-validation scores:", cross_val_scores)
print("Mean cross-validation score:", cross_val_scores.mean())

import matplotlib.pyplot as plt

# Accuracy scores for each model
accuracies = [accuracy1, accuracy2,accuracy3,accuracy4]
models = ['Logistic Regression', 'Decision Tree', 'SVM', 'Random Forest']

# Plotting
plt.figure(figsize=(8, 8))
plt.bar(models, accuracies, color=['blue', 'green', 'red', 'purple'])
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Accuracy of Different Models')
plt.ylim(0, 1)  # Set y-axis limit from 0 to 1 for accuracy scores
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()