## Introduction:
-This notebook explores the classification of spam and ham (non-spam) messages using natural language processing (NLP) techniques and machine learning algorithms. The dataset consists of text messages labeled as spam or ham. The goal is to preprocess the text data, engineer relevant features, and train various classifiers to predict whether a given message is spam or not. The notebook covers text preprocessing, feature engineering, exploratory data analysis (EDA), model training, evaluation, and comparison of multiple classifiers to identify the most effective approach for spam detection.


```python
# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
```


```python
# Loading the dataset
df = pd.read_csv('spam.csv',encoding='ISO-8859-1' )
```

## Data Processing


```python
# drop last 3 columns
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
```


```python
# renaming the columns
df.rename(columns={'v1':'target','v2':'text'},inplace=True)
```

```python
# Encoding the 'target' column
df['target'] = encoder.fit_transform(df['target'])
```
-  Visualizing the distribution of target classes

```python

import matplotlib.pyplot as plt
plt.pie(df['target'].value_counts(), labels=['ham','spam'],autopct="%0.2f")
plt.show()
```
<img width="333" alt="Screenshot 2024-02-10 213019" src="https://github.com/Pratiksha022002/SMS-Spam-Classifier/assets/99002937/9dda10ac-18fe-4e44-b9a4-0c3036aa5fd6">

- Data is imbalanced

## Text Processing:

```python
# Adding new feature: Number of characters in text
df['num_characters'] = df['text'].apply(len)
```

```python
# Adding new feature: Number of words in text
df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))
```

```python
# Adding new feature: Number of sentences in text
df['num_sentences'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))
```


```python
df.head()
```


```python
df[['num_characters','num_words','num_sentences']].describe()
```

## Data Visualization

- Visualizing the distribution of the number of characters in ham and spam messages

```python
plt.figure(figsize=(12,6))
sns.histplot(df[df['target'] == 0]['num_characters'])
sns.histplot(df[df['target'] == 1]['num_characters'],color='red')
```
<img width="416" alt="Screenshot 2024-02-10 213033" src="https://github.com/Pratiksha022002/SMS-Spam-Classifier/assets/99002937/57692313-3b18-406d-bf04-2897f077ff74">


- Visualizing the distribution of the number of words in ham and spam messages

```python
import seaborn as sns
plt.figure(figsize=(12,6))
sns.histplot(df[df['target'] == 0]['num_words'])
sns.histplot(df[df['target'] == 1]['num_words'],color='red')
```
<img width="453" alt="Screenshot 2024-02-10 213040" src="https://github.com/Pratiksha022002/SMS-Spam-Classifier/assets/99002937/b17f3f8a-a15c-47d8-9420-0278f90083f5">


##  Data Preprocessing
- Lower case
- Tokenization
- Removing special characters
- Removing stop words and punctuation
- Stemming


```python
# Function to preprocess text data
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
            
    return " ".join(y)
```


```python
# Applying text transformation to the 'text' column
df['transformed_text'] = df['text'].apply(transform_text)
df.head()
```
<img width="726" alt="Screenshot 2024-02-10 213212" src="https://github.com/Pratiksha022002/SMS-Spam-Classifier/assets/99002937/9aca1aa2-1d0f-42b7-814e-4ec62a0058b1">


- Generating word clouds for spam and ham messages
```python
wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')
spam_wc = wc.generate(df[df['target'] == 1]['transformed_text'].str.cat(sep=" "))
```


```python
plt.figure(figsize=(15,6))
plt.imshow(spam_wc)
```
<img width="387" alt="Screenshot 2024-02-10 213049" src="https://github.com/Pratiksha022002/SMS-Spam-Classifier/assets/99002937/887ebe86-bbd6-4634-bc30-157dc42b9b6f">

```python
ham_wc = wc.generate(df[df['target'] == 0]['transformed_text'].str.cat(sep=" "))
```


```python
plt.figure(figsize=(15,6))
plt.imshow(ham_wc)
```
<img width="374" alt="Screenshot 2024-02-10 213057" src="https://github.com/Pratiksha022002/SMS-Spam-Classifier/assets/99002937/fdcc1653-d807-4bf0-954b-26d5db8b09ed">


- Analysis of word frequency in spam messages
```python
spam_corpus = []
for msg in df[df['target'] == 1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)
        
```

- Visualizing the most common words in spam messages


```python
common_words_df = pd.DataFrame(Counter(spam_corpus).most_common(30))
common_words_df.columns = ['Word', 'Count']
sns.barplot(data=common_words_df, x='Word', y='Count')
plt.xticks(rotation='vertical')
plt.show()
```
<img width="462" alt="Screenshot 2024-02-10 213127" src="https://github.com/Pratiksha022002/SMS-Spam-Classifier/assets/99002937/1ea31a43-ba3c-4761-b199-9bd629d94d5d">

-  Analysis of word frequency in ham messages
```python
ham_corpus = []
for msg in df[df['target'] == 0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)
```

```python
common_words_df = pd.DataFrame(Counter(ham_corpus).most_common(30))
common_words_df.columns = ['Word', 'Count']
sns.barplot(data=common_words_df, x='Word', y='Count')
plt.xticks(rotation='vertical')
plt.show()
```
<img width="495" alt="Screenshot 2024-02-10 213136" src="https://github.com/Pratiksha022002/SMS-Spam-Classifier/assets/99002937/0d82eef5-1520-4cf2-b650-6c7ed67a21aa">


##  Model Building


```python
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['transformed_text']).toarray()
```


```python

label_encoder = LabelEncoder()
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = label_encoder.fit_transform(df[column])
corr_matrix = df.corr()
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

```
<img width="555" alt="Screenshot 2024-02-10 213159" src="https://github.com/Pratiksha022002/SMS-Spam-Classifier/assets/99002937/4a11093c-e1c9-463a-8006-85305c32d4f9">

```python
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
```
- Splitting the dataset into training and testing sets

```python
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)
```

```python
gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()
```

**GussianNB**


```python
gnb.fit(X_train,y_train)
y_pred1 = gnb.predict(X_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))
```

    0.8694390715667312
    [[788 108]
     [ 27 111]]
    0.5068493150684932


**MultinomialNB**


```python
mnb.fit(X_train,y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))
```

    0.9709864603481625
    [[896   0]
     [ 30 108]]
    1.0


**BernoulliNB**


```python
bnb.fit(X_train,y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))
```

    0.9835589941972921
    [[895   1]
     [ 16 122]]
    0.991869918699187


```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import VotingClassifier
```


```python
svc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50,random_state=2)
xgb = XGBClassifier(n_estimators=50,random_state=2)
```


```python
clfs = {
    'SVC' : svc,
    'KN' : knc, 
    'NB': mnb, 
    'DT': dtc, 
    'LR': lrc, 
    'RF': rfc, 
    'AdaBoost': abc, 
    'BgC': bc, 
    'ETC': etc,
    'GBDT':gbdt,
    'xgb':xgb
}
```


```python
def train_classifier(clf,X_train,y_train,X_test,y_test):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    
    return accuracy,precision
```



```python
accuracy_scores = []
precision_scores = []

for name,clf in clfs.items():
    
    current_accuracy,current_precision = train_classifier(clf, X_train,y_train,X_test,y_test)
    
    print("For ",name)
    print("Accuracy - ",current_accuracy)
    print("Precision - ",current_precision)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)
```

    For  SVC
    Accuracy -  0.9758220502901354
    Precision -  0.9747899159663865
    For  KN
    Accuracy -  0.9052224371373307
    Precision -  1.0
    For  NB
    Accuracy -  0.9709864603481625
    Precision -  1.0
    For  DT
    Accuracy -  0.9294003868471954
    Precision -  0.8282828282828283
    For  LR
    Accuracy -  0.9584139264990329
    Precision -  0.9702970297029703
    For  RF
    Accuracy -  0.9758220502901354
    Precision -  0.9829059829059829
    For  AdaBoost
    Accuracy -  0.960348162475822
    Precision -  0.9292035398230089
    For  BgC
    Accuracy -  0.9584139264990329
    Precision -  0.8682170542635659
    For  ETC
    Accuracy -  0.9748549323017408
    Precision -  0.9745762711864406
    For  GBDT
    Accuracy -  0.9468085106382979
    Precision -  0.9191919191919192
    For  xgb
    Accuracy -  0.9671179883945842
    Precision -  0.9262295081967213

<img width="812" alt="Screenshot 2024-02-10 213325" src="https://github.com/Pratiksha022002/SMS-Spam-Classifier/assets/99002937/322356ec-8eb6-408c-8e68-23b06bedeebf">


```python
performance_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores}).sort_values('Precision',ascending=False)
```


```python
sns.catplot(x = 'Algorithm', y='value', 
               hue = 'variable',data=performance_df1, kind='bar',height=5)
plt.ylim(0.5,1.0)
plt.xticks(rotation='vertical')
plt.show()
```
<img width="456" alt="Screenshot 2024-02-10 214112" src="https://github.com/Pratiksha022002/SMS-Spam-Classifier/assets/99002937/0d21526b-98ff-42a2-800f-680f330260cf">

- Voting Classifier
```python

svc = SVC(kernel='sigmoid', gamma=1.0,probability=True)
mnb = MultinomialNB()
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
```


```python
voting = VotingClassifier(estimators=[('svm', svc), ('nb', mnb), ('et', etc)],voting='soft')
```


```python
voting.fit(X_train,y_train)
mnb.fit(X_train,y_train)
```


```python
y_pred = voting.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))
```

    Accuracy 0.9816247582205029
    Precision 0.9917355371900827

-  Applying stacking

```python

estimators=[('svm', svc), ('nb', mnb), ('et', etc)]
final_estimator=RandomForestClassifier()
```



```python
clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)
```


```python
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))
```

    Accuracy 0.9796905222437138
    Precision 0.9465648854961832
## Conclusion:
-  Conducted EDA on the SMS spam dataset, revealing a class imbalance with 86.6% ham and 13.4% spam messages.
-  Achieved high accuracy and precision scores, with the Multinomial Naive Bayes model yielding the best precision score of 100% and the Voting Classifier achieving 98.16% accuracy and 99.17% precision. 
-  Voting Classifier has better accuracy and performance than stacking classifier.
