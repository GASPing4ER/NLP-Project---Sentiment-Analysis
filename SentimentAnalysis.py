#Loading the data to the variable as DataFrame

#importing pandas library
import pandas as pd

#import counter vectorized function from sklearn
from sklearn.feature_extraction.text import CountVectorizer

count=CountVectorizer()
data=pd.read_csv("Train.csv")
data.head()

#finding the positive and negative text in Data
pos=data[data['label']==1]
neg=data[data['label']==0]
print("Positive text \n",pos.head())
print("\nNegative text \n",neg.head())

#Plotting the Postive vs Negative in piechart.
#Importing matplotlib library to plot pie chart.
import matplotlib.pyplot as plt

fig=plt.figure(figsize=(5,5))
temp=[pos['label'].count(),neg['label'].count()]
plt.pie(temp,labels=["Positive","Negative"],autopct ='%2.1f%%',shadow = True,startangle = 50,explode=(0, 0.3))
plt.title('Positive vs Negative')

#importing re library 
import re

#Defining preprocessing function to process the data
def preprocess(text):
        text=re.sub('<[^>]*>','',text)
        emoji=re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text)
        text=re.sub('[\W]+',' ',text.lower()) +' '.join(emoji).replace('-','')
        return text   
      
#Applying the function preprocess on the data
data['text']=data['text'].apply(preprocess)      

#Displaying the dataframe after applying the preprocessing.
data.head()

#Defining a function called tokenizer which splits the sentence
def tokenizer(text):
        return text.split()
tokenizer("We are testing if tokenizer function splits the text")

#Importing stemmer function from NLTK library
from nltk.stem.porter import PorterStemmer
porter=PorterStemmer()

#Defining function for Tokenizer porter
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]
  
#Importing NLTK library.
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop=stopwords.words('english')

#Importing Word cloud
from wordcloud import WordCloud

#getting positive and negative data
positive_data = data[ data['label'] == 1]
positive_data = positive_data['text']
negative_data = data[data['label'] == 0]
negative_data= negative_data['text']

#Defining the function to plot the data in wordcloud
def plot_wordcloud(data, color = 'white'):
    words = ' '.join(data)
    clean_word = " ".join([word for word in words.split() if(word!='movie' and word!='film')])
    wordcloud = WordCloud(stopwords=stop,background_color=color,width=2500,height=2000).generate(clean_word)
    plt.figure(1,figsize=(10, 7))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
   
#Printing the positive data in wordcloud
print("Positive words")
plot_wordcloud(positive_data,'white')    

#Printing the negative data in wordcloud
print("Negative words")
plot_wordcloud(negative_data,'black')

#importing tfiVectorizer from sklearn for feature extraction.
from sklearn.feature_extraction.text import TfidfVectorizer
tfid=TfidfVectorizer(strip_accents=None,preprocessor=None,lowercase=False,use_idf=True,norm='l2',tokenizer=tokenizer_porter,smooth_idf=True)
y=data.label.values

#scaling the data
x=tfid.fit_transform(data.text)

#splitting the train and test split using train_test_split function of sklearn
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,random_state=1,test_size=0.5,shuffle=False)

#Importing Logisitic RegressionCV from sklearn library
from sklearn.linear_model import LogisticRegressionCV
model=LogisticRegressionCV(cv=6,scoring='accuracy',random_state=0,n_jobs=-1,verbose=3,max_iter=500).fit(X_train,y_train)
y_pred = model.predict(X_test)

#Importing metrics from sklesrn to calculate accuracy
from sklearn import metrics

# Accuracy of our built model
print("Accuracy of our model:",metrics.accuracy_score(y_test, y_pred))

# Accuracy of our model: 0.89045
