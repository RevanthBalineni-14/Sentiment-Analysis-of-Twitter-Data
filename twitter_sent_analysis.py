#for catch block
from os import error

import pandas as pd #reads csv file
#for corpus
import nltk
#plotting graphs
import matplotlib.pyplot as plt
from termcolor import cprint

import seaborn as sns #plotting graphs
import string
#import default stop words
from nltk.corpus import stopwords

from imblearn.over_sampling import SMOTE #distributes the dataset uniformly 
#naive bayes for classification
from sklearn.naive_bayes import MultinomialNB


from sklearn.metrics import classification_report , confusion_matrix , accuracy_score #create confusion matrix
#regex for unwanted character removal
import re
#split dataframe into training, testing part
from sklearn.model_selection import train_test_split
#for vectorization either of them can be used 
#however we are getting only 80 for count but for tfidf it is around 86

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.naive_bayes import BernoulliNB #for sarcasm
#for conversion into  array
import numpy as np

try:
  #parse csv using pandas
  tweets=pd.read_csv('C:\\Users\\revan\\Desktop\\Tweets.csv')
  cprint("Total number of sentiments of tweets :",'green')
  print(tweets.airline_sentiment.value_counts())
  #size adjustment
  plt.figure(figsize = (10, 8))
  ax = sns.countplot(x = 'airline_sentiment', data = tweets, palette = 'pastel')
  ax.set_title(label = 'Total number of sentiments of tweets', fontsize = 20)
  plt.show()

  #load onion database
  data = pd.read_json("C:\\Users\\revan\\Desktop\\Sarcasm_Headlines_Dataset.json", lines=True)
  #convert classification into arthematic values
  data["is_sarcastic"] = data["is_sarcastic"].map({0: "Not Sarcasm", 1: "Sarcasm"})
  data = data[["headline", "is_sarcastic"]]
  xsar = np.array(data["headline"])
  ysar = np.array(data["is_sarcastic"])
  cv = CountVectorizer()
  XSAR = cv.fit_transform(xsar) # Fit the Data
  print(XSAR)
  X_train, X_test, y_train, y_test = train_test_split(XSAR, ysar, test_size=0.20, random_state=42)
  model = BernoulliNB()
  model.fit(X_train, y_train)
  print("Sarcasm score",model.score(X_test, y_test))

  cprint("Total number of tweets for each airline :",'green')
  print(tweets.groupby('airline')['airline_sentiment'].count())

  plt.figure(figsize = (10, 8))
  ax = sns.countplot(x = 'airline', data = tweets, palette = 'pastel')
  ax.set_title(label = 'Total number of tweets for each airline', fontsize = 20)
  plt.show()

  #print reasons for -ve tweets
  cprint('Reasons Of Negative Tweets :','green')
  print(tweets.negativereason.value_counts())
  #determine size
  plt.figure(figsize = (24, 10))
  sns.countplot(x = 'negativereason', data = tweets, palette = 'hls')
  plt.title('Reasons Of Negative Tweets About Airlines', fontsize = 20)
  plt.show()

  
  # convert Sentiments to 0,1,2 because we need numerical format
  def convert_Sentiment(sentiment):
      if  sentiment == "positive":
          return 2
      elif sentiment == "neutral":
          return 1
      elif sentiment == "negative":
          return 0
  #convert each of the sentiments into numerical values using
  #convert sentiment function
  tweets.airline_sentiment = tweets.airline_sentiment.apply(lambda x : convert_Sentiment(x))
        
  # Remove stop words which were dowloaded form nltk
  def remove_stopwords(text):
      text = ' '.join([word for word in text.split() if word not in (stopwords.words('english'))])
      return text

  # Remove url of cited stuff
  def remove_url(text):
      url = re.compile(r'https?://\S+|www\.\S+')
      return url.sub(r'',text)

  # Remove punctuation basic step
  def remove_punctuation(text):
      table = str.maketrans('', '', string.punctuation)
      return text.translate(table)

  # Remove html since most of our tweets attach pic or retweet stuff
  def remove_html_tags(text):
      html=re.compile(r'<.*?>')
      return html.sub(r'',text)

  # Remove @username common in every tweet
  def remove_twitter_username(text):
      return re.sub('@[^\s]+','',text)

  # Removes emojis from the tweets improves accuracy
  def remove_emoji(text):
      emoji_pattern = re.compile("["
                            u"\U0001F600-\U0001F64F"  # emojis with faces
                            u"\U0001F300-\U0001F5FF"  # emojis like 100
                            u"\U0001F680-\U0001F6FF"  # map emojis
                            u"\U0001F1E0-\U0001F1FF"  # flags emojis
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            "]+", flags=re.UNICODE)
      return emoji_pattern.sub(r'', text)
  
  # abbreviate text
  #there are many others possible we did most we could find
  def abbreviate(text):
      text = re.sub(r"lol", " laughing out loud", text)
      text = re.sub(r"jk", " just kidding", text)
      text = re.sub(r"lmao", " laughing", text)
      text = re.sub(r"won\'t", " will not", text)
      text = re.sub(r"won\'t've", " will not have", text)
      text = re.sub(r"can\'t", " can not", text)
      text = re.sub(r"don\'t", " do not", text)      
      text = re.sub(r"can\'t've", " can not have", text)
      text = re.sub(r"ma\'am", " madam", text)
      text = re.sub(r"let\'s", " let us", text)
      text = re.sub(r"ain\'t", " am not", text)
      text = re.sub(r"shan\'t", " shall not", text)
      text = re.sub(r"sha\n't", " shall not", text)
      text = re.sub(r"o\'clock", " of the clock", text)
      text = re.sub(r"y\'all", " you all", text)
      text = re.sub(r"n\'t", " not", text)
      text = re.sub(r"n\'t've", " not have", text)
      text = re.sub(r"\'re", " are", text)
      text = re.sub(r"\'s", " is", text)
      text = re.sub(r"\'d", " would", text)
      text = re.sub(r"\'d've", " would have", text)
      text = re.sub(r"\'ll", " will", text)
      text = re.sub(r"\'ll've", " will have", text)
      text = re.sub(r"\'t", " not", text)
      text = re.sub(r"\'ve", " have", text)
      text = re.sub(r"\'m", " am", text)
      text = re.sub(r"\'re", " are", text)
      return text  

  # Seperate alphanumeric which are commonly found in twitter language
  def remove_numbers(text):
      words = text
      words = re.findall(r"[^\W\d_]+|\d+", words)
      return " ".join(words)

  #remove repeated characters
  def continuosly_repeated_char(text):
      tchr = text.group(0) 
      
      if len(tchr) > 1:
          return tchr[0:2] 

  def unique_char(rep, text):
      substitute = re.sub(r'(\w)\1+', rep, text)
      return substitute

  def char(text):
      substitute = re.sub(r'[^a-zA-Z]',' ',text)
      return substitute
  
  # combaine negative reason with  tweet (if exsist)
  tweets['final_text'] = tweets['negativereason'].fillna('') + ' ' + tweets['text'] 

  # Apply functions on tweets
  tweets['final_text'] = tweets['final_text'].apply(lambda x : remove_twitter_username(x))
  tweets['final_text'] = tweets['final_text'].apply(lambda x : remove_url(x))
  tweets['final_text'] = tweets['final_text'].apply(lambda x : remove_emoji(x))
  tweets['final_text'] = tweets['final_text'].apply(lambda x : abbreviate(x))
  tweets['final_text'] = tweets['final_text'].apply(lambda x : remove_numbers(x))
  tweets['final_text'] = tweets['final_text'].apply(lambda x : unique_char(continuosly_repeated_char,x))
  tweets['final_text'] = tweets['final_text'].apply(lambda x : char(x))
  tweets['final_text'] = tweets['final_text'].apply(lambda x : x.lower())
  tweets['final_text'] = tweets['final_text'].apply(lambda x : remove_stopwords(x))

  X=tweets['final_text']
  y=tweets['airline_sentiment']
  # Apply TFIDF on cleaned tweets
  tfid = TfidfVectorizer()
  X_final =  tfid.fit_transform(X)
  
  print(tweets['final_text'])
  #Our dataset has 70% negative tweets to handle it we are using smote
  #Smote increased our accuracy because it balances our dataset while training the model
  smote = SMOTE()
  x_sm,y_sm = smote.fit_resample(X_final,y)
  # Split Data into train & test 
  #final acc depends on tested data
  X_train , X_test , y_train , y_test = train_test_split(x_sm , y_sm , test_size=0.2)

  nb = MultinomialNB()
  nb.fit(X_train,y_train)
  nb_prediction =  nb.predict(X_test)
  print(accuracy_score(nb_prediction,y_test))
  print(classification_report(y_test,nb_prediction))
  
  cm = confusion_matrix(y_test,nb_prediction)
  # plot confusion matrix 
  #this is useful feature of nb class 
  #we can know what our model is doing wrong by correlating with matrix
  plt.figure(figsize=(8,6))
  sentiment_classes = ['Negative', 'Neutral', 'Positive']
  sns.heatmap(cm, cmap=plt.cm.Greens, annot=True, fmt='d', 
            xticklabels=sentiment_classes,
            yticklabels=sentiment_classes)
  plt.title('Confusion matrix', fontsize=16)
  plt.xlabel('Actual label', fontsize=12)
  plt.ylabel('Predicted label', fontsize=12)
  plt.show()
  useri=''
  #user input for testing the accuracy of our model
  while(useri!="quit"):
      useri=input("Enter sample data for testing:")
      useri=remove_stopwords(useri)
      useri=remove_url(useri)
      useri=remove_punctuation(useri)
      useri=remove_html_tags(useri)
      useri=remove_twitter_username(useri)
      useri=remove_emoji(useri)
      useri=abbreviate(useri)
      useri=remove_numbers(useri)
      udata = cv.transform([useri]).toarray()
      opt = model.predict(udata)
      print(opt)
      vectorizeduserinput=tfid.transform([useri])
      output=nb.predict_proba(vectorizeduserinput)
      print(output)
       #extract the values to display whether verdict was negative, neutral or postive
      listofval=output[0]
      #if else block for printing the verdict
      if listofval[0] > listofval[1] and listofval[0]>listofval[2]:
        print("negative")
      elif listofval[1] > listofval[0] and listofval[1]>listofval[2]:
        print("neutral")
      else:
        print("postive")
        
except error as e:
  print(e)