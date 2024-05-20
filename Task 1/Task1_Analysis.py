import gensim
from gensim import corpora
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Ensure you have nltk stopwords downloaded
import nltk
nltk.download('punkt')
nltk.download('stopwords')

df = pd.read_csv('cleaned_data.csv')

# Tokenize and remove stopwords
stop_words = set(stopwords.words('english'))
df['Tokens'] = df['reviews'].apply(lambda x: [word for word in word_tokenize(x) if word not in stop_words])

# Create a dictionary and corpus for LDA
dictionary = corpora.Dictionary(df['Tokens'])
corpus = [dictionary.doc2bow(tokens) for tokens in df['Tokens']]

# Perform LDA
lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)

# Display the topics
topics = lda_model.print_topics(num_words=5)
for topic in topics:
    print(topic)


# Function to get the sentiment
def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity


# Apply sentiment analysis
df['Sentiment'] = df['reviews'].apply(get_sentiment)

# Display the DataFrame with sentiment scores
print(df[['reviews', 'Sentiment']].head())

# Plot histogram of sentiment scores
plt.figure(figsize=(10, 6))
plt.hist(df['Sentiment'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Sentiment Scores')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.savefig('sentiment_score.png')


# Combine all reviews into a single string
all_reviews = ' '.join(df['reviews'])

# Create a word cloud
wordcloud = WordCloud(width=800, height=400, max_words=100).generate(all_reviews)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
# plt.show()
plt.title('Wordcloud')
plt.savefig('wordcloud.png')