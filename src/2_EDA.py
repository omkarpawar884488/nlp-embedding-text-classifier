#!/usr/bin/env python
# coding: utf-8

# ## Exploratory Data Analysis

# ### This notebook performs Exploratory Data Analysis (EDA) on the sentiment dataset for the challenge. It explores class balance, text length distributions, and text cleanliness.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import emoji

from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2


# In[2]:


def clean_for_eda(text, keep_emojis=True):
    """
    Normalize social media text for EDA.
    - Always normalizes URLs and mentions.
    - Emojis are kept or removed based on `keep_emojis`.
    """
    text = str(text)

    # Normalize URLs (replace with <URL>)
    text = re.sub(r'http\S+|www\S+|bit\.ly\S+|store\.playstation\.com\S*', '<URL>', text)

    # Normalize @mentions (replace with <USER>)
    text = re.sub(r'@\w+', '<USER>', text)

    # Handle emojis
    if keep_emojis:
        # Keep emojis as they are (for BERT)
        pass
    else:
        # Remove emojis for non-BERT EDA
        text = emoji.replace_emoji(text, replace='')

    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# In[3]:


eda_df = pd.read_csv('E://CV//Internship//Coding_Challenge_Omkar_Pawar//data//cleaned_sentiment_dataset.csv')


# In[4]:


df_1 = eda_df.copy()


# In[5]:


df_1


# In[ ]:





# ### Class balance

# In[6]:


pct = (df_1['sentiment'].value_counts(normalize=True)*100).round(2)
cnt = df_1['sentiment'].value_counts()
display(pd.DataFrame({'count':cnt, 'pct%':pct}))
plt.figure(figsize=(6,4))
df_1['sentiment'].value_counts().plot(kind='bar'); plt.title('Class counts'); plt.ylabel('count')


# - Sentiment distribution: Negative > Positive > Neutral > Irrelevant. 
# - Classes are moderately balanced with no category underrepresented (<10%).
# - Higher Negative count likely reflects natural user bias toward complaints. 
# - Positive and Neutral are close, suggesting a realistic mix of sentiment.
# - while Irrelevant is smallest.

# In[ ]:





# In[7]:


#Text length distribution (overall & by class)

df_1['n_chars'] = df_1['text'].str.len()
df_1['n_words'] = df_1['text'].str.split().str.len()

df_1[['n_words','n_chars']].describe()

ax = df_1['n_words'].plot(kind='hist', bins=50, alpha=0.7, title='Word count distribution')
plt.figure(figsize=(7,4))
df_1.groupby('sentiment')['n_words'].median().plot(kind='bar'); plt.title('Median words by sentiment');


# - Word Count Distribution:
# - Most texts are short (≈10–25 words), with very few exceeding 75 words.
# - The dataset resembles short, social-media-style inputs; long-form reviews are rare.
# - A max sequence length of ~50 words (≈95th percentile) is sufficient for transformer models.
# - Positive texts are typically shorter (brief praise), while Neutral and Negative texts are longer and more descriptive.
# - This may cause text length to act as an indirect sentiment signal.

# In[8]:


np.percentile(df_1['n_words'], [90, 95, 99])


# In[ ]:





# ### Token/character cleanliness check

# In[9]:


# share of texts with urls, @mentions, hashtags, emojis-like chars
has_url = df_1['text'].str.contains(r'http[s]?://|bit\.ly|store\.playstation', case=False, regex=True)
has_mention = df_1['text'].str.contains(r'@\w+')
has_emojiish = df_1['text'].str.contains(r'[\U0001F300-\U0001FAFF]')

pd.DataFrame({
 'has_url%': [has_url.mean()*100],
 'has_mention%':[has_mention.mean()*100],
 'has_emoji%':[has_emojiish.mean()*100]
}).round(2)


# In[10]:


eda_bert_df = df_1.copy()
eda_non_bert_df = df_1.copy()


# In[11]:


# BERT version (keep emojis)
eda_bert_df['text'] = eda_bert_df['text'].apply(lambda x: clean_for_eda(x, keep_emojis=True))

# Non-BERT version (remove emojis)
eda_non_bert_df['text'] = eda_non_bert_df['text'].apply(lambda x: clean_for_eda(x, keep_emojis=False))


# URLs replaced with <URL>
# 
# Mentions replaced with <USER>
# 
# Emojis kept in BERT and removed in non-BERT.

# In[ ]:





# ### Key Terms and Phrases per Sentiment (TF-IDF Analysis)

# In[12]:


## old code
sentiment_stop = {'user','url','com','www','http','https'}   # artifacts to drop

def top_ngrams_clean(sub, ngram=(1,2), topk=20):
    vec = TfidfVectorizer(
        ngram_range=ngram,
        min_df=5,
        max_df=0.85,                 # drop very-common terms
        stop_words='english',        # remove stopwords
        token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z]+\b',  # words with >=2 letters
        lowercase=True,
        sublinear_tf=True
    )
    X = vec.fit_transform(sub['text'])
    feats = np.array(vec.get_feature_names_out())

    # remove artifact features if present
    keep = ~np.isin(feats, list(sentiment_stop))
    X = X[:, keep]
    feats = feats[keep]

    scores = np.asarray(X.mean(axis=0)).ravel()
    out = pd.DataFrame({'term': feats, 'tfidf': scores})
    return out.sort_values('tfidf', ascending=False).head(topk)
from sklearn.feature_extraction.text import TfidfVectorizer

for s in eda_non_bert_df['sentiment'].unique():
    print(f'=== {s} ===')
    display(top_ngrams_clean(eda_non_bert_df[eda_non_bert_df['sentiment']==s], ngram=(1,2), topk=20))


# - Identify top unigrams and bigrams per sentiment using TF-IDF.
# - Highlights most distinctive words and phrases associated with each sentiment class.

# In[ ]:





# ### Chi-Square Feature Importance per Sentiment

# In[19]:


vec = TfidfVectorizer(
    ngram_range=(1,2),
    min_df=5,
    max_df=0.85,
    stop_words='english',
    token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z]+\b',
    lowercase=True,
    sublinear_tf=True
)
X = vec.fit_transform(eda_non_bert_df['text'])
y = eda_non_bert_df['sentiment']
feats = np.array(vec.get_feature_names_out())

for cls in y.unique():
    chi2_scores, p_values = chi2(X, (y == cls))
    top_idx = chi2_scores.argsort()[::-1][:20]
    print(f'=== Most discriminative for {cls} ===')
    display(pd.DataFrame({
        'term': feats[top_idx],
        'chi2': chi2_scores[top_idx],
        'p_value': p_values[top_idx]
    }))


#  - Identify the most discriminative unigrams and bigrams for each sentiment using Chi-square scores on TF-IDF features. 
#  - Highlights words and phrases most strongly associated with each sentiment class.

# In[ ]:





# In[ ]:





# ### Sentiment Distribution by Product

# In[24]:


pivot = pd.crosstab(eda_non_bert_df['product'], eda_non_bert_df['sentiment'], normalize='index').round(3)
display(pivot.sort_values('Negative', ascending=False).head(15))


# In[25]:


display(pivot.sort_values('Positive', ascending=False).head(15))


# - Compute normalized sentiment proportions per product to identify 
# - which products receive more positive or negative feedback on average.
# - Helps reveal product-level sentiment trends or bias in the dataset.
# - AssassinsCreed and Borderlands show the highest share of positive sentiment, suggesting strong user satisfaction.
# Over 70% of MaddenNFL posts were negative, indicating widespread dissatisfaction, while HomeDepot and Overwatch had more balanced sentiment distributions.
#     
# - The 'product' column is excluded from modeling to prevent data leakage.
# - Although products may show sentiment trends, the goal is to learn sentiment from text itself.
# - Product info is used only for EDA and not as a predictive feature.

# In[ ]:





# ### Vocabulary Analysis and Out-of-Vocabulary (OOV) Risk

# In[29]:


tokens = eda_non_bert_df['text'].str.lower().str.replace(r'[^a-z0-9\s]', ' ', regex=True).str.split()
vocab = Counter([t for row in tokens for t in row])
len_vocab = len(vocab); pct_singletons = (sum(1 for k,v in vocab.items() if v==1)/len_vocab)*100
len_vocab, round(pct_singletons,2)


# I analyzed the dataset’s vocabulary coverage to assess out-of-vocabulary risk for classical NLP models.
# The corpus contained about 30K unique tokens, with roughly 24% being singletons — words that appear only once.
# This indicates a moderately rich vocabulary but also some noise or rare terms, which could increase OOV risk during inference.
# Based on this, I would use TF-IDF with min_df=2–3 to filter rare terms.

# In[30]:


tokens = eda_non_bert_df['text'].str.lower().str.replace(r'[^a-z0-9\s]', ' ', regex=True).str.split()
vocab = Counter([t for row in tokens for t in row])
# all unique words (≈30K)
all_words = list(vocab.keys())

# words that appear only once
singleton_words = [word for word, count in vocab.items() if count == 1]

print(f"Total unique words: {len(all_words)}")
print(f"Singleton words: {len(singleton_words)} ({len(singleton_words)/len(all_words)*100:.2f}%)")


# In[31]:


singleton_words


#  Analyze vocabulary coverage to estimate OOV risk for classical NLP models.
# - The dataset contains ~30K unique tokens, with ~24% appearing only once (singletons).
# - Indicates a moderately rich vocabulary with some rare/noisy terms.
# - Suggests applying lemmatization and TF-IDF filtering (min_df=2–3) 

# In[ ]:





# ### Punctuation Analysis

# In[34]:


eda_non_bert_df.groupby('sentiment')['n_words'].describe()[['50%','75%','max']]
eda_non_bert_df['exclaim'] = eda_non_bert_df['text'].str.count('!')
eda_non_bert_df.groupby('sentiment')['exclaim'].mean().round(2)


# - Analyze text length and punctuation usage across sentiment classes.
# - Positive texts show higher average exclamation counts (0.56) than negative (0.26),
# - indicating greater expressiveness and emotional tone in positive messages.
# - This pattern supports label quality and suggests punctuation could be retained as a potential feature

# In[ ]:





# In[ ]:




