
# coding: utf-8

# # Jeopardy Questions

# In[1]:


import pandas

jeopardy = pandas.read_csv("jeopardy.csv")
jeopardy.head()


# In[2]:


jeopardy.columns


# It looks like there are some spaces in front of the column names. Let's clean them up:

# In[3]:


jeopardy.columns = [['Show Number', 'Air Date', 'Round', 'Category', 'Values', 'Question', 'Answer']]


# In[4]:


jeopardy.columns


# In[5]:


jeopardy.dtypes


# # Normalizing Text

# The texts in column Question and Answer need to be normalized so lowercase words are treated the same to uppercase words

# In[6]:


import re

def normalize_text(string):
    text = string.lower()
    text = re.sub("[^A-Za-z0-9\s]", "", text)
    return text


# In[7]:


jeopardy["clean_question"] = jeopardy['Question'].apply(normalize_text)
jeopardy['clean_answer'] = jeopardy['Answer'].apply(normalize_text)
jeopardy.head()


# # Normalizing Columns

# In[8]:


def normalize_value(string):
    text = re.sub("[^A-Za-z0-9\s]", "", string)
    try:
        text = int(text)
    except Exception:
        text = 0
    return text


# In[9]:


jeopardy['clean_value'] = jeopardy['Values'].apply(normalize_value)
jeopardy.head()


# In[10]:


jeopardy['Air Date'] = pandas.to_datetime(jeopardy['Air Date'])
jeopardy.dtypes


# # Answers in Questions

# In[11]:


# How many times words in the answer also occur in the question
def count_matches(row):
    split_answer = row["clean_answer"].split(" ")
    split_question = row["clean_question"].split(" ")
    if "the" in split_answer:
        split_answer.remove("the")
    if len(split_answer) == 0:
        return 0
    match_count = 0
    for item in split_answer:
        if item in split_question:
            match_count += 1
    return match_count / len(split_answer)


# In[12]:


jeopardy["answer_in_question"] = jeopardy.apply(count_matches, axis=1)
jeopardy["answer_in_question"].mean()


# The answer only appears in the question about 6% of the time. This isn't a huge number, and means that we probably can't just hope that hearing a question will enable us to figure out the answer. We'll probably have to study.

# # Recycled Questions

# In[13]:


question_overlap = []
terms_used = set()
for i, row in jeopardy.iterrows():
    split_question = row["clean_question"].split(" ")
    split_question = [q for q in split_question if len(q) > 5]
    match_count = 0
    for word in split_question:
        if word in terms_used:
            match_count += 1
    for word in split_question:
        terms_used.add(word)
    if len(split_question) > 0:
        match_count /= len(split_question)
    question_overlap.append(match_count)
    
jeopardy["question_overlap"] = question_overlap
jeopardy["question_overlap"].mean()


# There is about 69% overlap between terms in new questions and old questions. This only looks at a small set of questions, and it doesn't look at phrases, it looks at single terms. This makes it relatively insignificant, but it does mean that it's worth looking more into the recycling of questions.

# # Low Value vs High Value Questions

# In[14]:


def determine_low_high_value(row):
    value = 0
    if row['clean_value'] > 800:
        value = 1
    return value

jeopardy["high_value"] = jeopardy.apply(determine_low_high_value, axis=1)
jeopardy.tail()


# In[15]:


def count_usage(word):
    low_count = 0
    high_count = 0
    for i, row in jeopardy.iterrows():
        split_question = row['clean_question'].split(" ")
        if word in split_question:
            if row['high_value'] == 1:
                high_count += 1
            else:
                low_count += 1
    return high_count, low_count

observed_expected = []

comparison_terms = list(terms_used)[:5]
for term in comparison_terms:
    observed_expected.append(count_usage(term))
    
observed_expected


# # Applying the Chi-Squared Test

# In[22]:


from scipy.stats import chisquare
import numpy as np

high_value_count = jeopardy[jeopardy["high_value"] == 1].shape[0]
low_value_count = jeopardy[jeopardy["high_value"] == 0].shape[0]

chi_squared = []
for obs in observed_expected:
    total = sum(obs)
    total_prop = total / jeopardy.shape[0]
    high_value_exp = total_prop * high_value_count
    low_value_exp = total_prop * low_value_count
    
    observed = np.array([obs[0], obs[1]])
    expected = np.array([high_value_exp, low_value_exp])
    chi_squared.append(chisquare(observed, expected))

chi_squared

