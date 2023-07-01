#!/usr/bin/env python
# coding: utf-8

# # Importing the main Libraries

# In[10]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


# # Upload the dataset

# In[11]:


df = pd.read_csv('games.csv')


# # Read the dataset

# In[12]:


df.tail()


# In[13]:


df.shape


# In[14]:


df.columns


# In[15]:


my_list = list(df)


# In[16]:


print(my_list)


# In[17]:


df.info()


# In[18]:


df.describe()


# In[19]:


df.describe(include='all')


# In[20]:


df.to_string(columns=['Name'])


# In[21]:


df['Supported languages'].value_counts(dropna=False)


# # Cleaning the dataset

# In[22]:


#  Convert the 'Release date' column to datetime format
df['Release date'] = pd.to_datetime(df['Release date'], errors='coerce')


# In[23]:


df.info()


# # Null values

# In[24]:


df.isna().sum()


# # NULL values in 'Name' column

# In[25]:


df.loc[df['Name'].isna()].index


# In[26]:


df = df.drop(df.loc[df['Name'].isna()].index)


# # NULL values in 'About the game' column

# In[27]:


df.loc[df['About the game'].isna(),"Name"]


# In[28]:


df.loc[df["Name"]
       .str
       .contains('Playtest','playtest'),'About the game'] = "This is a playtest game. A playtest is the process by which a game designer tests a new game for bugs and design flaws before releasing it to market."


# In[29]:


df.loc[df['About the game'].isna(),"Name"]


# In[30]:


df.loc[df['About the game'].isna(),"Name"].tolist()


# In[31]:


for index, row in df.iterrows():
    if(pd.isnull(row['About the game'])):
        if 'Beta' in row['Name']:
            df.at[index, 'About the game'] = 'This game is beta and still under testing'
        elif "Alpha" in row['Name']:
            df.at[index, 'About the game'] = 'This game is Alpha and still under testing'
        elif "beta" in row['Name']:
            df.at[index, 'About the game'] = 'This game is beta and still under testing'
        elif "BETA" in row['Name']:
            df.at[index, 'About the game'] = 'This game is beta and still under testing'
        elif "Test" in row['Name']:
            df.at[index, 'About the game'] = 'This game is and still under testing'
        elif "SDK" in row['Name']:
            df.at[index, 'About the game'] = 'Software Development Kit of the game'
        elif "Demo" in row['Name']:
            df.at[index, 'About the game'] = 'This game is Demo and still under testing'
        elif "Server" in row['Name']:
            df.at[index, 'About the game'] = 'This is a Server for the game'
        elif "Editor" in row['Name']:
            df.at[index, 'About the game'] = 'This is an Editor for the game'
        else:
            df.at[index, 'About the game'] = 'This game does not have a description'


# #  'Reviews' column

# In[32]:


review = df[['Name', 'Reviews']].copy()


# In[33]:


review


# In[34]:


df = df.drop('Reviews', axis=1)


# # NULL values in 'Metacritic Score' column

# In[35]:


df["Metacritic score"].unique()


# In[36]:


df_score = df.groupby("Metacritic score").agg({"Name":"count"})
df_score


# In[37]:


df["Metacritic url"].unique()


# # NULL values in 'User Score' column

# In[38]:


df["User score"].unique()


# In[39]:


df_score = df.groupby("User score").agg({"Name":"count"})
df_score


# In[40]:


df[['Metacritic score','User score']].corr()


# # Converting Windows, Mac & Linux to one column 'OS'

# In[41]:


df["Windows"]


# In[42]:


for index, row in df.iterrows():
    if row["Windows"] == True:
        df.at[index, 'OS'] = "Windows"
        if row["Mac"] == True:
            df.at[index, 'OS'] = "Windows, Mac"
            if row["Linux"] == True:
                df.at[index, 'OS'] = "Windows, Mac, Linux"
        elif row["Linux"] == True:
            df.at[index, 'OS'] = "Windows, Linux"
    elif row["Windows"] == False and row["Mac"] == True:
        df.at[index, 'OS'] = "Mac"
        if row["Linux"] == True:
            df.at[index, 'OS'] = "Windows, Linux"
    elif row["Windows"] == False and row["Mac"] == False and row["Linux"] == True:
        df.at[index, 'OS'] = "Linux"


# In[43]:


df['OS']


# In[44]:


df = df.drop(['Windows', 'Mac', 'Linux'], axis=1)


# In[45]:


list(df)


# # Deal with 'Developer' & 'Publisher' columns

# In[46]:


df["Developers"].unique()


# In[47]:


df.loc[df['Developers'].isna()]


# In[48]:


df.loc[df['Publishers'].isna()]


# In[49]:


count = 0
for index, row in df.iterrows():
    if(pd.isnull(row['Publishers'])):
        if(pd.isnull(row['Developers'])):
            continue
        else:
            print(row['Developers'])
            count += 1
print(count)


# In[50]:


for index, row in df.iterrows():
    if(pd.isnull(row['Publishers'])):
        if(pd.isnull(row['Developers'])):
            continue
        else:
            df.at[index, 'Publishers'] = df.at[index, 'Developers']


# In[51]:


count = 0
for index, row in df.iterrows():
    if(pd.isnull(row['Developers'])):
        if(pd.isnull(row['Publishers'])):
            continue
        else:
            print(row['Publishers'])
            count += 1
print(count)


# In[52]:


for index, row in df.iterrows():
    if(pd.isnull(row['Developers'])):
        if(pd.isnull(row['Publishers'])):
            continue
        else:
            df.at[index, 'Developers'] = df.at[index, 'Publishers']


# In[53]:


for index, row in df.iterrows():
    if(pd.isnull(row['Developers'])):
        if 'Playtest' in row['Name']:
            df.at[index, 'Developers'] = 'this game is Playtest and still under development'
        elif "Alpha" in row['Name']:
            df.at[index, 'Developers'] = 'this game is Alpha and still under testing the developer is not mentioned'
        elif "beta" in row['Name']:
            df.at[index, 'Developers'] = 'this game is beta and still under testing the developer is not mentioned'
        elif "BETA" in row['Name']:
            df.at[index, 'Developers'] = 'this game is beta and still under testing the developer is not mentioned'
        elif "Test" in row['Name']:
            df.at[index, 'Developers'] = 'this game is and still under testing the developer is not mentioned'
        elif "playtest" in row['Name']:
            df.at[index, 'Developers'] = 'this game is still under testing the developer is not mentioned'
        elif "SDK" in row['Name']:
            df.at[index, 'Developers'] = 'Software Development Kit of the game the developer is not mentioned'
        elif "Demo" in row['Name']:
            df.at[index, 'Developers'] = 'this game is Demo and still under testing the developer is not mentioned'
        elif "Server" in row['Name']:
            df.at[index, 'Developers'] = 'this is a Server for the game the developer is not mentioned'
        elif "Editor" in row['Name']:
            df.at[index, 'Developers'] = 'this is an Editor for the game the developer is not mentioned'
        elif "Beta" in row['Name']:
            df.at[index, 'Developers'] = 'this game is beta and still under testing the developer is not mentioned'
        else:
            df.at[index, 'Developers'] = 'No developer mentioned for this game'


# In[54]:


for index, row in df.iterrows():
    if(pd.isnull(row['Publishers'])):
        if 'Playtest' in row['Name']:
            df.at[index, 'Publishers'] = 'this game is Playtest and still under development'
        elif "Alpha" in row['Name']:
            df.at[index, 'Publishers'] = 'this game is Alpha and still under testing the publisher is not mentioned'
        elif "beta" in row['Name']:
            df.at[index, 'Publishers'] = 'this game is beta and still under testing the publisher is not mentioned'
        elif "BETA" in row['Name']:
            df.at[index, 'Publishers'] = 'this game is beta and still under testing the publisher is not mentioned'
        elif "Test" in row['Name']:
            df.at[index, 'Publishers'] = 'this game is and still under testing the publisher is not mentioned'
        elif "playtest" in row['Name']:
            df.at[index, 'Publishers'] = 'this game is still under testing the publisher is not mentioned'
        elif "SDK" in row['Name']:
            df.at[index, 'Publishers'] = 'Software Development Kit of the game the publisher is not mentioned'
        elif "Demo" in row['Name']:
            df.at[index, 'Publishers'] = 'this game is Demo and still under testing the publisher is not mentioned'
        elif "Server" in row['Name']:
            df.at[index, 'Publishers'] = 'this is a Server for the game the publisher is not mentioned'
        elif "Editor" in row['Name']:
            df.at[index, 'Publishers'] = 'this is an Editor for the game the publisher is not mentioned'
        elif "Beta" in row['Name']:
            df.at[index, 'Publishers'] = 'this game is beta and still under testing the publisher is not mentioned'
        else:
            df.at[index, 'Publishers'] = 'No publisher mentioned for this game'


# # Dealing with 'Categories' column

# In[55]:


df["Categories"].unique()


# In[56]:


df["Genres"].unique()


# In[57]:


df["Tags"].unique()


# In[58]:


df.loc[df['Categories'].isna(),"Name"].unique()


# In[59]:


count = 0
for index, row in df.iterrows():
    if(pd.isnull(row['Categories'])):
        if 'Playtest' in row['Name']:
            count += 1
        elif "Alpha" in row['Name']:
            count += 1
        elif "beta" in row['Name']:
            count += 1
        elif "BETA" in row['Name']:
            count += 1
        elif "Test" in row['Name']:
            count += 1
        elif "playtest" in row['Name']:
            count += 1
        elif "SDK" in row['Name']:
            count += 1
        elif "Demo" in row['Name']:
            count += 1
        elif "Server" in row['Name']:
            count += 1
        elif "Editor" in row['Name']:
            count += 1
        elif "Beta" in row['Name']:
            count += 1
        else:
            continue
print(count)


# In[60]:


for index, row in df.iterrows():
    if(pd.isnull(row['Categories'])):
        if 'Playtest' in row['Name']:
            df.at[index, 'Categories'] = 'Playtest game not playable'
        elif "Alpha" in row['Name']:
            df.at[index, 'Categories'] = 'Alpha game not playable'
        elif "beta" in row['Name']:
            df.at[index, 'Categories'] = 'Beta game not playable'
        elif "BETA" in row['Name']:
            df.at[index, 'Categories'] = 'Beta game not playable'
        elif "Test" in row['Name']:
            df.at[index, 'Categories'] = 'test game not playable'
        elif "playtest" in row['Name']:
            df.at[index, 'Categories'] = 'Playtest game not playable'
        elif "SDK" in row['Name']:
            df.at[index, 'Categories'] = 'Software Development Kit of the game not playable'
        elif "Demo" in row['Name']:
            df.at[index, 'Categories'] = 'Demo game not playable'
        elif "Server" in row['Name']:
            df.at[index, 'Categories'] = 'Server of a game not playable'
        elif "Editor" in row['Name']:
            df.at[index, 'Categories'] = 'Editor of a game not playable'
        elif "Beta" in row['Name']:
            df.at[index, 'Categories'] = 'Beta game not playable'
        else:
            continue


# In[61]:


df.loc[df['Categories'].isna(),"Name"]


# In[62]:


for index, row in df.iterrows():
    if(pd.isnull(row['Categories'])):
        df.at[index, 'Categories'] = 'no Category added'


# In[63]:


df.isnull().sum()


# # Dealing with 'Genre' column

# In[64]:


count = 0
for index, row in df.iterrows():
    if(pd.isnull(row['Genres'])):
        if 'Playtest' in row['Name']:
            count += 1
        elif "Alpha" in row['Name']:
            count += 1
        elif "beta" in row['Name']:
            count += 1
        elif "BETA" in row['Name']:
            count += 1
        elif "Test" in row['Name']:
            count += 1
        elif "playtest" in row['Name']:
            count += 1
        elif "SDK" in row['Name']:
            count += 1
        elif "Demo" in row['Name']:
            count += 1
        elif "Server" in row['Name']:
            count += 1
        elif "Editor" in row['Name']:
            count += 1
        elif "Beta" in row['Name']:
            count += 1
        else:
            continue
print(count)


# In[65]:


for index, row in df.iterrows():
    if(pd.isnull(row['Genres'])):
        if 'Playtest' in row['Name']:
            df.at[index, 'Genres'] = 'Playtest game not playable'
        elif "Alpha" in row['Name']:
            df.at[index, 'Genres'] = 'Alpha game not playable'
        elif "beta" in row['Name']:
            df.at[index, 'Genres'] = 'Beta game not playable'
        elif "BETA" in row['Name']:
            df.at[index, 'Genres'] = 'Beta game not playable'
        elif "Test" in row['Name']:
            df.at[index, 'Genres'] = 'test game not playable'
        elif "playtest" in row['Name']:
            df.at[index, 'Genres'] = 'Playtest game not playable'
        elif "SDK" in row['Name']:
            df.at[index, 'Genres'] = 'Software Development Kit of the game not playable'
        elif "Demo" in row['Name']:
            df.at[index, 'Genres'] = 'Demo game not playable'
        elif "Server" in row['Name']:
            df.at[index, 'Genres'] = 'Server of a game not playable'
        elif "Editor" in row['Name']:
            df.at[index, 'Genres'] = 'Editor of a game not playable'
        elif "Beta" in row['Name']:
            df.at[index, 'Genres'] = 'Beta game not playable'
        else:
            continue


# In[66]:


df.loc[df["Genres"].isnull(),"Name"]


# In[67]:


for index, row in df.iterrows():
    if(pd.isnull(row['Genres'])):
        df.at[index, 'Genres'] = 'no Genres added'


# # Dealing with 'Tags' column

# In[68]:


df.loc[df['Tags'].isna()]


# In[69]:


for index, row in df.iterrows():
    if(pd.isnull(row['Tags'])):
        df.at[index, 'Tags'] = df.at[index, 'Genres']


# In[70]:


df.isnull().sum()


# # Dealing with 'Support email' column

# In[71]:


df["Support email"].unique()


# In[72]:


count = 0
for index, row in df.iterrows():
    if(pd.isnull(row['Support email'])):
        if 'Playtest' in row['Name']:
            count += 1
        elif "Alpha" in row['Name']:
            count += 1
        elif "beta" in row['Name']:
            count += 1
        elif "BETA" in row['Name']:
            count += 1
        elif "Test" in row['Name']:
            count += 1
        elif "playtest" in row['Name']:
            count += 1
        elif "SDK" in row['Name']:
            count += 1
        elif "Demo" in row['Name']:
            count += 1
        elif "Server" in row['Name']:
            count += 1
        elif "Editor" in row['Name']:
            count += 1
        elif "Beta" in row['Name']:
            count += 1
        else:
            continue
print(count)


# In[73]:


for index, row in df.iterrows():
    if(pd.isnull(row['Support email'])):
        if 'Playtest' in row['Name']:
            df.at[index, 'Support email'] = 'Playtest game no Support email available'
        elif "Alpha" in row['Name']:
            df.at[index, 'Support email'] = 'Alpha game no Support email available'
        elif "beta" in row['Name']:
            df.at[index, 'Support email'] = 'Beta game no Support email available'
        elif "BETA" in row['Name']:
            df.at[index, 'Support email'] = 'Beta game no Support email available'
        elif "Test" in row['Name']:
            df.at[index, 'Support email'] = 'test game no Support email available'
        elif "playtest" in row['Name']:
            df.at[index, 'Support email'] = 'Playtest game no Support email available'
        elif "SDK" in row['Name']:
            df.at[index, 'Support email'] = 'Software Development Kit of the game no Support email available'
        elif "Demo" in row['Name']:
            df.at[index, 'Support email'] = 'Demo game no Support email available'
        elif "Server" in row['Name']:
            df.at[index, 'Support email'] = 'Server of a game no Support email available'
        elif "Editor" in row['Name']:
            df.at[index, 'Support email'] = 'Editor of a game no Support email available'
        elif "Beta" in row['Name']:
            df.at[index, 'Support email'] = 'Beta game no Support email available'
        else:
            df.at[index, 'Support email'] = 'no Support email available'


# # Dealing with Impractical columns

# In[74]:


df = df.drop(['Website', 'Support url', 'Metacritic url', 'Screenshots', 'Movies'], axis=1)


# In[75]:


df.isnull().sum()


# # Visualise the dataset

# ### Most played games by user

# In[76]:


df.isnull().sum()


# In[77]:


def owners_clean(x):
    x = x.strip()
    x= x.split("-")
    x = [num.strip(' ') for num in x]
    num1 = int(x[0])
    num2 = int(x[1])
    med = (num2 - num1)/2
    return med


# In[78]:


for index, row in df.iterrows(): 
     df.at[index, 'owners clean'] = owners_clean(df.at[index, 'Estimated owners'])


# In[79]:


df["Estimated owners"].unique()


# In[80]:


df["owners clean"].unique()


# In[81]:


most_downloaded_games = df.groupby("Name").agg({
    "owners clean":"sum"
}).reset_index().sort_values("owners clean",ascending=False).head(10)


# In[82]:


most_downloaded_games.reset_index(inplace=True)


# In[83]:


most_downloaded_games


# In[86]:


# Draw plot
fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
ax.vlines(x=most_downloaded_games.index, ymin=0, ymax=most_downloaded_games['owners clean'], color='red', alpha=0.7, linewidth=2)
ax.scatter(x=most_downloaded_games.index, y=most_downloaded_games['owners clean'], s=75, color='blue', alpha=0.7)


ax.set_title('Most Played Games', fontdict={'size':22})
ax.set_ylabel('owners clean')
ax.set_xticks(most_downloaded_games.index)
ax.set_xticklabels(most_downloaded_games['Name'], rotation=60, fontdict={'horizontalalignment': 'right', 'size':12})
ax.set_ylim(14000000, 60000000)

# Annotate
#for row in df.itertuples():
 #   ax.text(row.Index, row['owners clean']+.5, s=round(row['owners clean'], 2), horizontalalignment= 'center', verticalalignment='bottom', fontsize=14)

plt.show()


# # Most Expensive Games

# In[87]:


df["Price"]


# In[88]:


most_expensive_games = df.groupby("Name").agg({
    "Price":"sum"
}).reset_index().sort_values("Price",ascending=False).head(10)


# In[90]:


most_expensive_games.reset_index(inplace=True)


# In[91]:


most_expensive_games


# In[93]:


import warnings
warnings.filterwarnings("ignore")

# Draw plot
fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
ax.vlines(x=most_expensive_games.index, ymin=0, ymax=most_expensive_games['Price'], color='red', alpha=0.7, linewidth=2)
ax.scatter(x=most_expensive_games.index, y=most_expensive_games['Price'], s=75, color='', alpha=0.7)

# Title, Label, Ticks and Ylim
ax.set_title('Most Expensive Games', fontdict={'size':22})
ax.set_ylabel('Price')
ax.set_xticks(most_expensive_games.index)
ax.set_xticklabels(most_expensive_games['Name'], rotation=60, fontdict={'horizontalalignment': 'right', 'size':12})
ax.set_ylim(50.00, 1200.00)

# Annotate
#for row in df.itertuples():
 #   ax.text(row.Index, row['owners clean']+.5, s=round(row['owners clean'], 2), horizontalalignment= 'center', verticalalignment='bottom', fontsize=14)

plt.show()


# # Free Games vs Paid Games 

# In[94]:


def free_or_paid(x):
    if x == 0:
        return "Free"
    else:
        return "paid"


# In[95]:


for index, row in df.iterrows(): 
     df.at[index, 'Free_or_paid'] = free_or_paid(df.at[index, 'Price'])


# In[96]:


paid_and_free_count = df.groupby("Free_or_paid").agg({
    "Name":"count"
}).reset_index()
paid_and_free_count


# In[100]:


import seaborn as sns
sns.set(font_scale = 1.2)
plt.figure(figsize=(8,8))

plt.title(
    label="Free or Paid Games Percentage", 
    fontdict={"fontsize":16},
    pad=20
)

plt.pie(
    x=paid_and_free_count['Name'], 
    labels=paid_and_free_count['Free_or_paid'],
    colors=sns.color_palette('Set2'),
    startangle=90,
    autopct='%1.2f%%',
    pctdistance=0.80,
    explode=[0.05, 0.05]
)

hole = plt.Circle((0, 0), 0.65, facecolor='white')

plt.gcf().gca().add_artist(hole)

plt.show()


# ## Most Supported Languages

# In[104]:


df["Supported languages"]


# In[105]:


language_count = {}
for language in df['Supported languages'].to_list():
    language = language.strip()
    language_sub = language.split(',')
    for key in language_sub:
        key = key.strip()
        key = key.replace("'", "")
        key = key.replace("[", "")
        key = key.replace("]", "")
        key = key.replace("&amp;lt;strong&amp;gt;&amp;lt;/strong&amp;gt;", "")
        key = key.replace("b/b", "")
        key = key.replace("/b", "")
        key = key.replace(" \\r\\n\\r\\nb/b ", "")
        key = key.replace("/b", "")
        key = key.replace("\\r\\nb/b", "")
        key = key.replace("\\r\\n", "")
        key = key.replace("#", "")
        key = key.replace("\r\\n\\r\\n", "")
        key = key.replace("Russian\\r\\nEnglish\\r\\nSpanish - Spain\\r\\nFrench\\r\\nJapanese\\r\\nCzech", "")
        key = key.replace("\r\\n", "")
        key = key.replace(" &amp;lt;br /&amp;gt;&amp;lt;br /&amp;gt; ", "")
        key = key.replace("RussianEnglishSpanish - SpainFrenchJapaneseCzech", "")
        language_count[key] = language_count.get(key, 0) + 1


# In[106]:


language_count


# In[107]:


language_count = pd.DataFrame.from_dict(language_count, orient='index').reset_index()


# In[108]:


language_count.columns = ['language', 'Frequency']
language_count = language_count.sort_values('Frequency', ascending = False).head(10)


# In[109]:


language_count


# ## Average Downloads for Games with Support vs Games without Support

# In[110]:


for index, row in df.iterrows():
    if  df.at[index, 'Support email'] == "no Support email available":
        df.at[index, 'Game have support'] = False
    else:
        df.at[index, 'Game have support'] = True


# In[111]:


downloads_for_supported_games = df.groupby("Game have support").agg({
    "owners clean":"mean"
}).reset_index()
downloads_for_supported_games


# ## Most Supported Operating System

# In[112]:


average_downloads_for_different_os = df.groupby("OS").agg({
    "owners clean":"mean"
}).reset_index()
average_downloads_for_different_os


# ## Top Games with 'Positive' & 'Negative' Rating

# In[113]:


top_positive_ratings_games = df.groupby("Name").agg({
    "Positive":"sum"
}).reset_index().sort_values("Positive",ascending = False).head(10)
top_positive_ratings_games


# In[114]:


top_positive_ratings_games = df.groupby("Name").agg({
    "Positive":"sum"
}).reset_index().sort_values("Positive",ascending = False).head(10)
top_positive_ratings_games


# ## Most Downloaded games with 'Positive' & 'Negative' Rating 

# In[115]:


top_downloaded_games_with_rating = df.groupby("Name").agg({
    "owners clean":"sum",
    "Negative":"sum",
    "Positive":"sum"
    
}).reset_index().sort_values("owners clean",ascending = False).head(10)
top_downloaded_games_with_rating


# # Top davelopers -

# Downloads

# In[116]:


top_developers_total_downloads = df.groupby("Developers").agg({
    "owners clean":"sum"
}).reset_index().sort_values("owners clean",ascending = False).head(10)
top_developers_total_downloads


# Ratings

# In[117]:


top_developers_downloads_with_positive_and_negative= df.groupby("Developers").agg({
    "owners clean":"sum",
    "Positive":"sum",
    "Negative":"sum"
}).reset_index().sort_values("owners clean",ascending = False).head(10)
top_developers_downloads_with_positive_and_negative


# Positive Rating

# In[118]:


top_developers_total_positive_ratings = df.groupby("Developers").agg({
    "Positive":"sum"
}).reset_index().sort_values("Positive",ascending = False).head(10)
top_developers_total_positive_ratings


# Negative Rating

# In[119]:


top_developers_total_negative_ratings = df.groupby("Developers").agg({
    "Negative":"sum"
}).reset_index().sort_values("Negative",ascending = False).head(10)
top_developers_total_negative_ratings


# ## Top publishers-

# Downloads

# In[120]:


top_publishers_total_downloads = df.groupby("Publishers").agg({
    "owners clean":"sum"
}).reset_index().sort_values("owners clean",ascending = False).head(10)
top_publishers_total_downloads


# Ratings

# In[121]:


top_Publishers_downloads_with_positive_and_negative= df.groupby("Publishers").agg({
    "owners clean":"sum",
    "Positive":"sum",
    "Negative":"sum"
}).reset_index().sort_values("owners clean",ascending = False).head(10)
top_Publishers_downloads_with_positive_and_negative


# ## Categories

# In[122]:


df["Categories"][0]


# In[123]:


def categories_clean(x):
    x = x.split(",")
    return x


# In[124]:


categories_clean('Single-player,Multi-player,Steam Achievements,Partial Controller Support')


# In[125]:


categories_count = {}
for categories in df['Categories']:
    words = categories_clean(categories)
    for word in words:
        categories_count[word] = categories_count.get(word, 0) + 1
categories_count


# In[126]:


categories_count = pd.DataFrame.from_dict(categories_count, orient='index').reset_index()


# In[127]:


categories_count.columns = ['category', 'Frequency']
categories_count = categories_count.sort_values('Frequency', ascending = False).reset_index()
categories_count


# In[128]:


categories_count.head(10)


# In[129]:


for index, row in categories_count.iterrows():
    if categories_count.at[index , "Frequency"] < 6000 :
        categories_count.at[index , "category"] = "others"


# In[130]:


categories_count


# In[131]:


new_categories_count = categories_count.groupby("category").agg({
    "Frequency":"sum"
}).reset_index()
new_categories_count


# Average Downloads for Top Categories

# In[132]:


df_categories = df[["Name","owners clean","Price","Categories"]].copy()
df_categories


# In[133]:


for index, row in df_categories.iterrows():
    if "Single-player" in row["Categories"]:
        df_categories.at[index , "Single-player"] = True
for index, row in df_categories.iterrows():
    if "Steam Achievements" in row["Categories"]:
        df_categories.at[index ,"Steam Achievements"] = True
for index, row in df_categories.iterrows():
    if "Steam Cloud" in row["Categories"]:
        df_categories.at[index ,"Steam Cloud"] = True

df_categories


# In[134]:


df_categories.groupby("Single-player").agg({
    "owners clean":"mean"
})


# In[135]:


df_categories.groupby("Steam Achievements").agg({
    "owners clean":"mean"
})


# In[136]:


df_categories.groupby("Steam Cloud").agg({
    "owners clean":"mean"
})


# ### Top genre 

# In[137]:


df["Genres"][0]


# In[138]:


def genres_clean(x):
    x = x.split(",")
    return x


# In[139]:


genres_clean('Casual,Indie,Sports')


# In[140]:


genres_count = {}
for genres in df['Genres']:
    words = genres_clean(genres)
    for word in words:
        genres_count[word] = genres_count.get(word, 0) + 1
genres_count


# In[141]:


genres_count = pd.DataFrame.from_dict(genres_count, orient='index').reset_index()
genres_count.columns = ['Genres', 'Frequency']
genres_count = genres_count.sort_values('Frequency', ascending = False)
genres_count


# In[142]:


genres_count.head(10)


# In[143]:


for index, row in genres_count.iterrows():
    if genres_count.at[index , "Frequency"] < 3000 :
        genres_count.at[index , "Genres"] = "others"


# In[144]:


genres_count


# ### Average Downloads for Top Genres

# In[145]:


df_Genres = df[["Name","owners clean","Price","Genres"]].copy()
df_Genres


# In[146]:


for index, row in df_Genres.iterrows():
    if "Indie" in row["Genres"]:
        df_Genres.at[index , "Indie"] = True
for index, row in df_Genres.iterrows():
    if "Action" in row["Genres"]:
        df_Genres.at[index ,"Action"] = True
for index, row in df_Genres.iterrows():
    if "Casual" in row["Genres"]:
        df_Genres.at[index ,"Casual"] = True


# In[147]:


df_Genres.groupby("Indie").agg({
    "owners clean":"mean"
})


# In[148]:


df_Genres.groupby("Action").agg({
    "owners clean":"mean"
})


# In[149]:


df_Genres.groupby("Casual").agg({
    "owners clean":"mean"
})


# ### Years with Most Number of Game Production

# In[150]:


df['year'] = df['Release date'].dt.year


# In[151]:


df['year'].unique()


# In[152]:


years = df.groupby("year").agg({
    "Name":"count"
}).reset_index()
years.sort_values("Name",ascending = False)


# In[ ]:




