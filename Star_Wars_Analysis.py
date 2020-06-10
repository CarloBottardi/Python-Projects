#!/usr/bin/env python
# coding: utf-8

# In[119]:


import pandas as pd
star_wars = pd.read_csv("star_wars.csv", encoding="ISO-8859-1")


# In[120]:


print(star_wars.columns)


# In[121]:


print(star_wars.head(10))


# In[122]:


total_rows=len(star_wars.index)
print(total_rows)


# In[123]:


star_wars=star_wars[pd.notnull(star_wars["RespondentID"])]
total_rows=len(star_wars.index)
print(total_rows)


# In[124]:


yes_no = {
    "Yes": True,
    "No": False
}
seen_films="Have you seen any of the 6 films in the Star Wars franchise?"
fan="Do you consider yourself to be a fan of the Star Wars film franchise?"


# In[125]:


star_wars[seen_films] = star_wars[seen_films].map(yes_no)
star_wars[fan] = star_wars[fan].map(yes_no)


# In[126]:


print(star_wars[seen_films].value_counts(dropna=False))
print(star_wars[fan].value_counts(dropna=False))


# In[127]:


print(star_wars.columns[3:9])
print(star_wars.iloc[:,3:9].head(3))


# In[128]:


import numpy as np
bool_map = {
    "Star Wars: Episode I  The Phantom Menace": True,
    "Star Wars: Episode II  Attack of the Clones": True,
    "Star Wars: Episode III  Revenge of the Sith": True,
    "Star Wars: Episode IV  A New Hope": True,
    "Star Wars: Episode V The Empire Strikes Back": True,
    "Star Wars: Episode VI Return of the Jedi": True,
    np.NaN: False
}


# In[129]:


which_film="Which of the following Star Wars films have you seen? Please select all that apply."
star_wars[which_film] = star_wars[which_film].map(bool_map)
star_wars["Unnamed: 4"] = star_wars["Unnamed: 4"].map(bool_map)
star_wars["Unnamed: 5"] = star_wars["Unnamed: 5"].map(bool_map)
star_wars["Unnamed: 6"] = star_wars["Unnamed: 6"].map(bool_map)
star_wars["Unnamed: 7"] = star_wars["Unnamed: 7"].map(bool_map)
star_wars["Unnamed: 8"] = star_wars["Unnamed: 8"].map(bool_map)


# In[130]:


columns_ren={
    "Which of the following Star Wars films have you seen? Please select all that apply.": "seen_1",
    "Unnamed: 4":"seen_2",
    "Unnamed: 5":"seen_3",
    "Unnamed: 6":"seen_4",
    "Unnamed: 7":"seen_5",
    "Unnamed: 8":"seen_6",
}
star_wars = star_wars.rename(columns=columns_ren)
print(star_wars.iloc[:,3:9].head(3))


# In[131]:


star_wars[star_wars.columns[9:15]] = star_wars[star_wars.columns[9:15]].astype(float)
columns_ren_2={
    "Please rank the Star Wars films in order of preference with 1 being your favorite film in the franchise and 6 being your least favorite film.": "ranking_1",
    "Unnamed: 10":"ranking_2",
    "Unnamed: 11":"ranking_3",
    "Unnamed: 12":"ranking_4",
    "Unnamed: 13":"ranking_5",
    "Unnamed: 14":"ranking_6",
}
star_wars = star_wars.rename(columns=columns_ren_2)
print(star_wars.columns)


# In[132]:


mean_rank=star_wars.iloc[:,9:15].mean(axis=0)
print(mean_rank)


# In[133]:


get_ipython().magic('matplotlib inline')
mean_rank.plot.bar()


# So far I've cleaned the columns 3:15 and I discovered that the favourite film according to the fans is "The empire strikes back".
# In fact it's an almost universal opinion that the original trilogy is better than the prequels

# In[134]:


how_much_seen=star_wars.iloc[:,3:9].sum(axis=0)
how_much_seen.plot.bar()


# This result could be explained in this way:
# "A new hope" was a new film and not everyone have seen it, but then the franchise became popular and the two sequels have reached more popularity.
# "Phantom menace" was hyped by the franchise popularity but disappointed the fans, consequently the two sequels were less watched.

# In[135]:


males = star_wars[star_wars["Gender"] == "Male"]
females = star_wars[star_wars["Gender"] == "Female"]


# In[136]:


mean_rank_males=males.iloc[:,9:15].mean(axis=0)
mean_rank_males.plot.bar()


# In[137]:


how_much_seen_males=males.iloc[:,3:9].sum(axis=0)
how_much_seen_males.plot.bar()


# In[138]:


mean_rank_females=females.iloc[:,9:15].mean(axis=0)
mean_rank_females.plot.bar()


# In[139]:


how_much_seen_females=females.iloc[:,3:9].sum(axis=0)
how_much_seen_females.plot.bar()


# The two genders agree each other

# In[140]:


grouped = star_wars.groupby('Education')
star_wars_ed=grouped.agg(np.mean)
#star_wars_ed.reset_index(inplace=True)
star_wars_ed.iloc[:,8:14].plot.bar(legend=False)


# In[141]:


grouped = star_wars.groupby('Location (Census Region)')
star_wars_loc=grouped.agg(np.mean)
star_wars_loc.iloc[:,8:14].plot.bar(legend=False)


# In[142]:


grouped = star_wars.groupby('Which character shot first?')
star_wars_sh=grouped.agg(np.mean)
star_wars_sh.iloc[:,8:14].plot.bar(legend=False)


# In[143]:


print(star_wars.iloc[:,15:29].head(4))


# In[144]:


columns_ren_3={
    "Please state whether you view the following characters favorably, unfavorably, or are unfamiliar with him/her.": "Han",
    "Unnamed: 16":"Luke",
    "Unnamed: 17":"Leia",
    "Unnamed: 18":"Anakin",
    "Unnamed: 19":"Obi Wan",
    "Unnamed: 20":"Palpatine",
    "Unnamed: 21":"Darth Vader",
    "Unnamed: 22":"Lando",
    "Unnamed: 23":"Boba Fett",
    "Unnamed: 24":"C-3P0",
    "Unnamed: 25":"R2 D2",
    "Unnamed: 26":"Jar Jar",
    "Unnamed: 27":"Padme",
    "Unnamed: 28":"Yoda",
}
star_wars = star_wars.rename(columns=columns_ren_3)


# In[145]:


print(star_wars["Darth Vader"].value_counts())


# In[146]:


def favorable(el):
    if el=="Very favorably":
        return 4
    elif el=="Somewhat favorably":
        return 3
    elif el=="Neither favorably nor unfavorably (neutral)":
        return 2
    elif el=="Somewhat unfavorably":
        return 1
    elif el=="Very unfavorably":
        return 0
    else:
        return np.nan
star_wars.iloc[:,15:29]=star_wars.iloc[:,15:29].applymap(favorable)


# In[147]:


fav_sum=star_wars.iloc[:,15:29].sum(axis=0)
fav_sum.plot.bar()


# In[ ]:





# In[ ]:




