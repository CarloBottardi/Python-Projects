#!/usr/bin/env python
# coding: utf-8

# # Read in the data

# In[1]:


import pandas as pd
import numpy
import re

data_files = [
    "ap_2010.csv",
    "class_size.csv",
    "demographics.csv",
    "graduation.csv",
    "hs_directory.csv",
    "sat_results.csv"
]

data = {}

for f in data_files:
    d = pd.read_csv("schools/{0}".format(f))
    data[f.replace(".csv", "")] = d


# # Read in the surveys

# In[2]:


all_survey = pd.read_csv("schools/survey_all.txt", delimiter="\t", encoding='windows-1252')
d75_survey = pd.read_csv("schools/survey_d75.txt", delimiter="\t", encoding='windows-1252')
survey = pd.concat([all_survey, d75_survey], axis=0)

survey["DBN"] = survey["dbn"]

survey_fields = [
    "DBN", 
    "rr_s", 
    "rr_t", 
    "rr_p", 
    "N_s", 
    "N_t", 
    "N_p", 
    "saf_p_11", 
    "com_p_11", 
    "eng_p_11", 
    "aca_p_11", 
    "saf_t_11", 
    "com_t_11", 
    "eng_t_11", 
    "aca_t_11", 
    "saf_s_11", 
    "com_s_11", 
    "eng_s_11", 
    "aca_s_11", 
    "saf_tot_11", 
    "com_tot_11", 
    "eng_tot_11", 
    "aca_tot_11",
]
survey = survey.loc[:,survey_fields]
data["survey"] = survey


# # Add DBN columns

# In[3]:


data["hs_directory"]["DBN"] = data["hs_directory"]["dbn"]

def pad_csd(num):
    string_representation = str(num)
    if len(string_representation) > 1:
        return string_representation
    else:
        return "0" + string_representation
    
data["class_size"]["padded_csd"] = data["class_size"]["CSD"].apply(pad_csd)
data["class_size"]["DBN"] = data["class_size"]["padded_csd"] + data["class_size"]["SCHOOL CODE"]


# # Convert columns to numeric

# In[4]:


cols = ['SAT Math Avg. Score', 'SAT Critical Reading Avg. Score', 'SAT Writing Avg. Score']
for c in cols:
    data["sat_results"][c] = pd.to_numeric(data["sat_results"][c], errors="coerce")

data['sat_results']['sat_score'] = data['sat_results'][cols[0]] + data['sat_results'][cols[1]] + data['sat_results'][cols[2]]

def find_lat(loc):
    coords = re.findall("\(.+, .+\)", loc)
    lat = coords[0].split(",")[0].replace("(", "")
    return lat

def find_lon(loc):
    coords = re.findall("\(.+, .+\)", loc)
    lon = coords[0].split(",")[1].replace(")", "").strip()
    return lon

data["hs_directory"]["lat"] = data["hs_directory"]["Location 1"].apply(find_lat)
data["hs_directory"]["lon"] = data["hs_directory"]["Location 1"].apply(find_lon)

data["hs_directory"]["lat"] = pd.to_numeric(data["hs_directory"]["lat"], errors="coerce")
data["hs_directory"]["lon"] = pd.to_numeric(data["hs_directory"]["lon"], errors="coerce")


# # Condense datasets

# In[5]:


class_size = data["class_size"]
class_size = class_size[class_size["GRADE "] == "09-12"]
class_size = class_size[class_size["PROGRAM TYPE"] == "GEN ED"]

class_size = class_size.groupby("DBN").agg(numpy.mean)
class_size.reset_index(inplace=True)
data["class_size"] = class_size

data["demographics"] = data["demographics"][data["demographics"]["schoolyear"] == 20112012]

data["graduation"] = data["graduation"][data["graduation"]["Cohort"] == "2006"]
data["graduation"] = data["graduation"][data["graduation"]["Demographic"] == "Total Cohort"]


# # Convert AP scores to numeric

# In[6]:


cols = ['AP Test Takers ', 'Total Exams Taken', 'Number of Exams with scores 3 4 or 5']

for col in cols:
    data["ap_2010"][col] = pd.to_numeric(data["ap_2010"][col], errors="coerce")


# # Combine the datasets

# In[7]:


combined = data["sat_results"]

combined = combined.merge(data["ap_2010"], on="DBN", how="left")
combined = combined.merge(data["graduation"], on="DBN", how="left")

to_merge = ["class_size", "demographics", "survey", "hs_directory"]

for m in to_merge:
    combined = combined.merge(data[m], on="DBN", how="inner")

combined = combined.fillna(combined.mean())
combined = combined.fillna(0)


# # Add a school district column for mapping

# In[8]:


def get_first_two_chars(dbn):
    return dbn[0:2]

combined["school_dist"] = combined["DBN"].apply(get_first_two_chars)


# # Find correlations

# In[9]:


correlations = combined.corr()
correlations = correlations["sat_score"]
print(correlations)


# # Plotting survey correlations

# In[10]:


# Remove DBN since it's a unique identifier, not a useful numerical value for correlation.
survey_fields.remove("DBN")


# In[11]:


get_ipython().magic('matplotlib inline')


# In[24]:


correlations_sat_score = combined.corr()
correlations_sat_score=correlations_sat_score.loc[survey_fields,["sat_score"]]
correlations_sat_score.plot.bar()


# It seems the columns more correlated with "sat_score" are:
# - the numbers of people respondent;
# - the safety and respect score based on teacher, students and total responses;
# - the students' academic expectations
# 

# In[25]:


combined.plot.scatter(x="saf_s_11",y="sat_score")


# It's quite scattered, but it seems there are 2 clusters:
# - one that has the sat_score more or less "constant";
# - one that is positively correlated with the saf_s_11

# In[27]:


import numpy as np
grouped = combined.groupby('school_dist')
districts=grouped.agg(np.mean)
districts.reset_index(inplace=True)
print(districts["sat_score"])


# In[31]:


from mpl_toolkits.basemap import Basemap
m = Basemap(
    projection='merc', 
    llcrnrlat=40.496044, 
    urcrnrlat=40.915256, 
    llcrnrlon=-74.255735, 
    urcrnrlon=-73.700272,
    resolution='i'
)

m.drawmapboundary(fill_color='#85A6D9')
m.drawcoastlines(color='#6D5F47', linewidth=.4)
m.drawrivers(color='#6D5F47', linewidth=.4)

longitudes=districts["lon"].tolist()
latitudes=districts["lat"].tolist()
m.scatter(longitudes,latitudes,s=60,zorder=2,latlon=True,c=districts["saf_s_11"],cmap="summer")


# The sat_score seems quite well distribuited, maybe the score is a bit higher at Manhattan and Bronx.

# In[35]:


races=["white_per","asian_per","black_per","hispanic_per"]
correlations=combined.corr()
correlations=correlations["sat_score"]
correlations[races].plot.bar()


# The races correlations are not surprising, white are averagely richer and there are many asian well integrated in the society, while black and hispanic are generally poorer

# In[37]:


combined.plot.scatter(x="hispanic_per",y="sat_score")


# The plot looks like an Hyperbole branch, as expected

# In[41]:


high_hisp=combined[combined["hispanic_per"]>95]
#from pprint import pprint as pp
#pp(combined.columns)
print(high_hisp["SchoolName"])


# In these schools there are a lot of english learners

# In[43]:


high_score_few_hisp=combined[(combined["hispanic_per"]<10)&(combined["sat_score"]>1800)]
print(high_score_few_hisp["SchoolName"])


# Schools with teaching specialized in science, whose students passed an entry test

# In[44]:


genders=["male_per","female_per"]
correlations_gend=combined.corr()
correlations_gend=correlations_gend["sat_score"]
correlations_gend[genders].plot.bar()


# Girls are generally better students (not only in New York, in the whole 1st world), thus it's not surprising. 

# In[45]:


combined.plot.scatter(x="female_per",y="sat_score")


# Actually the plot looks very "cloudy". In fact the correlation was just 0.1.

# In[46]:


high_score_lot_females=combined[(combined["female_per"]>60)&(combined["sat_score"]>1700)]
print(high_score_lot_females["SchoolName"])


# Liberal arts schools having high academic standards.

# In[47]:


combined["ap_per"]=combined["AP Test Takers "]/combined["total_enrollment"]


# In[48]:


combined.plot.scatter(x="ap_per",y="sat_score")


# There are clearly 2 clusters:
# - one seems highly and positively correlated;
# - the other is almost constant.

# In[49]:


combined.plot.scatter(x="AVERAGE CLASS SIZE",y="sat_score")


# In[ ]:





# In[ ]:




