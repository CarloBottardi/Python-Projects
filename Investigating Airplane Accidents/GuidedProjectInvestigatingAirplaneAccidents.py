### Investigating Airplane Accidents ###

"""
Accidents are an unfortunate fact of air travel. 
Although flying is statistically safer than driving, minor and major flying accidents occur daily.
In this project, I'll work with a data set of airplane accident statistics to analyze patterns and look for any common threads.
The data set contains 77,282 aviation accidents that occurred in the U.S. and the metadata associated with them. 
The data in AviationData.txt file comes from the National Transportation Safety Board (NTSB).
"""

### Introduction

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

with open('AviationData.txt') as file:
    aviation_data = file.readlines()

aviation_list=[]
for item in aviation_data:
    element=item.split("|")
    element_def=[]
    for word in element:
        word=word.strip()
        element_def.append(word)
    aviation_list.append(element_def)

### Linear Algorithms

lax_code=[]
search_word="LAX94LA336"
for item in aviation_list:
    if search_word in item:
        lax_code.append(item)

"""
    The downsides of this technique is that
    the time consuming is O(n)
    where n number of rows of 
    aviation_list.
"""

columns_names = aviation_data[0].split(" | ")
aviation_data_def = aviation_data[1:]

aviation_dict_list=[]
for item in aviation_data_def:
    element=item.split(" | ")
    element_dict={}
    for i in range(len(columns_names)):
        element_dict[columns_names[i]]=element[i]
    aviation_dict_list.append(element_dict)

lax_dict=[]
for item in aviation_dict_list:
    if search_word in item.values():
        lax_dict.append(item)

"""
    Still linear O(n) since I still have to loop through
    a list of n items.
"""

### Accidents by U.S. State

dict_incidents_states={}
for event in aviation_dict_list:
    if event["Country"]=="United States":
        location=event["Location"]
        state=location[-2:]
        dict_incidents_states[state]=dict_incidents_states.get(state,0)+1

ser_incidents_states=pd.Series(dict_incidents_states)
ser_incidents_states=ser_incidents_states.sort_values(ascending=False)

### Fatalities and Injuries by Month

dict_serious_injuries={}
dict_fatal_injuries={}

for event in aviation_dict_list:
    date=event["Event Date"]
    month=date[:2]
    year=date[-4:]
    year_month=year+'/'+month
    serious_inj=event["Total Serious Injuries"]
    fatal_inj=event["Total Fatal Injuries"]
    if serious_inj=="":
        serious_inj=0
    else:
        serious_inj=int(serious_inj)
    if fatal_inj=="":
        fatal_inj=0
    else:
        fatal_inj=int(fatal_inj)
    dict_serious_injuries[year_month]=dict_serious_injuries.get(year_month,0)+serious_inj
    dict_fatal_injuries[year_month]=dict_fatal_injuries.get(year_month,0)+fatal_inj

ser_serious_injuries=pd.Series(dict_serious_injuries, name="Serious Injuries")
ser_fatal_injuries=pd.Series(dict_fatal_injuries, name="Fatal Injuries")

df_injuries = pd.concat([ser_serious_injuries, ser_fatal_injuries], axis=1)

### Accidents Map

latitudes=[]
longitudes=[]
for event in aviation_dict_list:
    if (event["Latitude"]!="")&(event["Longitude"]!=""):
        latitudes.append(float(event["Latitude"]))
        longitudes.append(float(event["Longitude"]))
m = Basemap(projection='merc', llcrnrlat=-80, urcrnrlat=80, llcrnrlon=-180, urcrnrlon=180)
x, y = m(longitudes, latitudes)
m.scatter(x, y, s=1)
m.drawcoastlines()
plt.show()
    





