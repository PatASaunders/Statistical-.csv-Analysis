#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re

regex = re.compile("^.+(sub-.+)_(ses-.+)_(mod-.+)")
print(regex)


# In[3]:


strings = ["abcsub-033_ses-01_mod-mri", "defsub-044_ses-01_mod-mri", "ghisub-055_ses-02_mod-ctscan" ]
print([regex.findall(s)[0] for s in strings])


# In[4]:


import os
cwd = os.getcwd()
print(cwd)


# In[5]:


os.chdir(cwd)


# In[6]:


import tempfile

tmpdir = tempfile.gettempdir()
print(tmpdir)


# In[21]:


#Exercise 1

def calc(x,y,z="plus"):
    if z=="minus":
        return x-y
    if z=="times":
        return x*y
    if z=="divide":
        return x/y
    if z == "plus":
        return x+y
    else:
        return "Error, Unknown Operation"

calc(3,4, "minus")


# In[22]:


#Exercise 2

list=[1,1,2,2,3,3,4,4,5,5,9,9,4,8,8,6,4,8,9,9,3,6,3]
list2=[]
x=None
for i in list:
    if i != x:
        list2.append(i)
    x=i

print(list2)


# In[85]:


#Exercise 3

import re

bd = open('C:\\Users\\Patrice\\Desktop\\BSD-4.txt', encoding = 'utf-8')
bd = bd.read()
bd = re.sub(r'[^a-zA-Z\n ]',"", bd)
bd = bd.lower().split()

dict = {}

# 2 Methods:

for i in bd:
    dict[i] = dict.get(i, 0) + 1

#for i in bd:
#    if i in dict:
#        dict[i] += 1
#    else:
#        dict[i] = 1
        
print(dict)


# In[51]:


#Exercise 4

import statistics as stats

class Employee:
    def __init__(self, name, years_of_service):
        self.name = name
        self.years_of_service = years_of_service
    
    pay = 1500
    ppy = 100
    
    def __str__(self):
        return self.name + " " + str(self.years_of_service)
    
    def salary(self):
        return self.pay + (self.ppy*self.years_of_service)

class Manager(Employee):
    pay = 2500
    ppy = 120
    
emp_dict = {'Lucy':[4, 'Manager'], 'Simon':[5,'Employee'], 'Steven':[8, 'Employee'], 'Gavin':[4, 'Manager'], 'Annie':[10, 'Employee']}
#new_list = []
mean_list = []

for key, value in emp_dict.items():
    temp_list = []
    if value[1] == 'Manager':
        employee = Manager(key, value[0])
    else:
        employee = Employee(key, value[0])
    temp_list.append(str(employee.name))
    temp_list.append(str(employee.salary()))
    mean_list.append(int(employee.salary()))
#    new_list.append(temp_list)
    print(temp_list) #also optional

#print(new_list)
print(stats.mean(mean_list))


# In[72]:


#Exercise 5

import numpy as np
import matplotlib.pyplot as plt

X = np.random.randn(4,2)

def min_finder():
    index_min = np.argmin(X)

min_finder()
index_min


# In[37]:


#Exercise 6

import pandas as pd
import numpy as np

url = 'https://raw.githubusercontent.com/neurospin/pystatsml/master/datasets/iris.csv'

df = pd.read_csv(url, error_bad_lines=False)

new_col = df._get_numeric_data().columns.tolist()
data_list = []

for j in new_col:
    m_val = df.groupby('species', as_index=True)[j].mean()
    data_list.append(m_val)

final_df = pd.concat(data_list, axis=1, sort = False, join = 'outer')
final_df.reset_index(level=0, inplace=True)
final_df


# In[ ]:


#Exercise 7

import pandas as pd

columns = ['name', 'age', 'gender', 'job']

user1 = pd.DataFrame([['alice', 19, "F", "student"],['john', 26, "M", "student"]],columns=columns)
user2 = pd.DataFrame([['eric', 22, "M", "student"],['paul', 58, "F", "manager"]],columns=columns)
user3 = pd.DataFrame(dict(name=['peter', 'julie'],age=[33, 44], gender=['M', 'F'],job=['engineer', 'scientist']))
user4 = pd.DataFrame(dict(name=['alice', 'john', 'eric', 'julie'],height=[165, 180, 175, 171]))

users = pd.concat([user1, user2, user3])
users = pd.merge(users, user4, how='outer')

df = users.copy()
df.ix[[0, 2], "age"] = None
df.ix[[1, 3], "gender"] = None

def na_filler():
    df.fillna(df.mean(), inplace=True)
    df.select_dtypes(include=['object']).fillna(df.mode(), inplace=True)
    int_cols = df._get_numeric_data().columns.tolist()
    all_cols = df.columns.tolist()
    for i in all_cols:
        if i not in int_cols:
            df[i].fillna(df[i].mode()[0], inplace=True)
        else:
            pass
na_filler()

with pd.ExcelWriter('Desktop\\users.xlsx') as writer:
    users.to_excel(writer, sheet_name = 'original', index=False)
    df.to_excel(writer, sheet_name='imputed', index = False)

