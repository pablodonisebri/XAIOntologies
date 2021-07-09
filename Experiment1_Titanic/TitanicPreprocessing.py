import pandas as pd
import numpy as np
import sklearn
from sklearn.impute import SimpleImputer
df_train=pd.read_csv("./train.csv")
df_test=pd.read_csv("./test.csv")

df=pd.concat([df_train,df_test])

"""
The columns are: 

PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
"""

# We drop the null values
df=df.dropna(subset=["Survived"],axis=0)

#Imputers for the categorical and numerical attributes
imp = SimpleImputer(strategy="most_frequent")
imp2 = SimpleImputer(strategy="mean")

df[["PassengerId","Cabin","Embarked"]]=imp.fit_transform(df[["PassengerId","Cabin","Embarked"]])

df[["PassengerId","Age"]]=imp2.fit_transform(df[["PassengerId","Age"]])





df=df.astype({'PassengerId': 'int32'})
df_survived=df[df["Survived"]==1]
df_unsurvived=df[df["Survived"]==0]

df=df.set_index('PassengerId')




df["Age"]=df["Age"].apply(func=lambda x:str(int(x)))

#Dictionary to change the number of the class by its name
dict2={1:"upper",2:"middle",3:"lower"}
df["Pclass"]=df["Pclass"].replace(dict2)


df["Name"]=df["Name"].apply(func=lambda x: x.replace(" ","_"))
df["Cabin"]=df["Cabin"].apply(func=lambda x: x.replace(" ","_"))
#Save it to csv
df.to_csv("./clean.csv")
