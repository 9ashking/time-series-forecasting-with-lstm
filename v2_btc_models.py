#!/usr/bin/env python
# coding: utf-8

# If you do not have these libraries, you can install them via pip.

# In[1]:


pip install yfinance


# We need to load the following libraries:

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_log_error,mean_absolute_error


# In[3]:


from keras.models import Sequential
from keras.layers import Dropout,Dense,BatchNormalization ,LSTM ,GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model


# In[4]:


import yfinance as yf
import datetime as time


# get data as yahoo finance

# In[5]:


btc_data=yf.download("BTC-USD",start='2014-09-17',stop=time.date.today())


# In[6]:


min_date = str(btc_data.index.min())
max_date = str(btc_data.index.max())
print("We have collected Bitcoin stock price data from %s to %s (today)" %( min_date , max_date))


# In[13]:


btc_data.drop(btc_data.tail().index, axis =0, inplace = True)


# In[14]:


data = btc_data.to_csv("D:\Btc\data.csv")


# In[15]:


dataaa = pd.read_csv("D:\Btc\data.csv")
dataaa.head()


# In[7]:


min_date = str(btc_data.index.min())
max_date = str(btc_data.index.max())
print("We have collected Bitcoin stock price data from %s to %s (today)" %( min_date , max_date))


# see the data shape

# In[8]:


print("our data have %d rows and %d columns(Features)"%(btc_data.shape[0],btc_data.shape[1]))


# In[9]:


btc_data.head()


# In[10]:


btc_data.describe()


# In[11]:


btc_data.info()


# In[12]:


print(btc_data.isna().sum())


# In[13]:


print("The dataset contains %d duplicate data"%(btc_data.duplicated().sum()))


# In[14]:


plt.figure(figsize=(11,15),dpi=100)
plt.subplot(411)
btc_data.Close.plot()
plt.title("Close Values",loc="left")
plt.subplot(412)
btc_data.Open.plot(c="r")
plt.title("Open Values",loc="right")
plt.subplot(413)
btc_data.High.plot(c="g")
plt.title("High Values",loc="left")
plt.subplot(414)
btc_data.Low.plot(c="y")
plt.title("low values",loc="right")


# In[15]:


plt.figure(figsize=(15,6))
sns.scatterplot(x= btc_data.High, y= btc_data.Low)


# In[16]:


plt.figure(figsize = (30,10),dpi = 100)  
plt.plot(btc_data["Volume"],linestyle = "--" , color = "k",marker = "*",markerfacecolor = "red")
plt.xlabel("Volume of btcoin")
plt.grid()


# In[17]:


btc_data.drop("Volume" , axis = 1).plot(figsize = (11,5))
plt.legend(bbox_to_anchor=(1,1.03))


# In[18]:


plt.figure(figsize = (15,5))
c_df = btc_data.corr()
sns.heatmap(c_df ,annot =True , linewidths =0.1 )


# In[19]:


plt.figure(figsize = (15,7))
sns.kdeplot(x ="Open" ,y ="Volume" ,data =btc_data)
plt.scatter(btc_data["Open"], btc_data["Volume"],color = "red")


# In[20]:


sns.displot(data = btc_data ,x =btc_data["Close"], y =btc_data["Open"])


# In[21]:


fig = px.line(x= btc_data.index.values , y= btc_data["Low"])
fig.add_bar(x= btc_data.index.values , y= btc_data["High"])
fig.add_scatter(x= btc_data.index.values , y= btc_data["High"])
fig.add_bar(x= btc_data.index.values , y= btc_data["Low"])
fig.update_layout(font_family="Courier New",
    font_color="blue",
    title_font_family="Times New Roman",
    title_font_color="red",
    legend_title_font_color="green")
fig.update_xaxes(title_font_family="Arial")


# train test split

# In[22]:


x = btc_data.drop("Close",axis=1)
y = btc_data.Close
x_full_train,x_test,y_full_train,y_test = train_test_split(x,y,test_size=0.2)


# validation data

# In[23]:


x_train,x_val,y_train,y_val=train_test_split(x_full_train,y_full_train,test_size=0.2)


# In[49]:


print("shape of x train  is :",x_train.shape)
print("shape of y train is  :", y_train.shape)

print("shape of x test is  :", x_test.shape)
print("shape of y test is  :", y_test.shape)


print("shape of x val is  :", x_val.shape)
print("shape of y val is  :", y_val.shape)


# Construction of neural network(ann)

# In[24]:


ann_model = Sequential()
ann_model.add(Dense(45,activation="relu"))
ann_model.add(Dropout(0.3))
ann_model.add(Dense(60,activation="linear"))
ann_model.add(BatchNormalization())
ann_model.add(Dense(40,activation="relu"))
ann_model.add(Dense(1,activation="linear"))


# compile NN

# In[25]:


ann_model.compile(optimizer="adam",loss="mean_squared_error")


# Training neural network with data

# In[26]:


history = ann_model.fit(x_train,y_train,epochs=100,batch_size=32,validation_data=(x_val,y_val))


# Schematic drawing of the model

# In[27]:


plot_model(ann_model)


# Drawing a graph of the process of changing the error rate of the model in each epoch of training:
# 
# 1. The error rate on the training data has gradually decreased
# 
# 2. The error rate on the validation data has gradually decreased  
#   
# 

# In[28]:


pd.DataFrame(history.history).plot()


# predict:

# In[29]:


y_pred = ann_model.predict(x_test)
y_=pd.DataFrame(data=y_pred,index=y_test.index,columns=["y_pred"])
y_["y_test"]=y_test.values


# In[30]:


y_.sort_index(inplace =True)


# In[31]:


fig = px.bar(x= y_.index.values , y= y_["y_test"])
fig.add_scatter(x= y_.index.values , y= y_["y_test"])


# In[32]:


fig = px.bar(x= y_.index.values , y= y_["y_pred"])
fig.add_scatter(x= y_.index.values , y= y_["y_pred"])


# In[33]:


y_.plot(figsize = (15,5))


# Claculation model error using different metrics:
# 
# 1.   r2_score
# 2.   mean_squared_log_error
# 3.   mean_absolute_error

# In[36]:


print("r2 score:" ,r2_score(y_["y_test"],y_["y_pred"]))
print("mean squared error: ", mean_squared_log_error(y_["y_test"],y_["y_pred"]))
print("mean absolute erroe:",mean_absolute_error(y_["y_test"],y_["y_pred"]))


# Create and fit the LSTM network

# In[52]:


lstm_model = Sequential()
lstm_model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
lstm_model.add(LSTM(15))
lstm_model.add(Dense(30))
lstm_model.add(Dropout(0.25))
lstm_model.add(Dense(1))


# In[53]:


lstm_model.compile(loss='mean_squared_error', optimizer='adam')


# In[54]:


history = lstm_model.fit(x_train,y_train,epochs=100,batch_size=32,validation_data=(x_val,y_val))


# Schematic drawing of the model

# In[57]:


plot_model(lstm_model)


# Drawing a graph of the process of changing the error rate of the model in each epoch of training:
# 
# 1. The error rate on the training data has gradually decreased
# 
# 2. The error rate on the validation data has gradually decreased  
#   
# 

# In[58]:


pd.DataFrame(history.history).plot()


# make predictions

# In[61]:


y_pred = lstm_model.predict(x_test)
y_lstm=pd.DataFrame(data=y_pred,index=y_test.index,columns=["y_pred"])
y_lstm["y_test"]=y_test.values
y_lstm.sort_index(inplace =True)


# In[63]:


fig = px.bar(x= y_lstm.index.values , y= y_lstm["y_pred"])
fig.add_scatter(x= y_lstm.index.values , y= y_lstm["y_pred"])

fig.add_bar(x= y_lstm.index.values , y= y_lstm["y_test"])
fig.add_scatter(x= y_lstm.index.values , y= y_lstm["y_test"])


# In[64]:


y_lstm.plot(figsize = (15,5))


# Claculation model error using different metrics:
# 
# 1.   r2_score
# 2.   mean_squared_log_error
# 3.   mean_absolute_error

# In[65]:


print("r2 score:" ,r2_score(y_lstm["y_test"],y_lstm["y_pred"]))
print("mean squared error: ", mean_squared_log_error(y_lstm["y_test"],y_lstm["y_pred"]))
print("mean absolute erroe:",mean_absolute_error(y_lstm["y_test"],y_lstm["y_pred"]))


# In[67]:





# In[ ]:




