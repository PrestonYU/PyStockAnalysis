#!/usr/bin/env python
# coding: utf-8

# # 1.爬取單月資料

#  ## 1-a. 爬取鴻海公司單月「個股日本益比、殖利率及股價淨值比」資訊

# In[76]:


import requests
import json
import pandas as pd


# In[77]:


data = {
    'response': 'json',
    'date': '20181001',
    'stockNo':'2317',
    '_':'1540911963420'
}
res = requests.get('http://www.twse.com.tw/exchangeReport/BWIBBU?response=json&date=20181001&stockNo=2317&_=1540911963420')
print(res.text)


# In[78]:


jres = json.loads(res.text)
jres


# In[79]:


jres['stat']


# In[80]:


jres['data']


# In[81]:


df_temp = pd.DataFrame(jres['data'],columns=jres['fields'])
df_temp


#  ## 1-b. 爬取鴻海公司單月「各日成交資訊」資訊

# In[82]:


data = {
    'response': 'json',
    'date': '20181001',
    'stockNo':'2317',
    '_':'1540912291297'
}
res2 = requests.get('http://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date=20181001&stockNo=2317&_=1540912291297')
print(res2.text)


# In[83]:


jres2 = json.loads(res2.text)
jres2


# In[84]:


jres2['stat']


# In[85]:


jres2['data']


# In[86]:


df_temp2 = pd.DataFrame(jres2['data'],columns=jres2['fields'])
df_temp2


# ## 1-c. 合併兩張表：單月「個股日本益比、殖利率及股價淨值比」資訊 ＋ 單月「各日成交資訊」資訊

# In[87]:


#先預處理df_temp的資料格式
df_temp['日期'] = df_temp['日期'].str.split("年").str.get(0)+"/"+ df_temp['日期'].str.split("年").str.get(1).str.split("月").str.get(0)+"/"+ df_temp['日期'].str.split("年").str.get(1).str.split("月").str.get(1).str.split("日").str.get(0)


# In[88]:


#格式已改變
df_temp['日期']


# In[89]:


#Left join兩筆資料
df_temp_final = pd.merge(df_temp,df_temp2, left_on='日期', right_on='日期', how='left')
df_temp_final

#最終大表呈現的模樣


# # 2.爬取多月資訊

# In[90]:


import datetime
import calendar
import time

time = datetime.date(2018, 10, 1) 

#求前一個月的第一天
first_day = datetime.date(time.year, time.month, 1)
pre_month = first_day - datetime.timedelta(days = 1) 
first_day_of_pre_month = datetime.date(pre_month.year, pre_month.month, 1)
first_day_of_pre_month


# In[91]:


#製作爬蟲回傳存取目標Dataframe
column_list = list(df_temp.columns)
df = pd.DataFrame(columns=column_list)
df


# In[92]:


from datetime import datetime
from datetime import timedelta
import time

#開始爬蟲
crawl_date = datetime(2018,11,30) # start_date
df = df_temp

#第一份資料：單月「個股日本益比、殖利率及股價淨值比」資訊
for i in range(18):
    crawl_date -= timedelta(29)
    first_day = datetime(crawl_date.year, crawl_date.month, 1)
    pre_month = first_day - timedelta(days = 1) 
    first_day_of_pre_month = datetime(pre_month.year, pre_month.month, 1)
    crawl_date_str = datetime.strftime(first_day_of_pre_month, '%Y%m%d')
    
    res = requests.get('http://www.twse.com.tw/exchangeReport/BWIBBU?response=json&date=' + crawl_date_str + '&stockNo=2317&_=1540911963420')
    jres = json.loads(res.text)
    
    # 證交所回覆有資料
    if(jres['stat']=='OK'):
        print(crawl_date_str, ': crawling data...')
        
        # 將讀取回的json轉成的DataFrame(df_temp)
        df_temp = pd.DataFrame(jres['data'],columns=jres['fields'])
        
        # 更改df_temp的日期資料格式
        df_temp['日期'] = df_temp['日期'].str.split("年").str.get(0)+"/"+         df_temp['日期'].str.split("年").str.get(1).str.split("月").str.get(0)+"/"+         df_temp['日期'].str.split("年").str.get(1).str.split("月").str.get(1).str.split("日").str.get(0)
        
        # 欄位合併
        df = df.append(df_temp)
        
    else:
        print(crawl_date_str, ': no data')
        
    # 讓程式睡個10秒再繼續爬取下一天資料，避免頻繁抓取被台灣證券交易所封鎖IP拒絕存取
    time.sleep(10) 


# In[93]:


#製作爬蟲回傳存取目標Dataframe
column_list = list(df_temp2.columns)
df2 = pd.DataFrame(columns=column_list)
df2


# In[97]:


from datetime import datetime
import time 

#開始爬蟲
crawl_date = datetime(2018,11,30) # start_date
df2 = df_temp2

#第二份資料：單月「各日成交資訊」資訊
for i in range(18):
    crawl_date -= timedelta(29)
    first_day = datetime(crawl_date.year, crawl_date.month, 1)
    pre_month = first_day - timedelta(days = 1) 
    first_day_of_pre_month = datetime(pre_month.year, pre_month.month, 1)
    crawl_date_str = datetime.strftime(first_day_of_pre_month, '%Y%m%d')
    
    res2 = requests.get('http://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date=' + crawl_date_str + '&stockNo=2317&_=1540912291297')
    jres2 = json.loads(res2.text)

    # 證交所回覆有資料
    if(jres2['stat']=='OK'):
        print(crawl_date_str, ': crawling data...')
        
        # 將讀取回的json轉成的DataFrame(df_temp)
        df_temp2 = pd.DataFrame(jres2['data'],columns=jres2['fields'])
                
        # 欄位合併
        df2 = df2.append(df_temp2)
        
    else:
        print(crawl_date_str, ': no data')
        
    # 讓程式睡個15秒再繼續爬取下一天資料，避免頻繁抓取被台灣證券交易所封鎖IP拒絕存取
    time.sleep(15)


# In[98]:


# 整合這兩張表
df_all = pd.merge(df,df2, left_on='日期', right_on='日期', how='left')
pd.set_option('display.max.columns',30)
pd.set_option('display.max.rows',300)
df_all


# # 3.資料預處理

# In[99]:


# X0.00的部分應該直接換成0
df_all = df_all.replace('X0.00',0)
df_all


# In[100]:


df_all["成交股數"] = df_all["成交股數"].str.split(",").str.get(0)+df_all["成交股數"].str.split(",").str.get(1)+ df_all["成交股數"].str.split(",").str.get(2)
df_all


# In[101]:


df_all["成交金額"] = df_all["成交金額"].str.split(",").str.get(0)+df_all["成交金額"].str.split(",").str.get(1)+ df_all["成交金額"].str.split(",").str.get(2)+df_all["成交金額"].str.split(",").str.get(3)
df_all


# In[102]:


df_all["成交筆數"] = df_all["成交筆數"].str.split(",").str.get(0)+df_all["成交筆數"].str.split(",").str.get(1)
df_all


# In[103]:


# 重複的日期刪除
df_all = df_all.drop_duplicates(subset=None, keep='first', inplace=False)
df_all


# In[104]:


df_all['日期'].size


# In[105]:


#依照日期時間升冪排列
df_all = df_all.sort_values("日期")
df_all


# In[106]:


#重設index
df_all = df_all.reset_index(drop=True)
pd.set_option('display.max.columns',30)
pd.set_option('display.max.rows',400)
df_all


# In[107]:


for i in range(len(df_all)):
    df_all.loc[i,'日期'] = str(int(df_all.loc[i,'日期'][:3])+1911) + df_all.loc[i,'日期'][3:]


# In[109]:


#檢查日期格式已改為西元年
df_all


# In[110]:


del df_all["股利年度"]


# In[111]:


del df_all["財報年/季"]


# In[116]:


df_all["日期"] = df_all["日期"].str.split("/").str.get(0)+df_all["日期"].str.split("/").str.get(1)+ df_all["日期"].str.split("/").str.get(2)
df_all


# In[117]:


df_all = df_all.set_index(df_all['日期'], drop=True)
df_all.head()


# In[118]:


type(df.index)


# In[119]:


#轉成DatetimeTndex
df_all.index = pd.to_datetime(df_all.index,format='%Y%m%d')
type(df_all.index)


# In[121]:


del df_all["日期"]


# In[122]:


df_all.dtypes


# In[123]:


df_all["殖利率(%)"] = df_all["殖利率(%)"].astype(float)
df_all["本益比"] = df_all["本益比"].astype(float)
df_all["股價淨值比"] = df_all["股價淨值比"].astype(float)
df_all["成交股數"] = df_all["成交股數"].astype(int)
df_all["成交金額"] = df_all["成交金額"].astype(float)
df_all["開盤價"] = df_all["開盤價"].astype(float)
df_all["最高價"] = df_all["最高價"].astype(float)
df_all["最低價"] = df_all["最低價"].astype(float)
df_all["收盤價"] = df_all["收盤價"].astype(float)
df_all["漲跌價差"] = df_all["漲跌價差"].astype(float)
df_all["成交筆數"] = df_all["成交筆數"].astype(float)

df_all.dtypes


# # 4.視覺化

# In[124]:


import matplotlib.pyplot as plt
plt.rcParams['font.family']='SimHei' #顯示中文('SimHei' for MacOS)
plt.rcParams['axes.unicode_minus'] = False #正常顯示負號
plt.style.use('ggplot')
#圖片顯示於Jupyter Notebook上
get_ipython().run_line_magic('matplotlib', 'inline')


# In[211]:


df_all['收盤價'].plot(figsize=(10,8))


# In[126]:


df_all.loc[:,"開盤價":"收盤價"].plot(figsize=(20,16))


# In[158]:


df_all.loc[:,"成交股數":"成交金額"].plot(figsize=(20,16))


# In[210]:


df_all['本益比'].plot(figsize=(10,8))


# In[212]:


df_all['殖利率(%)'].plot(figsize=(10,8))


# In[213]:


df_all['股價淨值比'].plot(figsize=(10,8))


# In[128]:


df_all.plot(kind='scatter',x='殖利率(%)', y='本益比',figsize=(10,8))


# # 5.分析

# ## 相關分析

# In[129]:


df_all.loc[:,"殖利率(%)":"成交筆數"].corr()


# In[130]:


import seaborn as sns
corr = df_all.loc[:,"殖利率(%)":"成交筆數"].corr()

plt.figure(figsize=(20,20))
sns.heatmap(corr, square=True, annot=True)
plt.show()


# ## 統計分析

# In[131]:


#統計分析

df_all_stock = df_all.loc[:,"殖利率(%)":"成交筆數"]
df_all_stock.describe()


# In[132]:


df_all_stock.mean()


# # 6.機器學習模型預估

# ### 股票殖利率＝現金股利 / 股價 (越高越好)
# ### 本益比(PER) = 每股市價 / 每股盈餘(EPS) （越低越好）
# ### 股價淨值比(PBR) = 股票市值 / 每股淨值 （股價淨值比小於1時，代表現在比較便宜；股價淨值比大於1時，代表現在比較昂貴）

# In[133]:


X = df_all[['殖利率(%)','本益比','股價淨值比']]


# ## Kmeans分群

# In[134]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X)
X_std = sc.transform(X)


# In[135]:


from sklearn.cluster import KMeans
km = KMeans(n_clusters=5)
y_pred = km.fit_predict(X_std)


# In[136]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['font.family']='SimHei' #顯示中文('SimHei' for MacOS)

plt.figure(figsize=(10,8))
plt.scatter(X['殖利率(%)'],X['本益比'],c=y_pred)
plt.xlabel('殖利率(%)', fontsize=20)
plt.ylabel('本益比', fontsize=20)


# In[137]:


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10,8))
ax = Axes3D(fig)
ax.scatter(xs=X['殖利率(%)'],ys=X['股價淨值比'],zs=X['本益比'],c=y_pred)
ax.set_xlabel('殖利率(%)',fontsize=18)
ax.set_ylabel('股價淨值比',fontsize=18)
ax.set_zlabel('本益比',fontsize=18)


# ## 製作明日收盤價

# In[138]:


#製作明日收盤價
date_list = list(df_all.index)
for i in range(len(df_all)-1):
    df_all.loc[date_list[i],'明日收盤價'] = df_all.loc[date_list[i+1], '收盤價']
df_all


# In[139]:


df_all = df_all.dropna()
X = df_all[['開盤價','最高價','最低價','收盤價','漲跌價差','殖利率(%)','本益比','股價淨值比']]
y = df_all[['明日收盤價']]


# ### 切分資料

# In[141]:


#切分資料
X_train = X[:-1]
X_test = X[-1:]
y_train = y[:-1]
y_test = y[-1:]


# In[142]:


X_test


# In[143]:


y_test


# ### 標準化

# In[144]:


#標準化
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


# ### 訓練

# In[145]:


#訓練資料
from sklearn import linear_model

# linear regression 物件
regr = linear_model.LinearRegression()

# 訓練模型
regr.fit(X_train_std, y_train)


# In[146]:


regr.score(X_train_std, y_train)


# In[147]:


regr.coef_


# In[148]:


plt.figure(figsize=(16,6))
plt.plot(X_train.index, y_train.values, label='real')
plt.plot(X_train.index, regr.predict(X_train_std), label='predict')
plt.grid()
plt.legend()


# ### 預測單日收盤價

# In[149]:


print('2018/2/26 收盤價')
print('實際值', y_test.values)
print('預測值', regr.predict(X_test_std))
print('誤差百分比 =', (regr.predict(X_test_std)[0][0] - y_test.values[0][0])/y_test.values[0][0] * 100, '%')


# ## 製作明日漲跌價差

# In[150]:


#製作明日漲跌價差
df_all2 = df_all.copy()
date_list = list(df_all2.index)
for i in range(len(df_all2)-1):
    df_all2.loc[date_list[i],'明日漲跌價差'] = df_all2.loc[date_list[i+1], '漲跌價差']
df_all2


# In[151]:


df_all2 = df_all2.dropna()
X = df_all2[['開盤價','最高價','最低價','收盤價','漲跌價差','殖利率(%)','本益比','股價淨值比']]
y = df_all2[['明日漲跌價差']]


# ### 切分資料

# In[152]:


#切分資料
X_train = X[:-1]
X_test = X[-1:]
y_train = y[:-1]
y_test = y[-1:]


# ### 標準化

# In[153]:


#標準化
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


# ### 訓練

# In[154]:


#訓練資料
# linear regression 物件
regr = linear_model.LinearRegression()

# 訓練模型
regr.fit(X_train_std, y_train)


# In[155]:


regr.score(X_train_std, y_train)


# In[156]:


plt.rcParams['axes.unicode_minus'] = False #正常顯示負號
plt.figure(figsize=(16,6))
plt.plot(X_train.index, y_train.values,label='real')
plt.plot(X_train.index, regr.predict(X_train_std), label='predict')
plt.grid()
plt.legend()


# ### 預測單日漲跌價差

# In[157]:


print('2018/2/26 漲跌價差')
print('實際值', y_test.values)
print('預測值', regr.predict(X_test_std))
print('誤差百分比 =', (regr.predict(X_test_std)[0][0] - y_test.values[0][0])/y_test.values[0][0] * 100, '%')

