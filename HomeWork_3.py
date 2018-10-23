
# coding: utf-8

# In[2]:


import lightgbm
import pandas as pd
import numpy as np
import time
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns

import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# In[3]:


columns_to_normalize = ['device', 'geoNetwork','totals', 'trafficSource']
def normalize_json_data(filename):
    path = filename
    df = pd.read_csv(path, converters={column: json.loads for column in columns_to_normalize}, 
                     dtype={'fullVisitorId': 'str'})
    
    for column in columns_to_normalize:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}_{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    return df

train = normalize_json_data("train.csv")
test = normalize_json_data("test.csv")


# In[5]:


train.shape


# In[6]:


train.head()


# In[7]:


train_numerical_features = train.select_dtypes(include=[np.number])


# In[8]:


test_numerical_features = test.select_dtypes(include=[np.number])


# In[9]:


train_category_features = train.select_dtypes(include=[np.object])
test_category_features = test.select_dtypes(include=[np.object])
train_category_features.columns
test_category_features.columns


# In[10]:


train = train.loc[:, (train != train.iloc[0]).any()]
test = test.loc[:, (test != test.iloc[0]).any()]


# In[11]:


train["totals_transactionRevenue"] = train["totals_transactionRevenue"].astype('float')


# In[12]:


for df in [train, test]:
    df['v_date'] = pd.to_datetime(df['visitStartTime'], unit='s')
    df['dayofweek'] = df['v_date'].dt.dayofweek
    df['hours'] = df['v_date'].dt.hour
    df['day'] = df['v_date'].dt.day
    df.drop('visitStartTime', axis=1)


# In[13]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

agg_dict = {}
for col in ["totals_bounces", "totals_hits", "totals_newVisits", "totals_pageviews", "totals_transactionRevenue"]:
    train[col] = train[col].astype('float')
    agg_dict[col] = "sum"
tmp = train.groupby("fullVisitorId").agg(agg_dict).reset_index()
tmp.head()


# In[14]:


constant_columns = []
for column in train.columns:
    if len(train[column].value_counts()) == 1:
        constant_columns.append(column)

irrelevant_columns = ["visitNumber", "date", "fullVisitorId", "sessionId", "visitId", "visitStartTime", "v_date", 
                      "month", "weekday"]


# In[15]:


colms = irrelevant_columns + constant_columns


# In[16]:


from sklearn.preprocessing import LabelEncoder

category_columns = [c for c in train.columns if not c.startswith("total")]
category_columns = [c for c in category_columns if c not in colms]

#print(category_columns)

for c in category_columns:

    labelencode = LabelEncoder()
    train_vals = list(train[c].values.astype(str))
    test_vals = list(test[c].values.astype(str))
    
    labelencode.fit(train_vals + test_vals)
    
    train[c] = labelencode.transform(train_vals)
    test[c] = labelencode.transform(test_vals)


# In[17]:


def normalize_numerical_columns(df, isTrain = True):
    df["totals_hits"] = df["totals_hits"].astype(float)
    df["totals_hits"] = (df["totals_hits"] - min(df["totals_hits"])) / (max(df["totals_hits"]) - min(df["totals_hits"]))

    df["totals_pageviews"] = df["totals_pageviews"].astype(float)
    df["totals_pageviews"] = (df["totals_pageviews"] - min(df["totals_pageviews"])) / (max(df["totals_pageviews"]) - min(df["totals_pageviews"]))
    
    if isTrain:
        df["totals_transactionRevenue"] = df["totals_transactionRevenue"].fillna(0.0)
    return df 


# In[18]:


train = normalize_numerical_columns(train)
test = normalize_numerical_columns(test, isTrain = False)


# In[19]:


from sklearn.model_selection import train_test_split
features = [c for c in train.columns if c not in colms]
features.remove("totals_transactionRevenue")
train["totals_transactionRevenue"] = np.log1p(train["totals_transactionRevenue"].astype(float))


# In[20]:


train.head()


# In[21]:


print(train.corrwith(train['totals_transactionRevenue']))


# In[22]:


train_x, valid_x, train_y, valid_y = train_test_split(train[features], train["totals_transactionRevenue"], 
                                     test_size=0.25, random_state=20)


# In[23]:


import lightgbm as lgb 

lgb_params = {"objective" : "regression", "metric" : "rmse",
              "num_leaves" : 50, "learning_rate" : 0.02, 
              "bagging_fraction" : 0.75, "feature_fraction" : 0.8, "bagging_frequency" : 9}
    
lgb_train = lgb.Dataset(train_x, label=train_y)
lgb_val = lgb.Dataset(valid_x, label=valid_y)
model = lgb.train(lgb_params, lgb_train, 700, valid_sets=[lgb_val], early_stopping_rounds=250, verbose_eval=100)


# In[24]:


model = lgb.train(lgb_params, lgb_train, 700, valid_sets=[lgb_val], early_stopping_rounds=250, verbose_eval=100)


# In[25]:


test_prediction = model.predict(test[features], num_iteration=model.best_iteration)
test["PredictedLogRevenue"] = np.expm1(test_prediction)
submission = test.groupby("fullVisitorId").agg({"PredictedLogRevenue" : "sum"}).reset_index()
submission["PredictedLogRevenue"] = np.log1p(submission["PredictedLogRevenue"])
submission["PredictedLogRevenue"] =  submission["PredictedLogRevenue"].apply(lambda x : 0.0 if x < 0 else x)
submission.to_csv("submission6.csv", index=False)
submission.head()


# In[26]:


train_x, valid_x, train_y, valid_y = train_test_split(train[features], train["totals_transactionRevenue"], 
                                     test_size=0.25, random_state=20)

def permutation_test(train_X, train_y, val_X, val_y):
    params = {
        "objective" : "regression",
        "metric" : "rmse", 
        "num_leaves" : 50,
        "min_child_samples" : 100,
        "learning_rate" : 0.1,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.5,
        "bagging_frequency" : 5,
        "bagging_seed" : 2018,
        "verbosity" : -1,
        "njobs" : 3
    }
    
    lgtrain = lgb.Dataset(train_x, label=train_y)
    lgval = lgb.Dataset(valid_x, label=valid_y)
    model = lgb.train(params, lgtrain, 10000, valid_sets=[lgval], early_stopping_rounds=1000, verbose_eval=500)
    
#     pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    pred_val_y = model.predict(valid_X, num_iteration=model.best_iteration)
    return model, pred_val_y


# In[27]:


train.columns


# In[38]:


from datetime import *
List = ["totals_pageviews", "totals_hits", "visitNumber", "geoNetwork_country"]
for column_perm in List:
    for i in range(20):
        print(column_perm," ",i)
        train_permute = train.copy()
        train_permute[column_perm] = np.random.permutation(train_permute[column_perm])
        dev_df = train_permute[train_permute['date']<=20170531]
        val_df = train_permute[train_permute['date']>20170531]
        dev_y = np.log1p(dev_df["totals_transactionRevenue"].values)
        val_y = np.log1p(val_df["totals_transactionRevenue"].values)

        dev_X = dev_df[cat_cols + num_cols] 
        val_X = val_df[cat_cols + num_cols] 
        model, pred_val = permutation_test(dev_X, dev_y, val_X, val_y)


# In[42]:


train.columns


# In[43]:


train.info()


# In[45]:


numerical_colums = ["totals_hits", "totals_bounces", "totals_pageviews", "totals_newVisits", "visitNumber"]


# In[46]:


category_columns


# In[47]:


from datetime import *
List = ["totals_pageviews", "totals_hits", "visitNumber", "geoNetwork_country"]
for column_perm in List:
    for i in range(20):
        print(column_perm," ",i)
        train_permute = train.copy()
        train_permute[column_perm] = np.random.permutation(train_permute[column_perm])
        dev_df = train_permute[train_permute['date']<=20170531]
        val_df = train_permute[train_permute['date']>20170531]
        dev_y = np.log1p(dev_df["totals_transactionRevenue"].values)
        val_y = np.log1p(val_df["totals_transactionRevenue"].values)

        dev_X = dev_df[category_columns + numerical_colums] 
        val_X = val_df[category_columns + numerical_colums] 
        model, pred_val = permutation_test(dev_X, dev_y, val_X, val_y)


# In[50]:


model = lgb.train(params, lgtrain, 10000, valid_sets=[lgval], early_stopping_rounds=1000, verbose_eval=500)


# In[49]:


params = {
    "objective" : "regression",
    "metric" : "rmse", 
    "num_leaves" : 50,
    "min_child_samples" : 100,
    "learning_rate" : 0.1,
    "bagging_fraction" : 0.7,
    "feature_fraction" : 0.5,
    "bagging_frequency" : 5,
    "bagging_seed" : 2018,
    "verbosity" : -1,
    "njobs" : 3
}


# In[51]:


lgb_params = {"objective" : "regression", "metric" : "rmse",
              "num_leaves" : 50, "learning_rate" : 0.02, 
              "bagging_fraction" : 0.75, "feature_fraction" : 0.8, "bagging_frequency" : 9}
    
lgb_train = lgb.Dataset(train_x, label=train_y)
lgb_val = lgb.Dataset(valid_x, label=valid_y)
model = lgb.train(lgb_params, lgb_train, 700, valid_sets=[lgb_val], early_stopping_rounds=250, verbose_eval=100)


# In[56]:


train_x, valid_x, train_y, valid_y = train_test_split(train[features], train["totals_transactionRevenue"], 
                                     test_size=0.25, random_state=20)

def permutation_test(train_X, train_y, val_X, val_y):
    params = {
        "objective" : "regression",
        "metric" : "rmse", 
        "num_leaves" : 50,
        "min_child_samples" : 100,
        "learning_rate" : 0.1,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.5,
        "bagging_frequency" : 5,
        "bagging_seed" : 2018,
        "verbosity" : -1,
        "njobs" : 3
    }
    
    lgtrain = lgb.Dataset(train_x, label=train_y)
    lgval = lgb.Dataset(valid_x, label=valid_y)
    model = lgb.train(params, lgtrain, 10000, valid_sets=[lgval], early_stopping_rounds=250, verbose_eval=100)
    
#     pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    pred_val_y = model.predict(valid_X, num_iteration=model.best_iteration)
    return model, pred_val_y


# In[57]:


from datetime import *
List = ["totals_pageviews", "totals_hits", "visitNumber", "geoNetwork_country"]
for column_perm in List:
    for i in range(20):
        print(column_perm," ",i)
        train_permute = train.copy()
        train_permute[column_perm] = np.random.permutation(train_permute[column_perm])
        dev_df = train_permute[train_permute['date']<=20170531]
        val_df = train_permute[train_permute['date']>20170531]
        dev_y = np.log1p(dev_df["totals_transactionRevenue"].values)
        val_y = np.log1p(val_df["totals_transactionRevenue"].values)

        dev_X = dev_df[category_columns + numerical_colums] 
        val_X = val_df[category_columns + numerical_colums] 
        model, pred_val = permutation_test(dev_X, dev_y, val_X, val_y)


# In[58]:


dev_df.info()


# In[60]:


train["totals_bounces"].fillna(0, inplace=True)


# In[63]:


train["totals_bounces"].fillna(0, inplace=True)
train["totals_newVisits"].fillna(0, inplace=True)
train["totals_pageviews"].fillna(0, inplace=True)
train["totals_bounces"].fillna(0, inplace=True)


# In[64]:


train.info()


# In[65]:


train_x, valid_x, train_y, valid_y = train_test_split(train[features], train["totals_transactionRevenue"], 
                                     test_size=0.25, random_state=20)

def permutation_test(train_X, train_y, val_X, val_y):
    params = {
        "objective" : "regression",
        "metric" : "rmse", 
        "num_leaves" : 50,
        "min_child_samples" : 100,
        "learning_rate" : 0.1,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.5,
        "bagging_frequency" : 5,
        "bagging_seed" : 2018,
        "verbosity" : -1,
        "njobs" : 3
    }
    
    lgtrain = lgb.Dataset(train_x, label=train_y)
    lgval = lgb.Dataset(valid_x, label=valid_y)
    model = lgb.train(params, lgtrain, 10000, valid_sets=[lgval], early_stopping_rounds=250, verbose_eval=100)
    
#     pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    pred_val_y = model.predict(valid_X, num_iteration=model.best_iteration)
    return model, pred_val_y


# In[66]:


from datetime import *
List = ["totals_pageviews", "totals_hits", "visitNumber", "geoNetwork_country"]
for column_perm in List:
    for i in range(20):
        print(column_perm," ",i)
        train_permute = train.copy()
        train_permute[column_perm] = np.random.permutation(train_permute[column_perm])
        dev_df = train_permute[train_permute['date']<=20170531]
        val_df = train_permute[train_permute['date']>20170531]
        dev_y = np.log1p(dev_df["totals_transactionRevenue"].values)
        val_y = np.log1p(val_df["totals_transactionRevenue"].values)

        dev_X = dev_df[category_columns + numerical_colums] 
        val_X = val_df[category_columns + numerical_colums] 
        model, pred_val = permutation_test(dev_X, dev_y, val_X, val_y)


# In[69]:


lgb_params = {"objective" : "regression", "metric" : "rmse",
              "num_leaves" : 50, "learning_rate" : 0.02, 
              "bagging_fraction" : 0.75, "feature_fraction" : 0.8, "bagging_frequency" : 9}
    
lgb_train = lgb.Dataset(train_x, label=train_y)
lgb_val = lgb.Dataset(valid_x, label=valid_y)
model = lgb.train(lgb_params, lgb_train, 700, valid_sets=[lgb_val], early_stopping_rounds=1000, verbose_eval=250)


# In[68]:


pred_val_y = model.predict(valid_x, num_iteration=model.best_iteration)
pred_val_y


# In[77]:


def permutation_test(train_X, train_y, val_X, val_y):
    lgb_params = {"objective" : "regression", "metric" : "rmse",
              "num_leaves" : 50, "learning_rate" : 0.02, 
              "bagging_fraction" : 0.75, "feature_fraction" : 0.8, "bagging_frequency" : 9, "bagging_seed" : 2018,
               "verbosity" : -1,"njobs" : 3}
    
    lgb_train = lgb.Dataset(train_X, label=train_y)
    lgb_val = lgb.Dataset(val_X, label=val_y)
    model = lgb.train(lgb_params, lgb_train, 700, valid_sets=[lgb_val], early_stopping_rounds=1000, verbose_eval=250)
    pred_val_y = model.predict(val_X, num_iteration=model.best_iteration)

    return model, pred_val_y


# In[82]:


from datetime import *
List = ["totals_pageviews", "totals_hits", "visitNumber", "geoNetwork_country"]
for column_perm in List:
    for i in range(20):
        print(column_perm," ",i)
        train_permute = train.copy()
        train_permute[column_perm] = np.random.permutation(train_permute[column_perm])
        dev_df = train_permute[train_permute['date']<=20170531]
        val_df = train_permute[train_permute['date']>20170531]
        dev_y = np.log1p(dev_df["totals_transactionRevenue"].values)
        val_y = np.log1p(val_df["totals_transactionRevenue"].values)

        dev_X = dev_df[category_columns + numerical_colums] 
        val_X = val_df[category_columns + numerical_colums] 
        model, pred_val = permutation_test(dev_X, dev_y, val_X, val_y)


# In[83]:


dev_df.info()


# In[80]:


train["trafficSource_isTrueDirect"].fillna(0, inplace=True)


# In[81]:


train.info()


# In[79]:


dev_df.info()


# In[84]:


val_df.info()


# In[85]:


dev_df = train_permute[train['date']<=20170531]
val_df = train_permute[train['date']>20170531]
dev_y = np.log1p(dev_df["totals_transactionRevenue"].values)
val_y = np.log1p(val_df["totals_transactionRevenue"].values)
dev_X = dev_df[category_columns + numerical_colums] 
val_X = val_df[category_columns + numerical_colums]


# In[86]:


def permutation_test(train_X, train_y, val_X, val_y):
    lgb_params = {"objective" : "regression", "metric" : "rmse",
              "num_leaves" : 50, "learning_rate" : 0.02, 
              "bagging_fraction" : 0.75, "feature_fraction" : 0.8, "bagging_frequency" : 9, "bagging_seed" : 2018,
               "verbosity" : -1,"njobs" : 3}
    
    lgb_train = lgb.Dataset(train_X, label=train_y)
    lgb_val = lgb.Dataset(val_X, label=val_y)
    model = lgb.train(lgb_params, lgb_train, 700, valid_sets=[lgb_val], early_stopping_rounds=1000, verbose_eval=250)
    pred_val_y = model.predict(val_X, num_iteration=model.best_iteration)

    return model, pred_val_y


# In[87]:


from datetime import *
List = ["totals_pageviews", "totals_hits", "visitNumber", "geoNetwork_country"]
for column_perm in List:
    for i in range(20):
        print(column_perm," ",i)
        train_permute = train.copy()
        train_permute[column_perm] = np.random.permutation(train_permute[column_perm])
        dev_df = train_permute[train_permute['date']<=20170531]
        val_df = train_permute[train_permute['date']>20170531]
        dev_y = np.log1p(dev_df["totals_transactionRevenue"].values)
        val_y = np.log1p(val_df["totals_transactionRevenue"].values)

        dev_X = dev_df[category_columns + numerical_colums] 
        val_X = val_df[category_columns + numerical_colums] 
        model, pred_val = permutation_test(dev_X, dev_y, val_X, val_y)


# In[88]:


cnt_srs = train.groupby('fullVisitorId')['totals_transactionRevenue'].agg(['size', 'count', 'mean'])


# In[92]:


train['Purchase'] =  train['totals_transactionRevenue'].apply(lambda x : [0,1][x>0])


# In[94]:


cnt_srs = train.groupby('fullVisitorId')['Purchase'].agg(['size', 'sum'])


# In[96]:


cnt_mean = train.groupby('fullVisitorId')['totals_transactionRevenue'].agg(['size', 'mean'])


# In[97]:


cnt_srs['mean'] = cnt_mean['mean']


# In[99]:


cnt_srs['prob'] = cnt_srs['sum']/cnt_srs['size']


# In[102]:


sorted(cnt_srs)


# In[108]:


cnt_srs.sort_values(by=['prob'], ascending=False)


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
spearman_correlation = train.corr(method='spearman')
pick_columns=spearman_correlation.nlargest(20, 'totals_transactionRevenue').index
correlationmap = np.corrcoef(train[pick_columns].values.T)
sns.set(font_scale=1.0)
heatmap = sns.heatmap(correlationmap, cbar=True, annot=True, square=True , fmt='.2f', 
                      yticklabels=train.values, xticklabels=train.values)
plt.show()

