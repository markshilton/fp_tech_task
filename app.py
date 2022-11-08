import streamlit as st
import pandas as pd
import plotly.express as px
import xgboost as xgb
import shap

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix
from imblearn.over_sampling import SMOTE
from collections import Counter

st.set_option('deprecation.showPyplotGlobalUse', False)

def load_data(infile):
    df = pd.read_csv(infile)
    df['Failed'] = df['Failure Type'].apply(lambda x: False if x == 'No Failure' else True)
    df['Machine Type cleaned'] = df['Machine Type'].str.replace('_', '').str.lower()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Hour'] = df['Timestamp'].dt.hour
    df['Day'] = df['Timestamp'].dt.day
    df['Month'] = df['Timestamp'].dt.month
    df['Year'] = df['Timestamp'].dt.year
    df['Year_Month'] = df['Timestamp'].dt.to_period('M').dt.to_timestamp()
    df['Timestamp cleaned'] = df['Timestamp'].apply(lambda x: x if x.year < 2004 else x - pd.DateOffset(years=10))
    df['Air - Process diff [K]'] = df['Process temperature [K]'] - df['Air temperature [K]'] 
    df = df.dropna()
    return df

def train_model(df):
    #Scale continuous variables and one hot encode categorical variables
    scaler = MinMaxScaler()
    cols_to_scale = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Air - Process diff [K]']
    to_scale = df[cols_to_scale]
    print(len(to_scale))
    scaled = pd.DataFrame(scaler.fit_transform(to_scale)).reset_index(drop=True)
    scaled.columns = [x.lower().replace(' ', '_').replace('[','').replace(']','') for x in cols_to_scale]

    dummies = pd.get_dummies(df['Machine Type']).reset_index(drop=True)
    print(len(dummies))
    X = pd.concat([scaled, dummies], axis=1)
    y = df['Failed'].astype('int')

    print(len(X), len(y))

    #Make train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Balance the classes in the training data using SMOTE oversampling
    smote = SMOTE()
    x_smote, y_smote = smote.fit_resample(X_train, y_train)

    # fit model to training data
    model = xgb.XGBClassifier()
    model.fit(x_smote, y_smote)

    return model, X, y, x_smote, X_test, y_smote, y_test

infile = ('data/sample_interview_dataset.csv')
df = load_data(infile)

"""
# What can you tell me about my machine performance?

For this task I was provided with a csv of machine performance. Here I'll detail what I found out about the machine's performance - trends, useful insights and some recommndations about areas for further investgation.

### Data cleanliness

The data holds 10,000 rows of machine run data. This data has a curious distribution over time with the bulk of the data being logged in 2002, with some in 2001 and 2003. Then we have an almost ten year(!) gap in the data to 2011 where we see around 100 logs over 2011/2012/2013.
"""
fig_log_counts_by_year = px.histogram(df, x='Year', text_auto=True, title='Machine log counts by year')
fig_log_counts_by_year.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
st.plotly_chart(fig_log_counts_by_year)

"""
Relatedly, the machine types are split between different years. The x1/x_1/X1 types are all in the earlier period whilst the x2 machines are in the 2010s period:

#### Record count split by machine type and log year
"""
st.table(pd.pivot_table(df, index='Machine Type', columns='Year', values='UID', aggfunc='count', fill_value=0))

"""
The different variations of X and 1 jumped out at me. Are these typos and actually all the same machine type? I also wonder if there's a systematic typo in the dates for the X2 machines. Are we actually looking at data from 2001-2003 for two machine types - X1 and X2? In a real life situation I'd go back to the client to clarify this asap.

### Torque and rotational speed patterns

Plotting the distribution of torque and rotational speed split by machine type we can see that there's a fairly normal, slightly right skewed distribution of rotational speeds and a bimodal distribution for torque values that is very similar across all machine types. It would seem that these machines are running quite similar workloads?
"""
fig_rotation_hist = px.histogram(df, x='Rotational speed [rpm]', histnorm='percent', facet_row='Machine Type', title='Distribution of rotational speed across machine runs split by machine type')
fig_rotation_hist.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
st.plotly_chart(fig_rotation_hist)

fig_torque_hist = px.histogram(df, x='Torque [Nm]',  histnorm='percent', facet_row='Machine Type', title='Distribution of torque values across machine runs')
fig_torque_hist.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
st.plotly_chart(fig_torque_hist)

"""
Plotting rotational speed and torque against each other we can see there's an inverse correlation between the two variables that is common across both of the clusters of torque values. The range of rotational speeds is similar across both clusters. Are these different types of jobs that the machines are doing that require different levels of torque?
"""
fig_rotation_torque_scatter = px.scatter(df, x='Torque [Nm]', y='Rotational speed [rpm]', opacity=0.4, facet_row='Machine Type', title='Torque value plotted against rotational speed for each machine type')
fig_rotation_torque_scatter['layout']['yaxis']['title']['text']='rpm'
fig_rotation_torque_scatter['layout']['yaxis2']['title']['text']='rpm'
fig_rotation_torque_scatter['layout']['yaxis3']['title']['text']='rpm'
fig_rotation_torque_scatter['layout']['yaxis4']['title']['text']='rpm'
fig_rotation_torque_scatter.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
st.plotly_chart(fig_rotation_torque_scatter)

"""
### Tool wear

The other key operational feature of this data is tool wear. I'm assuming this to be a measure of the number of minutes the tool had been used for at the beginning of the job. There is a fairly uniform distribution across all machine types up to around 200 minutes where each distribution tails off:
"""
fig_tool_wear_hist = px.histogram(df, x='Tool wear [min]',  histnorm='percent', facet_row='Machine Type', title='Distribution of tool wear (mins) across machine runs')
st.plotly_chart(fig_tool_wear_hist)

"""
### Operating temperatures

For this analysis I've assumed for now that the timestamps for X2 machines were badly encoded and cleaned them up by substracting 10 years from them. Plotting a scatter of these values over time shows that there are some external environmental considerations to think about. There is a general trend to higher air temperatures in the summer months (assuming this is Northern hemisphere data!). There's also an interesting split in the observations with two distinct profiles around 20K apart. I'd be interested to understand more about this. My first thought woud be that perhaps these are different locations? Or different facilities with different cooling/heating systems? I'd be interested to understand more about why this is.
"""
fig_air_temps = px.scatter(df, x='Timestamp cleaned', y='Air temperature [K]', opacity=0.5, title='Air temperature by timestamp')
fig_air_temps.update_traces(marker=dict(size=3))
st.plotly_chart(fig_air_temps)

"""
There's a similar pattern in the process temperatures and a clear positive correlation between air and process temperatures:
"""
fig_process_temps = px.scatter(df, x='Timestamp cleaned', y='Process temperature [K]', opacity=0.5, title='Process temperature by timestamp')
fig_process_temps.update_traces(marker=dict(size=3))
st.plotly_chart(fig_process_temps)

fig_process_air_scatter = px.scatter(df, x='Process temperature [K]', y='Air temperature [K]', width=500, height=500, opacity=0.5, title='Air and process temperatures by machine run')
fig_process_air_scatter.update_traces(marker=dict(size=3))
st.plotly_chart(fig_process_air_scatter)

"""
Another interesting trend if we look into the differnce between the air and process temperatures. There appear to be three peaks in the distribution. Two big ones around 9 and 11K and a smaller one right up at 25-27K. I'd be interested to understand the reasons for this. It looks related to teh cluster of very low air tempratures that you can see in the charts above that aren't reflected in the process temperature plot over time. 
"""
fig_air_process_diffs = px.histogram(df, x='Air - Process diff [K]', title='Distribution of air and process temperature differences')
st.plotly_chart(fig_air_process_diffs)


"""
### Tool failures

I would assume that the most pressing issue in this dataset is the issue of tool failures. The final part of this analysis digs into the failure reasons and looks at how we might use some predictive models to better understand the reasons for tool failure and improve this failure rate over time.

I looked at failure rate by machine type and found that there seems to be a significantly higher failure rate for the 'x_1' machines compared to 'X1' and 'x1' machines. I'd again be interested to understand if these are typos or genuinely different machine types.
"""
failures = pd.pivot_table(df, index='Machine Type', columns='Failed', values='UID', aggfunc='count')
failures.columns = ['Successful runs', 'Failures']
failures['Failure rate (%)'] = failures['Failures'] / failures.sum(axis=1) * 100
st.table(failures)

"""
### Building a model to predict tool failures

This is a classification problem and so I built a very simple logisitc regression model as my first benchmark. This was pretty terrible. After oversampling the minority class the accuracy score was only around 0.66 with over 1/3 of non-failures predicted as failures. I next used boosted trees and had much more success with an area under the curve of 0.85. Using this model the confusion matrix looks a lot more healthy, although precision and recall are both still only just over 50%. Given the lack of tuning and the very small failure rate this is a decent first start. Some more thinking around feature engineering and model tuning would hopefuly improve this performance.

#### Confusion matrix for the Boosted Tree classifcation model
"""
model, X, y, x_smote, X_test, y_smote, y_test = train_model(df)
y_pred = model.predict(X_test)
pred = model.predict(X_test)
score = model.score(X_test, y_test)

st.table(confusion_matrix(y_test, pred))

y_score = model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_score)

fig_roc = px.area(
    x=fpr, y=tpr,
    title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
    labels=dict(x='False Positive Rate', y='True Positive Rate'),
    width=700, height=500
)
fig_roc.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)

fig_roc.update_yaxes(scaleanchor="x", scaleratio=1)
fig_roc.update_xaxes(constrain='domain')

st.plotly_chart(fig_roc)

"""
Even with this level of performance, we can use Shapley Value plots to understand the relative weights of the different features used in the model. Each feature is ranked by imortance with the most important at the top. Each data point is plotted with it's relative value color coded (red is high, blue is low) and it's impact on that individual prediction plotted left to right (far right means likely to fail, far left, less likely).

We can see that tool wear is the most important predictor with high values of tool wear being very likely to predict failure. Other features are a little less clear cut with a more mixed spread across the plot.
"""


X100 = shap.utils.sample(X, 100) # 100 instances for use as the background distribution
explainer_xgb = shap.Explainer(model, X100)
shap_values_xgb = explainer_xgb(X, check_additivity=False)

shap.plots.beeswarm(shap_values_xgb)
st.pyplot(bbox_inches='tight')

"""
### Questions for further investigation

- What causes the distinct groups of air/process temperatures? Are these different factories in different locations?
- Clarify the data cleaning issues. 
- Understand what the impact of failures is. Does it make sense to overpredict failures to avoid costly delays or is it less of an issue to miss some of the failures to reduce the number of times the jobs are stopped for maintenance?
- More investigation into features that could improve our model performance and explain failure rates.
"""

