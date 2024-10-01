# import pickle
import altair as alt
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


penguin_df = pd.read_csv("penguins.csv")
penguin_df.dropna(inplace=True)
output = penguin_df["species"]
features = penguin_df[
    [
        "island",
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
        "sex",
    ]
]
features = pd.get_dummies(features)
# print(output.head())
# print(features.head())

output, uniques = pd.factorize(output)
# print(output)
# print(uniques)

x_train, x_test, y_train, y_test = train_test_split(features, output, test_size=0.8)
rfc = RandomForestClassifier(random_state=15)
rfc.fit(x_train.values, y_train)
y_pred = rfc.predict(x_test.values)
score = accuracy_score(y_pred, y_test)
print("Our accuracy score for this model is {}".format(score))

st.title("计算结果对比")
st.write("预测模型和实际情况的对比")

# 假设 y_test 和 y_pred 是两个等长的序列
data = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
}).reset_index()

# 创建图表
# fig = alt.Chart(data).mark_bar().encode(
#     x=alt.X('index', title='样本索引'),
#     y=alt.Y('value:Q', stack=None),
#     color=alt.Color('variable:N')
# ).transform_fold(
#     ['Actual', 'Predicted']
# )

# fig = alt.Chart(data).mark_circle().encode(
#     x=alt.X('index:O', title='样本索引'),
#     y=alt.Y('value:Q', stack=None),
#     color=alt.Color('variable:N')
# ).transform_fold(
#     ['Actual', 'Predicted']
# )


st.write(data.head())

# 创建图表
fig = alt.Chart(data).mark_circle().encode(
    x=alt.X('index:O', title='样本索引'),
    y=alt.Y('value:Q', stack=None),
    color=alt.Color('variable:N')
).transform_fold(
    ['Actual', 'Predicted']
)

st.altair_chart(fig)
