
























# =========================================
# 1) IMPORT LIBRARIES
# =========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

sns.set(style="whitegrid")


# =========================================
# 2) LOAD DATA
# =========================================
df = pd.read_csv(r"C:/Users/swami/Downloads/mall_customers_extended_1000.csv")

print("Shape:", df.shape)
print(df.head())


# =========================================
# 3) BASIC EDA
# =========================================
print(df.info())
print(df.isnull().sum())
print(df.describe())


# =========================================
# 4) DATA CLEANING
# =========================================
df.columns = df.columns.str.strip().str.lower()


# =========================================
# 5) KPI SUMMARY
# =========================================
print("\n===== KPI SUMMARY =====")
print("Total Customers:", df["customerid"].nunique())
print("Average Income:", df["annual income (k$)"].mean())
print("Average Spending Score:", df["spending score (1-100)"].mean())
print("Top Category:", df["preferred category"].mode()[0])


# =========================================
# 6) LINE CHART
# =========================================
plt.figure()
sns.lineplot(x="age", y="spending score (1-100)", data=df)
plt.title("Age vs Spending Score")
plt.show()


# =========================================
# 7) BAR CHART
# =========================================
top_cat = df["preferred category"].value_counts()

plt.figure()
sns.barplot(x=top_cat.values, y=top_cat.index)
plt.title("Preferred Categories")
plt.show()


# =========================================
# 8) PIE CHART
# =========================================
plt.figure()
plt.pie(top_cat, labels=top_cat.index, autopct="%1.1f%%")
plt.title("Category Distribution")
plt.show()


# =========================================
# 9) SCATTER PLOT
# =========================================
plt.figure()
sns.scatterplot(x="annual income (k$)", y="spending score (1-100)", data=df)
plt.title("Income vs Spending")
plt.show()


# =========================================
# 10) BOX PLOT
# =========================================
plt.figure()
sns.boxplot(x=df["spending score (1-100)"])
plt.show()


# =========================================
# 11) DISTRIBUTION
# =========================================
plt.figure()
sns.kdeplot(df["spending score (1-100)"], fill=True)
plt.show()


# =========================================
# 12) CORRELATION
# =========================================
plt.figure()
corr = df[["age","annual income (k$)","spending score (1-100)"]].corr()
sns.heatmap(corr, annot=True)
plt.show()


# =========================================
# 13) STATISTICS
# =========================================
corr_val, p_val = stats.pearsonr(df["annual income (k$)"], df["spending score (1-100)"])

print("Correlation:", corr_val)
print("P-value:", p_val)


# =========================================
# 14) MACHINE LEARNING
# =========================================
X = df[["age","annual income (k$)"]]
y = df["spending score (1-100)"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("R2 Score:", r2_score(y_test, y_pred))


# =========================================
# 🔥 NEW COLORFUL VISUALS (ADDED)
# =========================================

# 🌈 15) COLORFUL LINE CHART
plt.figure()
sns.lineplot(
    x="age",
    y="spending score (1-100)",
    data=df,
    marker="o",
    color="red"
)
plt.title("Age vs Spending Score (Enhanced)")
plt.grid(True)
plt.show()


# 🎨 16) TREEMAP STYLE BAR
plt.figure()
top_city = df["city"].value_counts().head(10)
colors = sns.color_palette("viridis", len(top_city))

plt.barh(top_city.index, top_city.values, color=colors)
plt.title("City-wise Customers")
plt.gca().invert_yaxis()
plt.show()


# 📊 17) COMBINED BAR + LINE
plt.figure()
top_df = df.sort_values("annual income (k$)", ascending=False).head(10)

plt.bar(top_df["city"], top_df["annual income (k$)"], color="tomato", label="Income")
plt.plot(top_df["city"], top_df["spending score (1-100)"], 
         color="black", marker="o", label="Spending Score")

plt.xticks(rotation=45)
plt.title("Income vs Spending Score")
plt.legend()
plt.show()


# 🍩 18) DONUT CHART
plt.figure()
category_counts = df["preferred category"].value_counts()
colors = sns.color_palette("Set2", len(category_counts))

plt.pie(category_counts, labels=category_counts.index,
        colors=colors, autopct="%1.1f%%", startangle=90)

centre_circle = plt.Circle((0,0), 0.60, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.title("Preferred Category (Donut)")
plt.show()
