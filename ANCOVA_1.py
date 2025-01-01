import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# Data from the image
data = {
    "Student": ["Student " + str(i) for i in range(1, 16)],
    "Study_Technique": ["A"] * 5 + ["B"] * 5 + ["C"] * 5,
    "Current_Grade": [67, 88, 75, 77, 85, 92, 69, 77, 74, 88, 96, 91, 88, 82, 80],
    "Exam_Score": [77, 89, 72, 74, 69, 78, 88, 93, 94, 90, 85, 81, 83, 88, 79]
}

df = pd.DataFrame(data)

# Scatter plot of Exam Score vs. Current Grade, colored by Study Technique
plt.figure(figsize=(8, 6))
for technique, group in df.groupby("Study_Technique"):
    plt.scatter(group["Current_Grade"], group["Exam_Score"], label=technique)

plt.xlabel("Current Grade")
plt.ylabel("Exam Score")
plt.title("Exam Score vs. Current Grade by Study Technique")
plt.legend()
plt.show()

# Fit the ANCOVA model
model = smf.ols("Exam_Score ~ Current_Grade + Study_Technique", data=df).fit()

# Print the summary
print(model.summary())

# Residual plot
plt.figure(figsize=(8, 6))
plt.scatter(df["Current_Grade"], model.resid)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Current Grade")
plt.ylabel("Residuals")
plt.title("Residuals vs. Current Grade")
plt.show()

# Model with interaction term
interaction_model = smf.ols("Exam_Score ~ Current_Grade * Study_Technique", data=df).fit()

# Check interaction effect
print(interaction_model.summary())

# Predict adjusted means
df["Predicted"] = model.predict(df)

# Plot observed vs. predicted
plt.figure(figsize=(8, 6))
for technique, group in df.groupby("Study_Technique"):
    plt.scatter(group["Current_Grade"], group["Exam_Score"], label=f"Observed {technique}")
    plt.plot(group["Current_Grade"], group["Predicted"], label=f"Predicted {technique}")

plt.xlabel("Current Grade")
plt.ylabel("Exam Score")
plt.title("Observed vs. Predicted Exam Score by Study Technique")
plt.legend()
plt.show()