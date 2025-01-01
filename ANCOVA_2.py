import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt


# Creating the dataset
data = pd.DataFrame({
    "Student": [f"Student {i}" for i in range(1, 16)],
    "Study_Technique": ["A"] * 5 + ["B"] * 5 + ["C"] * 5,
    "Current_Grade": [67, 88, 75, 77, 85, 92, 69, 77, 74, 88, 96, 91, 88, 82, 80],
    "Exam_Score": [77, 89, 72, 74, 69, 78, 88, 93, 94, 90, 85, 81, 83, 88, 79],
})

print(data)


# Boxplot of Exam Scores by Study Technique
plt.figure(figsize=(8, 6))
data.boxplot(column="Exam_Score", by="Study_Technique", grid=False, color="blue")
plt.title("Exam Scores by Study Technique")
plt.suptitle("")  # Remove the automatic title
plt.xlabel("Study Technique")
plt.ylabel("Exam Score")
plt.show()



# Fit the ANCOVA model
model = smf.ols("Exam_Score ~ Current_Grade + Study_Technique", data=data).fit()

# Print the summary
print(model.summary())


# Predict adjusted exam scores
data["Predicted_Score"] = model.predict(data)

# Plot observed vs. predicted scores
plt.figure(figsize=(8, 6))
for technique, group in data.groupby("Study_Technique"):
    plt.scatter(group["Current_Grade"], group["Exam_Score"], label=f"Observed {technique}")
    plt.plot(group["Current_Grade"], group["Predicted_Score"], label=f"Predicted {technique}")

plt.xlabel("Current Grade")
plt.ylabel("Exam Score")
plt.title("Observed vs. Predicted Exam Scores by Study Technique")
plt.legend()
plt.show()



# Residual plot
plt.figure(figsize=(8, 6))
plt.scatter(data["Current_Grade"], model.resid)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Current Grade")
plt.ylabel("Residuals")
plt.title("Residuals vs. Current Grade")
plt.show()


# Model with interaction term
interaction_model = smf.ols("Exam_Score ~ Current_Grade * Study_Technique", data=data).fit()

# Print the summary
print(interaction_model.summary())
