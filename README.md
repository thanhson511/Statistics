# ANCOVA Analysis of Study Techniques and Exam Scores

## Overview
This project analyzes how different study techniques influence exam scores while controlling for students' current grades. The dataset comprises 15 students categorized by three study techniques (A, B, and C), their current grades, and their exam scores. Using Python, we conduct an ANCOVA analysis to determine the statistical significance of the study techniques on exam performance.

---

## Dataset

| Student    | Study Technique | Current Grade | Exam Score |
|------------|-----------------|---------------|------------|
| Student 1  | A               | 67            | 77         |
| Student 2  | A               | 88            | 89         |
| Student 3  | A               | 75            | 72         |
| Student 4  | A               | 77            | 74         |
| Student 5  | A               | 85            | 69         |
| Student 6  | B               | 92            | 78         |
| Student 7  | B               | 69            | 88         |
| Student 8  | B               | 77            | 93         |
| Student 9  | B               | 74            | 94         |
| Student 10 | B               | 88            | 90         |
| Student 11 | C               | 96            | 85         |
| Student 12 | C               | 91            | 81         |
| Student 13 | C               | 88            | 83         |
| Student 14 | C               | 82            | 88         |
| Student 15 | C               | 80            | 79         |

---

## Steps

### 1. Import Necessary Libraries

```python
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
```

---

### 2. Create the Dataset

```python
# Creating the dataset
data = pd.DataFrame({
    "Student": [f"Student {i}" for i in range(1, 16)],
    "Study_Technique": ["A"] * 5 + ["B"] * 5 + ["C"] * 5,
    "Current_Grade": [67, 88, 75, 77, 85, 92, 69, 77, 74, 88, 96, 91, 88, 82, 80],
    "Exam_Score": [77, 89, 72, 74, 69, 78, 88, 93, 94, 90, 85, 81, 83, 88, 79],
})

print(data)
```

---

### 3. Exploratory Data Analysis

```python
# Boxplot of Exam Scores by Study Technique
plt.figure(figsize=(8, 6))
data.boxplot(column="Exam_Score", by="Study_Technique", grid=False, color="blue")
plt.title("Exam Scores by Study Technique")
plt.suptitle("")  # Remove the automatic title
plt.xlabel("Study Technique")
plt.ylabel("Exam Score")
plt.show()
```

---

### 4. Fit the ANCOVA Model

```python
# Fit the ANCOVA model
model = smf.ols("Exam_Score ~ Current_Grade + Study_Technique", data=data).fit()

# Print the summary
print(model.summary())
```

---

### 5. Interpret Results

Key outputs from the ANCOVA model:
- **Coefficients for Study_Technique**: Reflect how each technique affects exam scores compared to the reference category.
- **P-value for Study_Technique**: Indicates whether the effect of study techniques on exam scores is significant.
- **R-squared**: Shows the proportion of variance in exam scores explained by the model.

---

### 6. Visualize Adjusted Exam Scores

```python
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
```

---

### 7. Check Assumptions

#### (a) Linearity

```python
# Residual plot
plt.figure(figsize=(8, 6))
plt.scatter(data["Current_Grade"], model.resid)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Current Grade")
plt.ylabel("Residuals")
plt.title("Residuals vs. Current Grade")
plt.show()
```

#### (b) Homogeneity of Regression Slopes

```python
# Model with interaction term
interaction_model = smf.ols("Exam_Score ~ Current_Grade * Study_Technique", data=data).fit()

# Print the summary
print(interaction_model.summary())
```

If the interaction term (`Current_Grade:Study_Technique`) is not significant, the effect of study techniques on exam scores is consistent across current grades.

---

## Requirements
- Python 3.x
- Libraries: `pandas`, `statsmodels`, `matplotlib`

---

## Usage
1. Clone this repository.
2. Install the required libraries using `pip install pandas statsmodels matplotlib`.
3. Run the Python script to perform ANCOVA analysis and visualize results.

---

## Notes
- Ensure the dataset matches the provided format.
- Interpret results carefully, considering the assumptions of ANCOVA.

---

## License
This project is open-source and available under the [MIT License](LICENSE).

---
