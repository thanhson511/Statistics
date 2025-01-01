'''
Table 1 presents data for 10 men, where 1st column represents years of experience and 2nd column denotes income. 
Table 2 provides analogous data for 10 women, with 1st column indicating years of experience and 2nd column representing income.


Year        Income  ( of Male )

5	          50,000
6		        55,000
7	        	60,000
8		        65,000
9		        70,000
10	      	75,000
11	      	80,000
12	      	85,000
13	      	90,000
14		      95,000


Year        Income  ( of Female )

5		        45,000
6		        50,000
7	        	55,000
8		        60,000
9		        65,000
10	      	70,000
11	      	75,000
12	      	80,000
13	      	85,000
14	      	90,000

(You can change this data to get more examples)
Test whether gender significantly affects income after controlling for years of experience.


'''
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# Data for men
men_data = {
    "Experience": [5, 6, 5, 8, 9, 10, 10, 12, 13, 14],
    "Income": [51000, 55000, 55000, 62000, 72000, 73000, 77000, 83000, 89000, 96000],
    "Gender": ["Male"] * 10
}

# Data for women
women_data = {
    "Experience": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    "Income": [45000, 52000, 54000, 61000, 64000, 70000, 74000, 77000, 82000, 92000],
    "Gender": ["Female"] * 10
}

# Combine into a single dataset

# Instead of:
# data = pd.DataFrame(men_data).append(pd.DataFrame(women_data), ignore_index=True)

# Use pd.concat:
data = pd.concat([pd.DataFrame(men_data), pd.DataFrame(women_data)], ignore_index=True)
print(data)

# Scatter plot of income vs. experience, colored by gender
plt.figure(figsize=(8, 6))
for gender, group in data.groupby("Gender"):
    plt.scatter(group["Experience"], group["Income"], label=gender)

plt.xlabel("Years of Experience")
plt.ylabel("Income")
plt.title("Income vs. Experience by Gender")
plt.legend()
plt.show()


# Fit the ANCOVA model
model = smf.ols("Income ~ Experience + Gender", data=data).fit()

# Print the summary
print(model.summary())


# Residual plot
plt.figure(figsize=(8, 6))
plt.scatter(data["Experience"], model.resid)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Years of Experience")
plt.ylabel("Residuals")
plt.title("Residuals vs. Experience")
plt.show()


# Model with interaction term
interaction_model = smf.ols("Income ~ Experience * Gender", data=data).fit()

# Check interaction effect
print(interaction_model.summary())



# Predict adjusted means
data["Predicted"] = model.predict(data)

# Plot observed vs. predicted
plt.figure(figsize=(8, 6))
for gender, group in data.groupby("Gender"):
    plt.scatter(group["Experience"], group["Income"], label=f"Observed {gender}")
    plt.plot(group["Experience"], group["Predicted"], label=f"Predicted {gender}")

plt.xlabel("Years of Experience")
plt.ylabel("Income")
plt.title("Observed vs. Predicted Income by Gender")
plt.legend()
plt.show()
