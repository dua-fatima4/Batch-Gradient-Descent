🧠 Batch Gradient Descent vs Scikit-Learn Linear Regression

This project compares:

Linear Regression using Scikit-Learn

Custom implementation of Batch Gradient Descent (from scratch)

Performance comparison using R² Score

Training time comparison

Dataset used: Diabetes Dataset

📂 Dataset

We used the built-in dataset from:

scikit-learn

Function:

from sklearn.datasets import load_diabetes
Dataset Details

Total Samples: 442

Input Features: 10

Target: Disease progression measure (continuous value)

🚀 Part 1 — Linear Regression (Using Scikit-Learn)

Model used:

from sklearn.linear_model import LinearRegression
Steps:

Load dataset

Train-Test Split (80-20)

Fit model

Predict on test set

Evaluate using R² score

Results

Intercept: 151.8833

R² Score: 0.4399

Scikit-learn uses an optimized closed-form solution (Normal Equation), which gives stable and accurate results.

🔥 Part 2 — Custom Batch Gradient Descent (From Scratch)

I implemented my own class:

class GDRegressor:

This implementation performs Batch Gradient Descent, meaning:

It uses the entire training dataset

Computes gradient using all samples

Updates weights after each full pass (epoch)

⚙️ Hyperparameters Used

Learning Rate = 0.5

Epochs = 300

🧮 Update Rule Used

For each epoch:

θ = θ − α × (∂L / ∂θ)
	​
Where:

α = learning rate

L = Mean Squared Error (MSE)

θ = parameters (weights + intercept)

📊 Results (Custom GD)

Intercept: 152.08

R² Score: 0.4253

Training Time: 0.0304 seconds

📈 Performance Comparison
Model	R² Score
Scikit-Learn Linear Regression	0.4399
Custom Batch Gradient Descent	0.4253

✅ The custom implementation performs very close to Scikit-Learn.

This shows:

Correct gradient implementation

Proper convergence

Good learning rate selection


⚡ Why Batch Gradient Descent Works Here

Because:

Dataset is small (442 samples)

Full dataset computation is fast

Stable convergence with tuned learning rate

For larger datasets, Mini-Batch or Stochastic Gradient Descent would be preferred.


🧰 Technologies Used

Python

NumPy

Scikit-Learn

Jupyter Notebook

🎯 Conclusion

Building Batch Gradient Descent from scratch strengthened my understanding of:

Optimization

Derivatives in ML

Convergence behavior

Model tuning

This project demonstrates foundational ML understanding beyond just using libraries.
