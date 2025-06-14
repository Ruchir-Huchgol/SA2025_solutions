# SA2025 Assignment 1 Solution 🎓

This repository contains the completed solutions for **Assignment 1** of the **Summer Analytics 2025** program, submitted by Ruchir Huchgol.

---

## 📚 Repository Contents

- `Cars.csv`  
  - Dataset used for all analyses (includes features like mpg, cylinders, horsepower, weight, origin, etc.).
  
- `cars_analysistask1.ipynb`  
  - Notebook performing Task 1: exploratory data analysis and statistical summaries on the Cars dataset.

- `horsepower.ipynb`  
  - Notebook focusing on analysis of the horsepower feature—insights, distributions, and relevant visualizations.

- `mpg_cars.ipynb`  
  - Notebook analyzing fuel efficiency (mpg) across car attributes such as origin and cylinder count.

- `Summer Analytics 2025-2.pdf`  
  - Assignment worksheet provided by the Summer Analytics program, containing problem statements and guidelines.

---

## 🧠 Analysis Overview

The project covers the following steps:

1. **Data Loading & Cleaning**  
   - Reading and inspecting `Cars.csv` for missing values, data types, and initial cleaning.

2. **Exploratory Data Analysis (EDA)**  
   - Computing summary statistics (mean, median, min/max) for numerical features.
   - Visualizing distributions for key variables: mpg, horsepower, weight, etc.

3. **Feature-Specific Investigations**  
   - **Horsepower Analysis (`horsepower.ipynb`)**  
     - Distribution plots and key statistics for horsepower.
   - **MPG Analysis (`mpg_cars.ipynb`)**  
     - Comparing mpg across different origins (e.g., USA, Europe, Asia) and cylinder groups.

4. **Statistical Grouping & Insights**  
   - Group-by operations: average mpg by origin, cylinders, model year.
   - Identifying trends and patterns in fuel efficiency.

---

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ruchir-Huchgol/SA2025_assignment1solution.git
   cd SA2025_assignment1solution

2.Setup environment
Ensure you have Python (≥3.7) and Jupyter installed.
Recommended libraries:
pip install pandas numpy matplotlib seaborn

3.Launch Jupyter Notebook
jupyter notebook

Open and run:
cars_analysistask1.ipynb
horsepower.ipynb
mpg_cars.ipynb
Ensure Cars.csv is in the same working directory.

🛠️ Tools & Libraries
Python 3.x

Jupyter Notebook

pandas – data loading, cleaning, grouping.

NumPy – numerical operations.

Matplotlib, Seaborn – plotting and visualization.

📈 Results & Insights
Cleaned and fully-documented versions of the Cars dataset.

Detailed exploratory analysis covering distributions and summary stats.

Visual insights into how horsepower and mpg vary across car origins and specifications.

Key findings such as average mpg by cylinder count and country of origin.

👤 Author
Ruchir Huchgol

Participant in Summer Analytics 2025

Contributor of this assignment’s Jupyter notebooks and analysis

📄 License & Use
Intended for educational and reference purposes. Attribution appreciated if reused.
Feel free to reach out for questions or suggestions!

✅ Next Steps
Dive deeper with modeling—e.g., regression of mpg on car specs.

Include more aggressive data cleaning (outlier detection, missing values).

Extend analysis to other dataset features like weight-to-power ratio or acceleration.
