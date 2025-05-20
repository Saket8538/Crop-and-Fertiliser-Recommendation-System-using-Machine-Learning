# RECOMMENDER SYSTEM : Crop and Fertilizers Recommendation System

# 1. ABSTRACT
This project aims to optimize agricultural productivity by developing a machine learning-based crop
and fertilizer recommendation system. It evaluates various algorithms—Naive Bayes, Random
Forest and Neural Networks against metrics like precision, recall, F1-score, and accuracy. The initial
phase involved exploratory data analysis and feature scaling to standardize the dataset. Naive
Bayes and Random Forest exhibited exceptional performance, its suitability for datasets with
independent features. Advanced models were also assessed to capture complex patterns.
Confusion matrices were used to fine-tune predictions, guiding improvements for misclassified
instances. The project underscores the importance of selecting an appropriate model based on the
dataset's nuances and cross-validation to ensure model reliability. Finally, a Flask application was
created as an interface, allowing for seamless interaction with the model's recommendations.

# 2. PROBLEM DEFINITION:
The Crop and Fertilizer Recommendation System is a Python Machine Learning project aimed at
recommending optimal crops to farmers based on various soil and environmental factors. The goal is
to leverage data-driven insights to suggest the most suitable crops using different fertilizers, thereby
enhancing agricultural productivity and sustainability.
In this project, we will develop a model that can analyze soil characteristics (like Nitrogen,
Phosphorus, Potassium levels), environmental conditions (temperature, humidity), and rainfall
patterns and type of fertilizer to recommend the most suitable crops for cultivation.
The goal is to predict the type of crop to be recommended, and the type of fertilizer which falls into
distinct categories or classes. For instance, the output might include classes such as Wheat, Rice,
Maize, etc. along with details for fertilizer required based on the soil type.This aligns with the
definition of a classification problem where the target variable is categorical.
# 3. DATASET:
The dataset for this project is sourced from a comprehensive agricultural study and includes key
parameters influencing crop growth.
This data will be used to train and validate our crop recommendation model.
The training and testing data set is obtained from Kaggle Dataset Crops Recommendation dataset:
Case Study on Kaggle Competition : Crop Recommendation Dataset | Kaggle
Fertilizers Recommendation dataset: Github:Yash Thorbole
DATA FIELDS
N - ratio of Nitrogen content in soil
P - ratio of Phosphorous content in soil
K - ratio of Potassium content in soil
temperature - temperature in degree Celsius
humidity - relative humidity in %
ph - ph value of the soil
rainfall - rainfall in mm
fertilizer - There are 7 unique types of fertilizers in the dataset.
# 4. EXPLORATORY DATA ANALYSIS:
3.1 Descriptive Statistics
• Continuous Variables
N, P, K, temperature, humidity, ph, rainfall are all continuous variables.
Nitrogen (N): Ranges from 0 to 140 with a mean of around 50.55.
Phosphorus (P): Ranges from 5 to 145 with a mean of approximately 53.36.
Potassium (K): Has a wide range from 5 to 205, average near 48.15.
Temperature: Varies from 8.83°C to 43.68°C, average around 25.62°C.
Humidity: Ranges widely from 14.26% to nearly 100%, with an average of 71.48%.
pH: Varies from 3.50 to 9.94, with a mean value close to 6.47, which is slightly acidic.
Rainfall: Ranges from 20.21 mm to 298.56 mm, with an average of 103.46 mm.
Fertilizer:
Urea: Contains 37% Nitrogen, 0% Potassium, and 0% Phosphorous.
DAP (Diammonium phosphate): It contains 12% Nitrogen, 0% Potassium, and 36% Phosphorous.
Fourteen-Thirty Five-Fourteen: It contains 7% Nitrogen, 9% Potassium, and 30% Phosphorous.
Twenty Eight-Twenty Eight: It contains 22% Nitrogen, 0% Potassium, and 20% Phosphorous.
Seventeen-Seventeen-Seventeen: Contains 17% Nitrogen, 17% Potassium, and 17% Phosphorous.
Ten-Twenty Six-Twenty Six: Comprises 10% Nitrogen, 26% Potassium, and 26% Phosphorous.
• Categorical Variables for crop Recommendation
Label (Crop Type): There are 22 unique types of crops in the dataset.
• Categorical Variables for fertilizer Recommendation
Fertilizer Name (fertilizer Type): There are 7 unique types of fertilizer in the dataset.
3.2(a) Data Distributions for Crop Data
Continuous Variables
The histograms show the distributions of each continuous variable:
N, P, K: Some display a bimodal nature (having two peaks), suggesting different groups in the data.
Temperature: Appears to be normally distributed.
Humidity: Shows left-skewed distribution, with a high frequency of values towards the higher end.
Rainfall: Displays right-skewed distribution, indicating higher rainfall amounts are less common.
3.2(b) Data Distributions for Fertilizer Data
OBSERVATIONS:
1. Nitrogen:
The histogram exhibits a multimodal distribution with several peaks, suggesting multiple common
values of Nitrogen in the dataset.
The line shows these modes as peaks in the probability density, indicating clusters of data points.
2. Potassium:
The distribution of Potassium is highly skewed towards the lower end, with a sharp peak at the
lowest bin. This skewness is evident in the curve, which has a steep drop-off as the values increase.
3. Phosphorous:
Phosphorous levels are more evenly spread across the range, with a slight concentration at the
lower end. The curve for Phosphorous is flatter than that of Potassium, suggesting less skewness
3.3(a) Outlier Detection for Crop Recommendation
The boxplots provide insights into potential outliers in the continuous variables:

Soil Nutrients (N, P, K) : These features show some outliers, particularly on the higher end. This
could be due to specific crops requiring significantly different nutrient levels.
Temperature: Few outliers are observed, particularly on the lower end.

3.3(b) Outlier Detection for Fertilizer Recommendation
Nitrogen Box Plot:
The interquartile range (IQR), represented by the box, encapsulates the middle 50% of the Nitrogen
data. The median is the central line in the box, dividing the IQR into two equal parts.
Whiskers extend from the hinges of the box to the highest & lowest values within 1.5 * IQR.
Potassium Box Plot:
The box plot for Potassium shows a similar IQR and median.
An outlier is noticeable, marked by a circle above the upper whisker, indicating an unusual value
that stands apart from the rest of the data.
Phosphorous Box Plot:
The IQR and median for Phosphorous are displayed in a similar fashion to the other nutrients.
The distribution of Phosphorous levels appears relatively symmetric around the median.
3.4(a) Correlation Analysis for Crop prediction
The correlation values range from -1 to 1, where
1 indicates a perfect positive correlation,
-1 indicates a perfect negative correlation, and
0 indicates no correlation between the columns.
The heatmap displays the correlation coefficients between the continuous variables:
Strong Correlations: There aren't any extremely strong correlations (> 0.8 or < -0.8) observed,
which is generally a positive sign for building machine learning models
Moderate Correlations: Some moderate correlations are noted. For example, temperature and
humidity show a moderate negative correlation (higher temperatures and lower humidity levels).
Weak Correlations: Most variables show weak correlations with each other, indicating that each
provides unique information for the crop recommendation.
3.4(b) Correlation Analysis for Fertilizer prediction
In this case, we can see that there is a negative correlation between nitrogen and potassium, and
between nitrogen and phosphorous. This means that as the amount of nitrogen increases, the
amount of potassium and phosphorous tends to decrease. A positive correlation between
potassium & phosphorous, as potassium increases, the amount of phosphorous tends to increase.

# 5. FEATURE ENGINEERING:
5.1 (a) Categorical variables for Crop Recommendation
Scale the features using MinMaxScaler
MinMaxScaler is a feature scaling technique that normalizes each feature to a specified range,
typically [0, 1]. It does this by subtracting the minimum value of the feature and then dividing by
the range (the maximum value minus the minimum value).
Why Use MinMaxScaler?
1. Normalizing Measurement Scales: In crop recommendation datasets, features like
temperature, humidity, and soil pH can have different scales and units. MinMaxScaler
ensures that these features with varying ranges don't disproportionately influence model.
2. Improving Model Performance: Many machine learning algorithms perform better when data
is on a similar scale. MinMaxScaler can help in faster convergence and improved
performance, especially for algorithms like neural networks and k-nearest neighbors.
3. Maintaining Proportions: Unlike some scalers, MinMaxScaler preserves the shape of the
original distribution, scaling data points uniformly without reducing importance of outliers.
Importance of Standardization for Your Crop Recommendation Project
In the context of a crop recommendation project, standardizing features like temperature, rainfall,
and pH levels is crucial due to their varying units and scales. Standardization ensures that all these
features contribute equally to the model's predictions, preventing any single feature from
dominating due to its variance or unit.
5. MODEL SELECTION & DEPLOYMENT:
5.1 (a) Model Implementation for Crop Recommendation
1. Naive Bayes (Accuracy: 99.55%)
High Accuracy: With an accuracy of 99.55%, Naive Bayes shows excellent performance in
classifying crop types.
Probability-Based: As a probabilistic classifier, Naive Bayes is effective in making predictions based
on the likelihood of various outcomes, which is valuable in crop recommendation where multiple
factors influence the result.
2. Decision Trees (Accuracy: 98.64%)
Interpretability: Decision Trees provide a clear visualization of the decision-making process, making
it easier to understand how different features contribute to the final recommendation.
Handling Non-Linear Relationships: They are capable of capturing complex, non-linear relationships
between features, which is common in agricultural datasets.
3. Random Forest (Accuracy: 99.32%)
Robustness: Random Forest, an ensemble of Decision Trees, is more robust and less prone to
overfitting compared to a single Decision Tree.
Handling Large Datasets: It excels in handling large datasets with many features, making it ideal for
comprehensive agricultural data.


# 6. ACKNOWLEDGEMENT & REFERENCES:
Here’s your acknowledgment and references section with the same references:  

---

### **ACKNOWLEDGEMENT & REFERENCES**  

Acknowledgement 
We would like to express our sincere gratitude to everyone who contributed to the successful completion of this project. The work was collaboratively divided among team members to ensure efficiency and accuracy.  

I worked on data preprocessing, exploratory data analysis (EDA), and feature engineering. Additionally, model implementation, including Naïve Bayes and Neural Networks, was carried out by Ankur. He also developed a Flask framework to integrate the model for web-based training.  
I contributed to feature engineering, data visualization, and model implementation using Random Forest and Neural Networks. They also developed an HTML interface for interactive crop and fertilizer recommendation.  

We extend our gratitude to our mentors, peers, and online resources for their invaluable guidance and insights, which greatly aided our learning and implementation process.  

References
1. Recommender System lecture Python notebook by Prof. Junwei Huang  
2. [Crop Recommendation using ML](https://ieeexplore.ieee.org/document/9734173)  
3. [Kaggle: Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)  
4. [Fertilizer Kaggle: Plant Disease Classification - ResNet- 99.2%](https://www.kaggle.com)  
5. OpenAI ChatGPT 
