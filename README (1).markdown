# SVM Breast Cancer Classification Project



## Project Overview
Welcome to the **SVM Breast Cancer Classification Project**! This project leverages Support Vector Machines (SVM) to classify breast cancer tumors as malignant (M) or benign (B) using the Breast Cancer Wisconsin dataset. As a data analyst, I designed this project to showcase a complete machine learning pipeline, including data preprocessing, model training, hyperparameter tuning, cross-validation, and decision boundary visualization. The project experiments with both linear and RBF kernels to provide insights into model behavior and performance.

### Key Objectives
- **Data Preprocessing**: Clean and scale the dataset for optimal model performance.
- **Model Training**: Implement SVM with linear and RBF kernels for binary classification.
- **Visualization**: Visualize decision boundaries in 2D to understand model separability.
- **Hyperparameter Tuning**: Optimize `C` and `gamma` for improved accuracy.
- **Evaluation**: Use cross-validation and multiple metrics to assess model robustness.

## Dataset
The Breast Cancer Wisconsin dataset contains 569 samples with 30 features, such as `radius_mean`, `texture_mean`, and more, along with a binary target variable `diagnosis` (M = Malignant, B = Benign). The dataset is stored in `breast-cancer.csv`.

### Dataset Features
- **id**: Unique identifier (dropped during preprocessing)
- **diagnosis**: Target variable (M = 1, B = 0)
- **Features**: 30 numerical features (e.g., `radius_mean`, `texture_mean`, `perimeter_mean`)

## Technologies Used
- **Python 3.8+**: Core programming language
- **Pandas**: Data loading and manipulation
- **NumPy**: Numerical computations
- **Scikit-learn**: SVM implementation, scaling, and evaluation
- **Matplotlib/Seaborn**: Visualization of decision boundaries
- **Jupyter Notebook** (optional): For interactive development

## Prerequisites
Before running the project, ensure you have the following:
- Python 3.8 or higher
- Required libraries:
  ```bash
  pip install pandas numpy scikit-learn matplotlib seaborn
  ```
- The `breast-cancer.csv` dataset in the project directory

## Installation and Usage
Follow these steps to set up and run the project on your local machine:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/svm-breast-cancer-classification.git
   cd svm-breast-cancer-classification
   ```

2. **Set Up a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Alternatively, install the required libraries directly:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```

4. **Prepare the Dataset**:
   - Ensure `breast-cancer.csv` is in the project directory.
   - The script handles data cleaning and preprocessing automatically.

5. **Run the Script**:
   Execute the main script to train the models, tune hyperparameters, and visualize results:
   ```bash
   python svm_breast_cancer_classification.py
   ```
   Alternatively, use a Jupyter Notebook by opening `svm_breast_cancer_classification.ipynb` (if provided).

6. **View Outputs**:
   - **Console Output**: Performance metrics (accuracy, precision, recall, F1-score) for linear and RBF kernel SVMs, best hyperparameters, and cross-validation scores.
   - **Visualizations**: Decision boundary plots for both kernels using `radius_mean` and `texture_mean`.

## Project Workflow
1. **Data Loading and Preprocessing**:
   - Load the dataset using Pandas.
   - Drop the `id` column and map `diagnosis` to binary values (M=1, B=0).
   - Select two features (`radius_mean`, `texture_mean`) for visualization.
   - Scale features using `StandardScaler` for SVM compatibility.

2. **Model Training**:
   - Train SVM models with linear and RBF kernels.
   - Use initial hyperparameters: `C=1`, `gamma='scale'`.

3. **Decision Boundary Visualization**:
   - Plot 2D decision boundaries for both kernels using scaled `radius_mean` and `texture_mean`.
   - Visualize how the models separate malignant and benign classes.

4. **Hyperparameter Tuning**:
   - Use `GridSearchCV` to tune `C` and `gamma` for both kernels.
   - Test `C` values: [0.1, 1, 10, 100]; `gamma` values: [0.001, 0.01, 0.1, 'scale', 'auto'].

5. **Model Evaluation**:
   - Evaluate models using accuracy, precision, recall, and F1-score.
   - Perform 5-fold cross-validation on the best model to ensure robustness.

## Example Output
```plaintext
Linear Kernel SVM Performance:
Accuracy: 0.9123
Precision: 0.9145
Recall: 0.9123
F1-Score: 0.9120

RBF Kernel SVM Performance:
Accuracy: 0.9298
Precision: 0.9302
Recall: 0.9298
F1-Score: 0.9295

Best Parameters from GridSearchCV:
{'C': 10, 'gamma': 0.1, 'kernel': 'rbf'}
Accuracy with Best Model: 0.9415

5-Fold Cross-Validation Scores:
Mean Accuracy: 0.9250 (+/- 0.0321)
```

## Learning Outcomes
This project offers valuable learning opportunities for data analysts and machine learning practitioners:

- **SVM Fundamentals**: Understand how SVM works with linear and RBF kernels for binary classification.
- **Data Preprocessing**: Learn the importance of feature scaling for distance-based algorithms like SVM.
- **Hyperparameter Tuning**: Gain hands-on experience with `GridSearchCV` for optimizing model performance.
- **Model Evaluation**: Explore evaluation techniques like cross-validation and classification metrics.
- **Visualization**: Interpret decision boundaries to understand model behavior and feature separability.
- **Practical Application**: Apply machine learning to a real-world medical dataset for impactful insights.

## Key Insights
- **Model Performance**: The RBF kernel generally outperforms the linear kernel due to its ability to capture non-linear patterns in the data.
- **Feature Separability**: `radius_mean` and `texture_mean` show clear separability between classes, as seen in decision boundary plots.
- **Hyperparameter Impact**: Higher `C` values (e.g., 10) improve accuracy by reducing misclassification, while `gamma` affects the flexibility of the RBF kernel.
- **Robustness**: Cross-validation confirms the model’s generalizability with a mean accuracy of ~92.5%.
- **Ethical Considerations**: Accurate classification is critical in medical applications to avoid misdiagnosis, emphasizing the need for robust evaluation.

## Visualizations
The project generates decision boundary plots to visualize how the SVM models separate malignant and benign tumors:

- **Linear Kernel**: Produces a straight-line boundary, suitable for linearly separable data.
- **RBF Kernel**: Creates a more flexible boundary, capturing non-linear relationships.

![Decision Boundary Example](https://via.placeholder.com/600x400.png?text=Decision+Boundary+Plot)

## Future Improvements
- **Feature Selection**: Incorporate more features or use PCA to improve model performance.
- **Advanced Visualization**: Add interactive Plotly visualizations for better exploration.
- **Ensemble Methods**: Compare SVM with other classifiers like Random Forest or XGBoost.
- **Real-Time Predictions**: Build a web app for users to input tumor measurements and get predictions.
- **Explainability**: Integrate SHAP or LIME to explain model predictions for medical use.

## Contributing
Contributions are welcome! If you'd like to contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Pull Request.



⭐ If you found this project helpful, please give it a star on GitHub! ⭐