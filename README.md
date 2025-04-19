
# Machine Learning for Early Prediction, Detection, and Monitoring of Sepsis in A&E and ICU Settings

## Project Overview  
This project focuses on the development of an intelligent system that leverages machine learning to support early detection, prediction, and continuous monitoring of sepsis in high-risk hospital environments such as Accident & Emergency (A&E) and Intensive Care Units (ICU). The system analyses physiological and biochemical markers to flag patients at risk of sepsis, aiming to reduce patient mortality and enhance clinical decision-making.  

Three machine learning models—**Logistic Regression**, **XGBoost**, and **Gradient Boosting**—were trained and evaluated, with performance assessed using real-world metrics such as F1 score, AUC-ROC, and precision-recall balance.

**Research Question**:  
How can Machine Learning models (Logistic Regression, XGBoost, and Gradient Boosting) be utilised to improve early detection, prediction, and monitoring of sepsis in A&E and ICU, thereby reducing patient mortality rates and enhancing clinical decision-making?

## Data Ethics  
This project exclusively utilises anonymised data from the Kaggle-hosted Sepsis dataset, originally compiled by Johns Hopkins University. The dataset contains no personally identifiable information and complies with both **HIPAA** and **GDPR** regulations. As such, no ethical approval was required.

The research follows University of Hertfordshire ethical guidance and ensures transparency, reproducibility, and responsible use of machine learning in healthcare settings. The deployment-ready dashboard is explicitly intended for educational and prototyping purposes—not live clinical use—until externally validated.

## Data Description  
**Source**: Kaggle – Johns Hopkins Sepsis Dataset  
**Format**: CSV  
**Samples**: 600 ICU patient records  
**Features**:  
- Clinical: Plasma glucose, Blood Pressure, BMI, multiple blood work markers  
- Demographic: Age, Insurance status  
- Target: Sepsis diagnosis (binary classification)

## Project Structure  
- **Data Collection**: Kaggle dataset (2022 upload by chaunguynnghunh)  
- **Data Preparation**:
  - Cleaning missing/zero values, normalisation, and outlier handling
  - Feature selection and transformation
- **Exploratory Data Analysis (EDA)**:
  - Univariate, bivariate, statistical tests
  - SHAP analysis for feature interpretability
- **Model Development**:
  - Logistic Regression, XGBoost, Gradient Boosting  
  - Stratified K-Fold cross-validation  
  - Hyperparameter tuning using RandomizedSearchCV  
- **Evaluation**:
  - F1 Score, Accuracy, Precision, Recall, AUC-ROC  
  - Confusion matrix and ROC analysis  
- **Deployment**:
  - Streamlit dashboard for real-time clinical risk prediction  
  - SHAP explanations and risk score outputs

## Key Features  
- **Balanced Dataset**: Random oversampling used to address class imbalance.  
- **Streamlit App**: Dashboard deployed for real-time sepsis prediction and explanation.  
- **SHAP Explainability**: Identifies top contributing factors to sepsis (e.g., BMI, blood markers).  
- **Model Tuning**: Optimised parameters for best performance.  
- **Performance-Driven**: Gradient Boosting model selected based on F1 score of **0.861**.

## Results  
| Model               | Accuracy | Precision | Recall | F1 Score | AUC-ROC | F1 (Tuned) |
|--------------------|----------|-----------|--------|----------|---------|------------|
| Logistic Regression| 75.4%    | 0.620     | 0.756  | 0.681    | 0.736   | 0.728      |
| XGBoost            | 78%      | 0.640     | 0.780  | 0.703    | 0.753   | 0.854      |
| Gradient Boosting  | 75%      | 0.607     | 0.756  | 0.673    | 0.729   | **0.861**  |

## Requirements  
- Python 3.x  
- Libraries:
  - `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `shap`, `streamlit`, `imblearn`

## Usage  
1. **Clone the repository**:  
```bash
git clone https://github.com/sparkysparo/Sepsis-Prediction-ML.git
cd Sepsis-Prediction-ML
```

2. **Install dependencies**:  
```bash
pip install -r requirements.txt
```

3. **Prepare the dataset**:  
Ensure the Kaggle CSV is available in the `/data` directory.

4. **Run the notebook or training script**:  
Train and evaluate models using:
```bash
Project_MSC_Machine_Learning_for_Early_Prediction,Detection_and_Monitoring_of_Sepsis_in_A&E_and_ICU_Settings.ipynb
```
5. **loading the trained model**:  
The pre-trained models are saved and uploaded to this directory in the repository:

- `Sepsis_gb_model.pkl`: The Gradient Boosting model


6. **Launch Streamlit app**:  
```bash
streamlit run app.py
```

## Streamlit Deployment  
A live demo of the deployed model is available at:  
**🔗 https://prediction-detection-and-monitoring-of-sepsis.streamlit.app**

## Future Work and Directions  
- **Larger Datasets**: Incorporate external ICU/EHR datasets for validation.  
- **Real-Time Monitoring**: Include time-series and wearable sensor data.  
- **Advanced Sampling**: Try SMOTE variants to enhance minority representation.  
- **Clinical Integration**: Collaborate with hospitals for field testing and validation.

## Acknowledgments  
I would like to thank my academic supervisor and peers for their invaluable feedback. This work also benefited from the open-source community and the University of Hertfordshire’s support.

## Contact  
For questions, feedback, or collaboration, please contact:  
📧 sparkysparo@yahoo.com

## Contributing  
Pull requests and issue reports are welcome. Contributions that improve performance, interpretability or deployment are especially encouraged.
