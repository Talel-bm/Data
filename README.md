# Machine Learning Trainer for Auto Insurance Pricing

This project provides a Streamlit-based web application for training machine learning models to predict costs and frequencies in auto insurance pricing. It offers a user-friendly interface for data preparation, model selection, hyperparameter tuning, and model evaluation.

## Features

- Data import and preprocessing for insurance datasets
- Support for various machine learning models including scikit-learn, XGBoost, CatBoost, and LightGBM
- Hyperparameter tuning using Optuna
- Model evaluation with cross-validation
- Interactive visualizations of model performance and feature importance
- Export trained models for future use

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/auto-insurance-ml-trainer.git
   cd auto-insurance-ml-trainer
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run __main__.py
   ```

2. Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

3. Follow the on-screen instructions to import your data, select model parameters, and train your model.

## Project Structure

- `__main__.py`: Entry point of the application
- `app.py`: Main application logic and Streamlit UI
- `Importer.py`: Data importing and preprocessing functions
- `jointeurs.py`: Data joining and feature engineering functions
- `ML_trainer.py`: Machine learning model training and evaluation functions
- `Hyperparamter_data.py`: include dictionaries necessary for model hyperparametrisation and explanation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
