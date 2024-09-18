import datetime
import numpy as np
import pandas as pd
import optuna
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, balanced_accuracy_score, cohen_kappa_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error
from sklearn.metrics import median_absolute_error, mean_poisson_deviance, mean_gamma_deviance
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN
import shap
from interpret import show
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
import statsmodels.api as sm
import xgboost as xgb
with open('.venv/Scripts/baseline_data.pkl', 'rb') as infile:
    data = pickle.load(infile)

"""PARTIE PREPROCESSING"""

# GOUVERNORAT IMPUTER
class GouvernoratImputer(BaseEstimator, TransformerMixin):
    def __init__(self, baseline_data):
        self.baseline_data = baseline_data

    def fit(self, X, y=None):
        self.identifiant = self.baseline_data['identifiant_agence_gouvernorat']
        return self

    def transform(self, X):
        # Copy the input DataFrame to avoid modifying the original data
        X = X.copy()

        # Create a list to store the imputed values for 'GOUVERNORAT_AGENCE'
        L = []

        # Iterate over the rows of the input DataFrame
        for i in X.index:
            # Find the corresponding 'GOUVERNORAT' for the agency code in the current row
            gouvernorat = self.identifiant.loc[self.identifiant['Code'] == X['AGENCE'][i], 'GOUVERNORAT']
            # Append the 'GOUVERNORAT' value if found, otherwise append 'NO DATA'
            if gouvernorat.empty:
                L.append('NO DATA')
            else:
                L.append(gouvernorat.iloc[0])

        # Add the list as a new column 'GOUVERNORAT_AGENCE' in the input DataFrame
        X['GOUVERNORAT_AGENCE'] = L

        # Find the indices of rows where 'GOUVERNORAT' is missing
        missing_gouvernorat_indices = X[X['GOUVERNORAT'].isna()].index

        # Impute the missing 'GOUVERNORAT' values using 'GOUVERNORAT_AGENCE'
        for i in missing_gouvernorat_indices:
            X.at[i, 'GOUVERNORAT'] = X.at[i, 'GOUVERNORAT_AGENCE']
        # Return the modified DataFrame
        return X


class GovernorateClassifier(BaseEstimator, TransformerMixin):
    def __init__(self, target):
        self.target = target
        self.optimal_clusters = None
        self.class_map = None
        self.max_clusters = None

    def fit(self, X):
        df2 = X[['GOUVERNORAT', self.target]]
        yo = df2.groupby('GOUVERNORAT').mean()
        self.max_clusters = len(yo)
        # Determine the optimal number of clusters using the silhouette score
        silhouette_scores = []
        for k in range(2, self.max_clusters):
            kmeans = KMeans(n_clusters=k)
            labels = kmeans.fit_predict(yo)
            score = silhouette_score(yo, labels)
            silhouette_scores.append(score)
        self.optimal_clusters = np.argmax(silhouette_scores) + 2

        # Ensure that the optimal number of clusters is within the valid range
        self.optimal_clusters = min(self.optimal_clusters, len(yo))

        # Clustering with optimal number of clusters
        yup = yo.sort_values(by=self.target, ascending=False)
        kmeans = KMeans(n_clusters=self.optimal_clusters)
        label = kmeans.fit_predict(yup)
        gouvernorat = list(yup.index)
        label_ = list(label)
        self.class_map = {key: value for key, value in zip(gouvernorat, label_)}
        return self

    def transform(self, X):
        last_column = X.columns[-1]
        X_copy = X.copy()
        X_copy['CODE_GOUV'] = X_copy['GOUVERNORAT'].map(self.class_map)
        
        X_copy.drop(['GOUVERNORAT', 'GOUVERNORAT_AGENCE','AGENCE'], axis=1, inplace=True)
        return X_copy.dropna()


class VehicleAgeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, reference_date='2021-12-31'):
        self.reference_date = pd.to_datetime(reference_date)
        self.model = LinearRegression()
        self.imputer = SimpleImputer(strategy='mean')

    def calculate_age(self, date):
        if pd.isna(date):
            return np.nan
        reference_date = pd.to_datetime(self.reference_date)
        date = pd.to_datetime(date)
        return (reference_date - date).days / 365.25

    def extract_y(self, text):
        try:
            if 'TU' in text:
                return int(text[text.find("TU")+2:])
            else:
                return np.nan
        except ValueError:
            return np.nan

    def fit(self, X, y=None):
        recherche = X[['IMMATRICULATION', 'DATE_M_CIRC']].copy()
        recherche['AGE_VEHICULE'] = recherche['DATE_M_CIRC'].apply(self.calculate_age)
        recherche['IMMATRICULATION'] = recherche['IMMATRICULATION'].astype(str)
        recherche['X'] = recherche['IMMATRICULATION'].apply(self.extract_y)

        df = recherche[['X', 'AGE_VEHICULE']]
        mask = (df['AGE_VEHICULE'].notna()) & (df['AGE_VEHICULE'] < 95)
        X_train = df[mask]['X'].values.reshape(-1, 1)
        Y_train = df[mask]['AGE_VEHICULE'].values.reshape(-1, 1)

        X_train = self.imputer.fit_transform(X_train)
        self.model.fit(X_train, Y_train)

        return self

    def transform(self, X):
        if X.empty:
            print("Input X is empty")
        if 'IMMATRICULATION' not in X.columns:
            print("IMMATRICULATION column is missing")
        if X['IMMATRICULATION'].empty:
            print("IMMATRICULATION column is empty")

        recherche = X[['IMMATRICULATION', 'DATE_M_CIRC']].copy()
        recherche['AGE_VEHICULE'] = recherche['DATE_M_CIRC'].apply(self.calculate_age)
        recherche['IMMATRICULATION'] = recherche['IMMATRICULATION'].astype(str)
        recherche['X'] = recherche['IMMATRICULATION'].apply(self.extract_y)

        df = recherche[['X', 'AGE_VEHICULE']]
        mask = (df['AGE_VEHICULE'].notna()) & (df['AGE_VEHICULE'] < 95)

        known_ages = df[mask]['AGE_VEHICULE'].values

        test = df[~mask]['X'].values.reshape(-1, 1)
        if test.size == 0:
            print("all ages are perfect !")
            X_transformed = X.copy()
            X_transformed.drop(['IMMATRICULATION', 'DATE_M_CIRC'], axis=1, inplace=True)
            columns = list(X_transformed.columns)
            X_transformed.insert(len(columns) - 1, 'AGE_VEHICULE', known_ages)
            return X_transformed
        else:
            test = self.imputer.transform(test)
            predicted_ages = self.model.predict(test)
            all_ages = np.concatenate((known_ages, predicted_ages.flatten()))

            X_transformed = X.copy()

            # Delete 'IMMATRICULATION' and 'DATE_M_CIRC' columns
            X_transformed.drop(['IMMATRICULATION', 'DATE_M_CIRC'], axis=1, inplace=True)

            # Add 'AGE_VEHICULE' as the second-to-last column
            columns = list(X_transformed.columns)
            X_transformed.insert(len(columns) - 1, 'AGE_VEHICULE', all_ages)
            return X_transformed


class Descritizer(BaseEstimator, TransformerMixin):
    def __init__(self, baseline_data):
        self.baseline_data = baseline_data

    def fit(self, X, y=None):
        # Extract class boundaries and labels for AGE_VEHICULE and PUISSANCE
        self.age_class_boundaries = self.baseline_data['age_boundaries']
        self.age_class_labels = self.baseline_data['age_labels']
        self.puissance_class_boundaries = self.baseline_data['puissance_boundaries']
        self.puissance_class_labels = self.baseline_data['puissance_labels']

        # Extract code_usage and class_usage for USAGE
        self.code_usage = self.baseline_data['code_usage']
        self.class_usage = self.baseline_data['class_usage']
        self.class_map = {key: value for key, value in zip(self.code_usage, self.class_usage)}

        # Extract mappings for BM
        self.map_210 = self.baseline_data['map_210']
        self.map_autres = self.baseline_data['map_autres']

        return self

    def transform(self, X):
        last_column = X.columns[-1]

        # Copy the input DataFrame to avoid modifying the original data
        X_ = X.copy()

        # Discretize the 'AGE_VEHICULE' column and create a new column named 'c_age'
        X_['c_age'] = pd.cut(X_['AGE_VEHICULE'], bins=self.age_class_boundaries, labels=[int(x[7]) for x in self.age_class_labels])
        # Discretize the 'PUISSANCE' column and create a new column named 'Puissance_Fiscale'
        X_['Puissance_Fiscale'] = pd.cut(X_['PUISSANCE'], bins=self.puissance_class_boundaries, labels=self.puissance_class_labels)

        # Map 'USAGEE' to 'c_usage'
        X_['c_usage'] = X_['USAGEE'].astype(str).map(self.class_map)
        X_['c_usage'].fillna(3, inplace=True)

        # Fill missing values in 'CLASSE_BM' with 0
        X_['CLASSE_BM'].fillna(0, inplace=True)

        # Define the function to map the new BM class
        def map_nouvelle_classe(row):
            if row['CLASSE_BM'] == 0:
                return None
            else:
                if str(row['USAGEE']) == '210':
                    return self.map_210[row['CLASSE_BM']]
                else:
                    if row['USAGEE'] == '2 roues':
                        return 6
                    else:
                        return self.map_autres[row['CLASSE_BM']]

        # Apply the function to create the 'NOUVELLE_CLASSE_BM' column
        X_['NOUVELLE_CLASSE_BM'] = X_.apply(map_nouvelle_classe, axis=1)

        # Drop discretized columns
        X_.drop(['AGE_VEHICULE', 'PUISSANCE', 'USAGEE', 'CLASSE_BM'], axis=1, inplace=True)
        print('X after Descritizer', X_)
        return X_

"""PARTIE FONCTIONS FIT, PREDICT ET SCORE pour tous les modèles utilisés"""
def default_fit(model, X, y):
    return model.fit(X, y)

def default_predict(model, X):
    return model.predict(X)

def default_score(model, X, y, scoring):
    predictions = model.predict(X)
    return scoring(y, predictions)

# Custom functions for statsmodels GLM
def custom_fit_glm(model, X, y):
    X = sm.add_constant(X)  # Add intercept term
    fitted_model = model.fit()
    return fitted_model


def custom_predict_glm(fitted_model, X):
    # Check if the model already has a constant term
    if 'const' not in fitted_model.model.exog_names:
        X = sm.add_constant(X)

    # Ensure X has the same columns as the model's exog_names
    model_features = fitted_model.model.exog_names
    if 'const' in model_features:
        model_features = model_features[1:]  # Exclude 'const' if present
    X = X[model_features]

    return fitted_model.predict(X)

def custom_score_glm(fitted_model, X, y, scoring):
    predictions = custom_predict_glm(fitted_model, X)
    return scoring(y, predictions)


def custom_fit_xgb(model, X, y):
    is_classifier = isinstance(model, xgb.XGBClassifier)
    le = LabelEncoder()

    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].astype('category')

    if is_classifier:
        y_encoded = le.fit_transform(y)
        num_classes = len(le.classes_)
        params = model.get_params()
        if num_classes > 2:
            params['objective'] = 'multi:softprob'
            params['num_class'] = num_classes
        else:
            params['objective'] = 'binary:logistic'
    else:
        y_encoded = y
        params = model.get_params()

    dtrain = xgb.DMatrix(X, label=y_encoded, enable_categorical=True)
    trained_model = xgb.train(params, dtrain)

    # Return the booster model and the label encoder if it's a classifier
    return trained_model


def custom_predict_xgb(model, X):
    import xgboost as xgb
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].astype('category')
    dtest = xgb.DMatrix(X, enable_categorical=True)
    raw_predictions = model.predict(dtest)

    if model.attr('objective') == 'multi:softprob':
        return np.argmax(raw_predictions, axis=1)
    elif model.attr('objective') == 'binary:logistic':
        return (raw_predictions > 0.5).astype(int)
    else:
        return raw_predictions  # For regression tasks


def custom_predict_proba_xgb(model, X):
    import xgboost as xgb
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].astype('category')
    dtest = xgb.DMatrix(X, enable_categorical=True)
    predictions = model.predict(dtest)

    if model.attr('objective') == 'multi:softprob':
        return predictions
    elif model.attr('objective') == 'binary:logistic':
        return np.column_stack((1 - predictions, predictions))
    else:
        raise ValueError("Probability predictions are only available for classification tasks.")


def custom_score_xgb(model, X, y, scoring):
    is_classifier = isinstance(model, xgb.XGBClassifier)
    le = LabelEncoder()
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].astype('category')
    if is_classifier:
        y = le.fit_transform(y)
    predictions = custom_predict_xgb(model, X)
    return scoring(y, predictions)

# Custom functions for LightGBM
def custom_fit_lgbm(model, X, y):
    import lightgbm as lgb
    dtrain = lgb.Dataset(X, label=y)
    params = model.get_params()
    model = lgb.train(params, dtrain)
    return model

def custom_predict_lgbm(model, X):
    import lightgbm as lgb
    return model.predict(X)

def custom_score_lgbm(model, X, y, scoring):
    predictions = custom_predict_lgbm(model, X)
    return scoring(y, predictions)

# Custom functions for CatBoost
def custom_fit_catboost(model, X, y):
    model.fit(X, y, verbose=0)
    return model

def custom_predict_catboost(model, X):
    return model.predict(X)

def custom_score_catboost(model, X, y, scoring):
    predictions = custom_predict_catboost(model, X)
    return scoring(y, predictions)

"""PARTIE RESAMPLING (IMPORTANT POUR LA MODELISATION FREQUENCE)"""
def apply_resampling(X, y, method, random_state):
    if method == 'smote':
        resampler = SMOTE(k_neighbors=1,random_state=random_state)
    elif method == 'random_under':
        resampler = RandomUnderSampler(random_state=random_state)
    elif method == 'smoteenn':
        resampler = SMOTEENN(random_state=random_state)
    elif method == 'smotetomek':
        resampler = SMOTETomek(random_state=random_state)
    else:
        raise ValueError(f"Unsupported resampling method: {method}")
    
    X_resampled, y_resampled = resampler.fit_resample(X, y)
    return X_resampled, y_resampled

"""PARTIE EXPLICATION DU MODELE"""
def explain_model(model, X, y,feature_names, task='Classification'):
    """
    Explain a machine learning model using InterpretML and SHAP.
    
    Parameters:
    - model: The trained model object
    - X: Feature matrix (numpy array or pandas DataFrame)
    - y: Target variable (numpy array or pandas Series)
    - task: 'classification' or 'regression'    
    Returns:
    - A dictionary containing various explanations and performance metrics
    """
    
    # Convert X to numpy array if it's a pandas DataFrame
    if hasattr(X, 'values'):
        X = X.values
    
    # Determine if the model is interpretable or black-box
    if isinstance(model, (ExplainableBoostingRegressor, ExplainableBoostingClassifier)):
        explainer = model
        global_explanation = explainer.explain_global().visualize()
        local_explanation = explainer.explain_local(X[:10], y[:10]).visualize()

    else:
        # Use SHAP for black-box models
        explainer = shap.KernelExplainer(model.predict, X)
        
        # Generate SHAP values for the entire dataset
        shap_values = explainer.shap_values(X)
        
        # Generate global explanations using SHAP summary plot
        global_explanation = shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
        
        # Generate local explanations for a few samples
        num_samples = min(10, X.shape[0])
        local_explanation = shap.force_plot(explainer.expected_value, shap_values[:num_samples], X[:num_samples], feature_names=feature_names, show=False)
    
    # Compile results
    explanations = {
        'global': global_explanation,
        'local': local_explanation,
    }
    return explanations

"""LES DIFFERENTES METRIQUES UTILISEES"""
scoring_functions_regression = {
            'neg_mean_squared_error': lambda y, y_pred: -mean_squared_error(y, y_pred),
            'neg_root_mean_squared_error': lambda y, y_pred: -np.sqrt(mean_squared_error(y, y_pred)),
            'neg_mean_absolute_error': lambda y, y_pred: -mean_absolute_error(y, y_pred),
            'r2': r2_score,
            'adjusted_r2': lambda y, y_pred, X: 1 - (1 - r2_score(y, y_pred)) * (len(y) - 1) / (len(y) - X.shape[1] - 1),
            'neg_mean_absolute_percentage_error': lambda y, y_pred: -np.mean(np.abs((y - y_pred) / y)) * 100,
            'mae': mean_absolute_error,
            'mse': mean_squared_error,
            'rmse': lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred)),
            'friedman_mse': lambda y, y_pred: mean_squared_error(y, y_pred, squared=False),
            'neg_median_absolute_error': lambda y, y_pred: -median_absolute_error(y, y_pred),
            'neg_mean_squared_log_error': lambda y, y_pred: -mean_squared_log_error(y, y_pred),
            'neg_mean_poisson_deviance': lambda y, y_pred: -mean_poisson_deviance(y, y_pred),
            'neg_mean_gamma_deviance': lambda y, y_pred: -mean_gamma_deviance(y, y_pred),
            'explained_variance': lambda y, y_pred: 1 - np.var(y - y_pred) / np.var(y),
            'max_error': lambda y, y_pred: np.max(np.abs(y - y_pred)),
        }

scoring_functions_classification = {
    'accuracy': accuracy_score,
    'balanced_accuracy': balanced_accuracy_score,
    'precision': lambda y, y_pred: precision_score(y, y_pred, average='weighted'),
    'recall': lambda y, y_pred: recall_score(y, y_pred, average='weighted'),
    'f1': lambda y, y_pred: f1_score(y, y_pred, average='weighted'),
    'roc_auc': lambda y, y_pred: roc_auc_score(y, y_pred, multi_class='ovo', average='macro'),
    'neg_log_loss': lambda y, y_pred: -log_loss(y, y_pred),
    'cohen_kappa': cohen_kappa_score,
    'matthews_corrcoef': lambda y, y_pred: np.corrcoef(y, y_pred)[0, 1],
    'jaccard': lambda y, y_pred: np.mean(np.minimum(y, y_pred) / np.maximum(y, y_pred)),
    'hamming_loss': lambda y, y_pred: np.mean(y != y_pred)
}

study_direction_dict = {
    # Regression metrics
    'neg_mean_squared_error': 'minimize',
    'neg_root_mean_squared_error': 'minimize',
    'neg_mean_absolute_error': 'minimize',
    'r2': 'maximize',
    'adjusted_r2': 'maximize',
    'neg_mean_absolute_percentage_error': 'minimize',
    'mae': 'minimize',
    'mse': 'minimize',
    'rmse': 'minimize',
    'friedman_mse': 'minimize',
    'neg_median_absolute_error': 'minimize',
    'neg_mean_squared_log_error': 'minimize',
    'neg_mean_poisson_deviance': 'minimize',
    'neg_mean_gamma_deviance': 'minimize',
    'explained_variance': 'maximize',
    'max_error': 'minimize',

    # Classification metrics
    'accuracy': 'maximize',
    'balanced_accuracy': 'maximize',
    'precision': 'maximize',
    'recall': 'maximize',
    'f1': 'maximize',
    'roc_auc': 'maximize',
    'neg_log_loss': 'minimize',
    'cohen_kappa': 'maximize',
    'matthews_corrcoef': 'maximize',
    'jaccard': 'maximize',
    'hamming_loss': 'minimize'
}


def optimize_and_evaluate_model(
    probleme,
    df,
    target_column,
    model_class,
    param_distributions,
    split_method,
    test_size=0.2,
    time_column=None,
    train_years=None,
    test_year=None,
    n_splits=5,
    n_trials=100,
    scoring='neg_mean_squared_error',
    random_state=42,
    custom_fit=None,
    custom_predict=None,
    custom_score=None,
    resampling_method=None,
    explain=False,
    progress_bar=None,
    status_text=None,
    annee_reference = None,
):
    scoring_functions = {}
    # Define scoring functions
    if probleme == "Regression":
        scoring_functions = scoring_functions_regression
    elif probleme == "Classification":
        scoring_functions = scoring_functions_classification
    else:
        raise ValueError("Invalid problem type. Choose either 'Regression' or 'Classification'.")

    if scoring not in scoring_functions:
        raise ValueError(f"Unsupported scoring metric: {scoring}. Choose from {list(scoring_functions.keys())}")
    
    # Select the scoring function
    scoring_func = scoring_functions[scoring]

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Perform train-test split
    if split_method == 'Random':
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    elif split_method == 'Backtesting':
        if time_column is None or train_years is None or test_year is None:
            raise ValueError("For backtesting, time_column, train_years, and test_year must be specified.")
        
        train_mask = df[time_column].dt.year.isin(train_years)
        test_mask = df[time_column].dt.year == test_year
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        X_train.drop(time_column,axis =1, inplace = True)
        X_test.drop(time_column,axis = 1, inplace = True)

    def objective(trial):
        # Generate hyperparameters for the model
        model_params = {
        param: (
            trial.suggest_categorical(param, choices) if isinstance(choices, list)
            else trial.suggest_int(param, *choices) if isinstance(choices, tuple) and all(isinstance(x, int) for x in choices)
            else trial.suggest_float(param, *choices) if isinstance(choices, tuple) and any(isinstance(x, float) for x in choices)
            else choices  # for fixed parameters
        )
        for param, choices in param_distributions.items()
    }

        # Perform cross-validation
        if split_method == 'Random':
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        else:  # backtesting
            cv = [np.arange(len(X_train)), np.arange(len(X_train))]  # No CV for backtesting

        # Use custom or default fit, predict, and score functions
        fit_func = custom_fit if custom_fit else default_fit
        predict_func = custom_predict if custom_predict else default_predict
        score_func = custom_score if custom_score else default_score

        scores = []
        for train_idx, val_idx in cv.split(X_train):
            X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

            preprocessing_pipeline = Pipeline([
                        ('GouvernoratImputer', GouvernoratImputer(baseline_data=data)),
                        ('GovernorateClassifier', GovernorateClassifier(target=target_column)),
                        ('vehicle_age', VehicleAgeTransformer(reference_date= pd.to_datetime(f"{annee_reference}-12-31")))
                        #('descritize', Descritizer(baseline_data=data))
                        ], verbose=True)

            X_ = X_train_fold.copy()
            X_[target_column] = y_train_fold
            X_transformed = preprocessing_pipeline.fit_transform(X_)
            X_transformed = X_transformed.dropna()
            X_train_fold_transformed = X_transformed.drop(columns=[target_column])
            if "SURVENANCE" in X_train_fold_transformed.columns:
                X_train_fold_transformed = X_train_fold_transformed.drop(columns=["SURVENANCE"])
            y_train_fold = X_transformed[target_column]

            X__ = X_val_fold.copy()
            X__[target_column] = y_val_fold
            X__transformed = preprocessing_pipeline.transform(X__)
            X__transformed = X__transformed.dropna()
            X_val_fold_transformed = X__transformed.drop(columns=[target_column])
            if "SURVENANCE" in X_val_fold_transformed.columns:
                X_val_fold_transformed = X_val_fold_transformed.drop(columns=["SURVENANCE"])
            y_val_fold = X__transformed[target_column]


            #treating some non float categories
            if 'Puissance_Fiscale' in X_train_fold_transformed.columns:
                X_train_fold_transformed['Puissance_Fiscale'] = X_train_fold_transformed['Puissance_Fiscale'].astype('int')
            if 'ENERGIE' in X_train_fold_transformed.columns:
                le = LabelEncoder()
                X_train_fold_transformed['ENERGIE'] = le.fit_transform(X_train_fold_transformed['ENERGIE'])

            if 'Puissance_Fiscale' in X_val_fold_transformed.columns:
                X_val_fold_transformed['Puissance_Fiscale'] = X_val_fold_transformed['Puissance_Fiscale'].astype('int')
            if 'ENERGIE' in X_val_fold_transformed.columns:
                le = LabelEncoder()
                X_val_fold_transformed['ENERGIE'] = le.fit_transform(X_val_fold_transformed['ENERGIE'])
            if 'SEXE' in X_val_fold_transformed.columns:
                X_val_fold_transformed['SEXE'] = X_val_fold_transformed['SEXE'].map({'M':2,'F':1})
            print(X_val_fold_transformed)
            #Resampling :
            if resampling_method:
                X_train_fold_transformed, y_train_fold = apply_resampling(
                    X_train_fold_transformed, y_train_fold, resampling_method, random_state=42
                )

            # Fit the model with the formula
            if model_class == sm.GLM:
                model = sm.GLM(y_train_fold, X_train_fold_transformed, **model_params)
            elif (model_class == ExplainableBoostingClassifier) or (model_class == ExplainableBoostingRegressor):
                model = model_class(feature_names = X_train_fold_transformed.columns, **model_params)
            else :
                model = model_class(**model_params)

            fitted_model = fit_func(model, X_train_fold_transformed, y_train_fold)

            # Score the predictions
            score = score_func(fitted_model, X_val_fold_transformed, y_val_fold, scoring_functions[scoring])
            scores.append(score)

        mean_score = np.mean(scores)

        return mean_score

    def optimization_callback(study, trial):
        if progress_bar is not None and status_text is not None:
            progress = len(study.trials) / n_trials
            progress_bar.progress(progress)
            status_text.text(f'Iteration {len(study.trials)}/{n_trials}')

    # Create a study object and optimize the objective function
    study = optuna.create_study(direction=study_direction_dict[scoring])
    study.optimize(objective, n_trials=n_trials, callbacks=[optimization_callback])
    
    # Get the best parameters and score
    best_params = study.best_params
    best_cv_score = study.best_value
    
    #Creating preprocessed dataframes
    preprocessing_pipeline = Pipeline([
        ('GouvernoratImputer', GouvernoratImputer(baseline_data=data)),
        ('GovernorateClassifier', GovernorateClassifier(target=target_column)),
        ('vehicle_age', VehicleAgeTransformer(reference_date=str(annee_reference)+'-12-31'))
        #('descritize', Descritizer(baseline_data=data))
    ], verbose=True)

    X_ = X_train.copy()
    X_[target_column] = y_train
    X_transformed = preprocessing_pipeline.fit_transform(X_)
    X_transformed = X_transformed.dropna()
    y_train = X_transformed[target_column]
    X_train_transformed = X_transformed.drop(columns=[target_column])
    if "SURVENANCE" in X_train_transformed.columns:
        X_train_transformed = X_train_transformed.drop(columns=["SURVENANCE"])
           

    X__ = X_test.copy()
    X__[target_column] = y_test
    X__transformed = preprocessing_pipeline.transform(X__)
    X__transformed = X__transformed.dropna()
    y_test = X__transformed[target_column]
    X_test_transformed = X__transformed.drop(columns=[target_column])
    if "SURVENANCE" in X_test_transformed.columns:
        X_test_transformed = X_test_transformed.drop(columns=["SURVENANCE"])
    #treating some non float categories
    if 'Puissance_Fiscale' in X_train_transformed.columns:
        X_train_transformed['Puissance_Fiscale'] = X_train_transformed['Puissance_Fiscale'].astype('int')
    if 'ENERGIE' in X_train_transformed.columns:
        le = LabelEncoder()
        X_train_transformed['ENERGIE'] = le.fit_transform(X_train_transformed['ENERGIE'])

    if 'Puissance_Fiscale' in X_test_transformed.columns:
        X_test_transformed['Puissance_Fiscale'] = X_test_transformed['Puissance_Fiscale'].astype('int')
    if 'ENERGIE' in X_test_transformed.columns:
        le = LabelEncoder()
        X_test_transformed['ENERGIE'] = le.fit_transform(X_test_transformed['ENERGIE'])
    if 'SEXE' in X_test_transformed.columns:
        le = LabelEncoder()
        X_test_transformed['SEXE'] = X_test_transformed['SEXE'].map({'M':2,'F':1})
    #Resampling
    if resampling_method:
        X_train_transformed, y_train = apply_resampling(
            X_train_transformed, y_train, resampling_method, random_state
        )
    # Create the best model with optimized hyperparameters

    if model_class == sm.GLM:
        best_model = sm.GLM(y_train, X_train_transformed, **best_params)

    elif (model_class == ExplainableBoostingClassifier) or (model_class == ExplainableBoostingRegressor):
        best_model = model_class(feature_names=X_train_transformed.columns, **best_params)

    else:
        best_model = model_class(**best_params)
    
    # Fit the best pipeline on the entire training data
    fit_func = custom_fit if custom_fit else default_fit
    fitted_best_model = fit_func(best_model, X_train_transformed, y_train)
    
    # Evaluate the best pipeline on the test data
    predict_func = custom_predict if custom_predict else default_predict
    score_func = custom_score if custom_score else default_score
    
    predictions = predict_func(fitted_best_model, X_test_transformed)
    test_score = score_func(fitted_best_model, X_test_transformed, y_test, scoring_functions[scoring])

     # Model explanation
    if explain:
        feature_names = list(X_train_transformed.columns)
        explanations = explain_model(
            fitted_best_model, 
            X_test_transformed, 
            y_test, 
            task='Classification' if probleme == 'Classification' else 'Regression',
            feature_names = feature_names
        )
    else:
        explanations = None
    
    return best_params, best_cv_score, test_score, fitted_best_model, explanations