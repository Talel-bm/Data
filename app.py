#Importations
import pandas as pd
import json
import os
from pathlib import Path
import ML_trainer
from Importer import DataImporter
from hyperparameter_data import hyperparameters_data,model_explanation
import streamlit as st
import tempfile
from io import BytesIO
import time
import statsmodels.api as sm
import xgboost as xgb
import catboost as catb
import lightgbm as lgbm
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet, BayesianRidge, SGDRegressor, PassiveAggressiveRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier, AdaBoostRegressor
from sklearn.svm import SVR, NuSVC, NuSVR, OneClassSVM
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, RadiusNeighborsClassifier, RadiusNeighborsRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.linear_model import SGDClassifier, Perceptron, RidgeClassifier, RidgeClassifierCV
from sklearn.dummy import DummyClassifier, DummyRegressor
import shap
import pickle

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    st.components.v1.html(shap_html, height=height)

model_mapping = {
    "LogisticRegression": LogisticRegression,
    "Ridge": Ridge,
    "Lasso": Lasso,
    "ElasticNet": ElasticNet,
    "BayesianRidge": BayesianRidge,
    "SGDRegressor": SGDRegressor,
    "PassiveAggressiveRegressor": PassiveAggressiveRegressor,
    "RandomForestRegressor": RandomForestRegressor,
    "ExtraTreesRegressor": ExtraTreesRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "RandomForestClassifier": RandomForestClassifier,
    "ExtraTreesClassifier": ExtraTreesClassifier,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "SVR": SVR,
    "KNeighborsClassifier": KNeighborsClassifier,
    "KNeighborsRegressor": KNeighborsRegressor,
    "AdaBoostClassifier": AdaBoostClassifier,
    "AdaBoostRegressor": AdaBoostRegressor,
    "MLPClassifier": MLPClassifier,
    "MLPRegressor": MLPRegressor,
    "GaussianProcessRegressor": GaussianProcessRegressor,
    "GaussianProcessClassifier": GaussianProcessClassifier,
    "SGDClassifier": SGDClassifier,
    "NuSVC": NuSVC,
    "NuSVR": NuSVR,
    "OneClassSVM": OneClassSVM,
    "RadiusNeighborsClassifier": RadiusNeighborsClassifier,
    "RadiusNeighborsRegressor": RadiusNeighborsRegressor
}



def create_test_data():
    st.title("Machine Learning Trainer pour la tarification auto")
    st.subheader("La Base de modélisation")
    option = st.radio("Choisissez une option", ["Créer une nouvelle base de modélisation","Importer un fichier depuis le PC"])

    if option == "Importer un fichier depuis le PC":
        st.subheader("Importer les données de test")
        test_data_file = st.file_uploader("Importer les données de test (CSV ou Excel)", type=["csv", "xlsx"])

        if test_data_file:
            if test_data_file.type == "text/csv":
                test_data = pd.read_csv(test_data_file)
            else:
                test_data = pd.read_excel(test_data_file, engine='calamine')
            st.session_state.test_data = test_data
            st.success("Données de test importées avec succès!")
        st.subheader("On vise modéliser :")
        model_type = st.radio("", ("Cout", "Fréquence"))

        st.subheader("Paramètres")

        # Axes de modélisation parameter
        axes = st.multiselect("Axes de modélisation",
                              ["Age Vehicule", "Gouvernorat", "Classe BM", "Puissance Fiscale", "Energie", "Usage",
                               "Sexe"])

        axes_modelisation = ['AGENCE', 'USAGEE']
        if "Age Vehicule" in axes:
            axes_modelisation.append('IMMATRICULATION')
            axes_modelisation.append('DATE_M_CIRC')
        if "Gouvernorat" in axes:
            axes_modelisation.append('GOUVERNORAT')
        if "Classe BM" in axes:
            axes_modelisation.append('CLASSE_BM')
        if "Puissance Fiscale" in axes:
            axes_modelisation.append('PUISSANCE')
        if "Energie" in axes:
            axes_modelisation.append('ENERGIE')
        if "Sexe" in axes:
            axes_modelisation.append('SEXE')

        # Annee de reference parameter (if model_type is "Cout")
        annee_reference = st.text_input("Année de référence",
                                        key="annee_reference")


        # Save to session state and proceed
        if st.button("Sauvegarder les paramètres et continuer"):
            st.session_state.model_type = model_type
            st.session_state.axes_modelisation = axes_modelisation
            st.session_state.annee_ref = annee_reference
            st.success("Paramètres sauvegardés avec succès! Vous pouvez maintenant passer à la page suivante.")

            st.session_state.page = 'process_model'



    elif option == "Créer une nouvelle base de modélisation":
        st.subheader("On vise modéliser :")
        model_type = st.radio("", ("Cout", "Fréquence"))

        if 'data_importer' not in st.session_state:
            st.session_state.data_importer = DataImporter()
        importer = st.session_state.data_importer

        # Prod Data Section
        st.subheader("Bases production")
        prod_years = st.text_input("Années (séparer par des virgules)", "", key="prod_years")
        prod_files = {}
        for i, year in enumerate(prod_years.split(",")):
            year = year.strip()
            file = st.file_uploader(f"Importer la base production pour l'année {year}", type=["csv", "xlsx"], key=f"prod_file_{i}")
            sheet_name = st.text_input(f"Nom de la feuille (si fichier Excel) pour l'année {year}", key=f"prod_sheet_{i}")
            if file:
                prod_files[year] = {'file': file, 'sheet_name': sheet_name}

        # Sinistres Data Section
        st.subheader("Bases sinistres")
        sinistres_years = st.text_input("Années (séparer par des virgules)", "", key="sinistres_years")
        sinistres_files = {}
        for i, year in enumerate(sinistres_years.split(",")):
            year = year.strip()
            file = st.file_uploader(f"Importer la base sinistres pour l'année {year}", type=["csv", "xlsx"], key=f"sinistres_file_{i}")
            sheet_name = st.text_input(f"Nom de la feuille (si fichier Excel) pour l'année {year}", key=f"sinistres_sheet_{i}")
            if file:
                sinistres_files[year] = {'file': file, 'sheet_name': sheet_name}

        st.subheader("Paramètres")

        # Garantie and Usage parameters
        garantie = st.radio("Garantie", ["RC MAT", "RC CRP", "BG", "VOL&INC"])
        usage = st.radio("Usage", ["210", "NON 210"])

        # Axes de modélisation parameter
        axes = st.multiselect("Axes de modélisation", ["Age Vehicule", "Gouvernorat", "Classe BM", "Puissance Fiscale", "Energie", "Usage", "Sexe"])
        axes_modelisation = ['AGENCE','USAGEE']
        if "Age Vehicule" in axes:
            axes_modelisation.append('IMMATRICULATION')
            axes_modelisation.append('DATE_M_CIRC')
        if "Gouvernorat" in axes:
            axes_modelisation.append('GOUVERNORAT')
        if "Classe BM" in axes:
            axes_modelisation.append('CLASSE_BM')
        if "Puissance Fiscale" in axes:
            axes_modelisation.append('PUISSANCE')
        if "Energie" in axes:
            axes_modelisation.append('ENERGIE')
        if "Sexe" in axes:
            axes_modelisation.append('SEXE')


        # Seuil MAT and Seuil CRP parameters
        seuil_mat = st.text_input("Seuil MAT") if garantie == 'RC MAT' else None
        seuil_crp = st.text_input("Seuil CRP") if garantie == 'RC CRP' else None

        # RC Parameters
        inflation_values = {}
        if (model_type == 'Cout') and ('RC' in garantie):
            inflation_rc = st.checkbox("Ajouter de l'inflation pour la RC", key="inflation_rc")
            if inflation_rc:
                for i, year in enumerate(prod_years.split(",")):
                    year = year.strip()
                    inflation_values[year] = st.text_input(f"Inflation de l'année {year}", key=f"inflation_{i}")

        # Annee de reference parameter
        annee_reference = st.text_input("Année de référence", key="annee_reference")

        # Create test data button
        if st.button("Create Test Data"):
            prod_file_paths = {}
            for year, file_info in prod_files.items():
                file = file_info['file']
                sheet_name = file_info['sheet_name']
                if file is not None:
                    file_extension = Path(file.name).suffix.lower()
                    if file_extension in ['.xlsx', '.csv']:
                        prod_file_paths[int(year)] = {'file': file, 'sheet_name': sheet_name}
                    else:
                        st.error(f"Unsupported file format for year {year}. Please upload .xlsx or .csv files.")

            sinistres_file_paths = {}
            for year, file_info in sinistres_files.items():
                file = file_info['file']
                sheet_name = file_info['sheet_name']
                if file is not None:
                    file_extension = Path(file.name).suffix.lower()
                    if file_extension in ['.xlsx', '.csv']:
                        sinistres_file_paths[int(year)] = {'file': file, 'sheet_name': sheet_name}
                    else:
                        st.error(f"Unsupported file format for year {year}. Please upload .xlsx or .csv files.")

                # Get test data
            with st.spinner('Création des données de test en cours...'):
                test_data = importer.get_test_data(
                    prod_files=prod_files,
                    sinistres_files=sinistres_files,
                    garantie=garantie,
                    usage=usage,
                    axes_modelisation=axes_modelisation,
                    seuil_mat=float(seuil_mat) if seuil_mat else None,
                    seuil_crp=float(seuil_crp) if seuil_crp else None,
                    use_cout=(model_type.lower() == "cout"),
                    annee_reference=int(annee_reference) if annee_reference else None,
                    capital_assure=1000,
                    inflation_rcmat={year: float(value) for year, value in
                                     inflation_values.items()} if garantie == 'RC MAT' else None,
                    inflation_rccrp={year: float(value) for year, value in
                                     inflation_values.items()} if garantie == 'RC CRP' else None
                )

            # Display test data
            st.success('Données de test créées avec succès!')
            st.write("Test Data:")
            st.write(test_data)
            # Store in session state
            st.session_state.test_data = test_data
            st.session_state.model_type = model_type
            st.session_state.axes_modelisation = axes_modelisation
            st.session_state.annee_ref = annee_reference

            # Excel limitations
            EXCEL_MAX_ROWS = 1_048_576
            EXCEL_MAX_COLS = 16_384

            # Prepare the file name (without extension)
            file_name_base = f"{model_type}_{garantie}_{usage}{prod_years}_test_data"

            # Check if the dataframe exceeds Excel limitations
            if test_data.shape[0] > EXCEL_MAX_ROWS or test_data.shape[1] > EXCEL_MAX_COLS:
                # If it exceeds Excel limitations, create and offer CSV download
                csv_data = test_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Test Data as CSV",
                    data=csv_data,
                    file_name=f"{file_name_base}.csv",
                    mime="text/csv"
                )
                st.info("The dataset exceeds Excel limitations. A CSV file is provided instead.")
            else:
                # If it's within Excel limitations, offer Excel download
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    test_data.to_excel(writer, index=False, sheet_name='Test Data')
                excel_data = output.getvalue()

                st.download_button(
                    label="Download Test Data as Excel",
                    data=excel_data,
                    file_name=f"{file_name_base}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            # Offer "Next" button to proceed to model processing
            st.session_state.page = 'process_model'


    return None

def process_model():
    st.markdown("## Model Processing")

    if 'test_data' not in st.session_state:
        st.error("No test data available. Please create test data first.")
        return


    test_data = st.session_state.test_data
    model_type = st.session_state.model_type
    axes_modelisation = st.session_state.axes_modelisation
    annee_reference = st.session_state.annee_ref
    st.markdown("#### **Type de problème**")
    problem_type = st.selectbox("", ["Regression","Classification", ])
    st.write('NB: la prédiction du cout est un problème de régression alors que la prédiction de la fréquence est un problème de classification')
    st.markdown("#### **Bibliothèque**")
    if problem_type == "Regression":
        model_library = st.selectbox("", ["scikit-learn", "sm.glm", "xgboost", "catboost", "lightgbm","interpret"])

        st.markdown("#### **Model class**")
        regression_models = [
            "GradientBoostingRegressor",
            "RandomForestRegressor",
            "ExtraTreesRegressor",
            "LinearRegression",
            "Ridge",
            "Lasso",
            "ElasticNet",
            "BayesianRidge",
            "SVR",
            "KNeighborsRegressor",
            "AdaBoostRegressor",
            "MLPRegressor",
            "GaussianProcessRegressor",
            "SGDRegressor",
            "PassiveAggressiveRegressor",
            "NuSVR",
            "RadiusNeighborsRegressor"
        ]
        model_classes = {
        "scikit-learn": regression_models,
        "sm.glm": ["GLM"],
        "xgboost": ["XGBRegressor"],
        "catboost": ["CatBoostRegressor"],
        "lightgbm": ["LGBMRegressor"],
        "interpret" :["ExplainableBoostingRegressor"]
    }
        model_class = st.selectbox("", model_classes[model_library])
        if model_class =="GLM":
            st.write('NB : Pour le GLM il est obligatoire de choisir un hyperparamètre "family"')

    if problem_type == "Classification":
        model_library = st.selectbox("", ["scikit-learn", "xgboost", "catboost", "lightgbm",'interpret'])

        st.markdown("#### **Model class**")
        classification_models = [
            "GradientBoostingClassifier",
            "RandomForestClassifier",
            "ExtraTreesClassifier",
            "LogisticRegression",
            "KNeighborsClassifier",
            "AdaBoostClassifier",
            "MLPClassifier",
            "GaussianProcessClassifier",
            "SGDClassifier",
            "NuSVC",
            "RadiusNeighborsClassifier"
        ]
        model_classes = {
        "scikit-learn": classification_models,
        "xgboost": ["XGBClassifier"],
        "catboost": ["CatBoostClassifier"],
        "lightgbm": ["LGBMClassifier"],
        "interpret": ["ExplainableBoostingClassifier"]
    }
        model_class = st.selectbox("", model_classes[model_library])

    if st.checkbox("Voir l'explication du modèle"):
        st.markdown(model_explanation[model_class])

    def convert_to_numeric(value):
        """Helper function to convert a string to a numeric type if possible."""
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            return value  # Return as is if conversion fails

    def create_hyperparameter_widgets(model_class, hyperparameters_data):
        st.markdown("#### **Hyperparameters**")
        hyperparameters = {}
        int_params = [
            # XGBoost
            'max_depth', 'n_estimators', 'num_parallel_tree', 'max_leaves',
            'max_bin', 'min_child_weight', 'gpu_id',

            # LightGBM
            'num_leaves', 'max_depth', 'max_bin', 'min_data_in_leaf',
            'min_sum_hessian_in_leaf', 'n_estimators', 'num_iterations',
            'bagging_freq', 'min_data_per_group', 'max_cat_threshold','subsample_for_bin',

            # CatBoost
            'iterations', 'depth', 'border_count', 'thread_count',

            # Scikit-learn tree-based models
            'max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf',
            'max_features', 'max_leaf_nodes', 'min_impurity_decrease',
            'ccp_alpha', 'n_iter_no_change','n_neighbors','max_iter'
        ]
        for param, config in hyperparameters_data[model_class].items():
            include = st.checkbox(
                f"**Include {param}**",
                value=False,
                help=config["help"]
            )

            if include:
                widget_type = config["type"]
                args = config["args"]
                if widget_type == "checkbox":
                    hyperparameters[param] = st.checkbox(*args)
                elif widget_type == "text_input":
                    if param in int_params:
                        hyperparameters[param] = [
                            int(x) for x in st.text_input(*args).split(',')
                        ]
                    else:
                        hyperparameters[param] = [
                        float(x) for x in st.text_input(*args).split(',')
                    ]
                elif widget_type == "multiselect":
                    selected_values = st.multiselect(*args)
                    if param == "family":
                        # Mapping of family names to statsmodels family objects
                        family_mapping = {
                            'gaussian': sm.families.Gaussian(),
                            'binomial': sm.families.Binomial(),
                            'poisson': sm.families.Poisson(),
                            'gamma': sm.families.Gamma(),
                            'inverse_gaussian': sm.families.InverseGaussian(),
                            'tweedie': sm.families.Tweedie()
                        }
                        hyperparameters[param] = [family_mapping[value] for value in selected_values]
                    else :
                        hyperparameters[param] = selected_values

            else:
                hyperparameters[param] = None

        # Filter out None values and convert string-based lists to appropriate types
        hyperparameters = {key: value for key, value in hyperparameters.items() if value is not None}
        for key, value in hyperparameters.items():
            if isinstance(value, str):
                if value.lower() == '0':
                    hyperparameters[key] = None
                elif ',' in value:
                    try:
                        # Try to convert to float list
                        hyperparameters[key] = [float(x.strip()) for x in value.split(',')]
                    except ValueError:
                        # If conversion fails, keep as string list
                        hyperparameters[key] = [x.strip() for x in value.split(',')]
                else:
                    # Try to convert single values to numeric
                    hyperparameters[key] = convert_to_numeric(value)

        return hyperparameters

    hyperparameters = create_hyperparameter_widgets(model_class, hyperparameters_data)
    st.json(hyperparameters)

    st.markdown("#### **Split method**")
    split_method = st.selectbox("", ["Random", "Backtesting"])

    test_size = None
    train_years = None
    test_year = None
    if split_method == "Random":
        st.markdown("#### **Test size**")
        test_size = st.number_input("", min_value=0.0, max_value=1.0, value=0.2)

    if split_method == "Backtesting":
        st.markdown("#### **Test year**")
        test_year = st.text_input("", key="test_year")
        st.markdown("#### **Train years**")
        train_years = st.text_input("(comma separated)", key="train_years")

    st.markdown("#### Cross-Validation params : ")
    st.markdown("#### **Number of splits**")
    n_splits = st.number_input("", min_value=1, value=5, key="n_splits")

    st.markdown("#### **Number of trials**")
    n_trials = st.number_input("", min_value=1, value=10, key="n_trials")

    st.markdown("#### **Scoring/Métrique**")
    if problem_type == 'Classification':
        scoring = st.selectbox("", ['roc_auc', 'accuracy', 'balanced_accuracy', 'recall', 'f1', 'neg_log_loss',
                                    'cohen_kappa', 'matthews_corrcoef', 'jaccard', 'hamming_loss'])
    elif problem_type == 'Regression':
        scoring = st.selectbox("", ['mse','r2','mae','rmse','neg_mean_squared_error','neg_mean_absolute_error','neg_root_mean_squared_error',
    'explained_variance','max_error','adjusted_r2','neg_mean_absolute_percentage_error','friedman_mse','neg_median_absolute_error',
    'neg_mean_squared_log_error','neg_mean_poisson_deviance','neg_mean_gamma_deviance'
])

    resampling_method = None
    if model_type == 'Fréquence':
        st.markdown("#### **Resampling**")
        resampling_method = st.selectbox('',['None','smote','random_under','smoteenn','smotetomek'])
        if resampling_method == "None":
            resampling_method = None

    #Showing progress of optimisation function
    def show_optimization_progress(
            probleme,
            df,
            target_column,
            model_class,
            param_distributions,
            split_method,
            test_size,
            time_column,
            train_years,
            test_year,
            n_splits,
            n_trials,
            scoring,
            random_state,
            custom_fit,
            custom_predict,
            custom_score,
            resampling_method,
            explain,
            annee_reference
    ):
        st.write("### Optimization and Evaluation Progress")

        # Setup progress bar and status text
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Setup expandable sections for detailed progress
        data_prep = st.expander("Data Preparation", expanded=False)
        cv_progress = st.expander("Cross-Validation Progress", expanded=False)
        best_model = st.expander("Best Model Details", expanded=False)
        final_eval = st.expander("Final Evaluation", expanded=False)

        # Start optimization
        start_time = time.time()

        def optimization_callback(study, trial):
            progress = len(study.trials) / n_trials
            progress_bar.progress(progress)
            status_text.text(f'Optimization Progress: {len(study.trials)}/{n_trials} trials')

            with cv_progress:
                st.write(f"Trial {len(study.trials)}: Best score so far: {study.best_value:.4f}")
                st.write(f"Current trial params: {trial.params}")

        best_params, best_cv_score, test_score, fitted_best_model, explanations = ML_trainer.optimize_and_evaluate_model(
            probleme=probleme,
            df=df,
            target_column=target_column,
            model_class=model_class,
            param_distributions=param_distributions,
            split_method=split_method,
            test_size=test_size,
            time_column=time_column,
            train_years=train_years,
            test_year=test_year,
            n_splits=n_splits,
            n_trials=n_trials,
            scoring=scoring,
            random_state=random_state,
            custom_fit=custom_fit,
            custom_predict=custom_predict,
            custom_score=custom_score,
            resampling_method=resampling_method,
            explain=explain,
            progress_bar=progress_bar,
            status_text=status_text,
            annee_reference=annee_reference
        )

        end_time = time.time()

        # Best Model Details
        with best_model:
            st.write("### Best Model Details")
            st.write(f"Best parameters: {best_params}")
            st.write(f"Best cross-validation score: {best_cv_score:.4f}")

        # Final Evaluation
        with final_eval:
            st.write("### Final Evaluation")
            st.write(f"Test score: {test_score:.4f}")
            st.write(f"Total optimization time: {end_time - start_time:.2f} seconds")

        return best_params, best_cv_score, test_score, fitted_best_model, explanations

    # Process model button
    if st.button("Process Model"):
        # Parameterization of the function
        custom_fit = None
        custom_predict = None
        custom_score = None

        model = None  # Initialize model to None

        if model_library == 'scikit-learn':
            model = model_mapping[model_class]

        if model_library == "sm.glm":
            custom_fit = ML_trainer.custom_fit_glm
            custom_predict = ML_trainer.custom_predict_glm
            custom_score = ML_trainer.custom_score_glm
            model = sm.GLM

        if model_library == 'xgboost':
            custom_fit = ML_trainer.custom_fit_xgb
            custom_predict = ML_trainer.custom_predict_xgb
            custom_score = ML_trainer.custom_score_xgb
            if problem_type == "Regression":
                model = xgb.XGBRegressor
            elif problem_type == "Classification":
                model = xgb.XGBRFClassifier

        if model_library == "catboost":
            custom_fit = ML_trainer.custom_fit_catboost
            custom_predict = ML_trainer.custom_predict_catboost
            custom_score = ML_trainer.custom_score_catboost
            if problem_type == "Regression":
                model = catb.CatBoostRegressor
            elif problem_type == "Classification":
                model = catb.CatBoostClassifier

        if model_library == 'lightgbm':
            custom_fit = ML_trainer.custom_fit_lgbm
            custom_predict = ML_trainer.custom_predict_lgbm
            custom_score = ML_trainer.custom_score_lgbm
            if problem_type == "Regression":
                model = lgbm.LGBMRegressor
            elif problem_type == "Classification":
                model = lgbm.LGBMClassifier

        if model_library == 'interpret':
            if problem_type == "Regression":
                model = ExplainableBoostingRegressor
            elif problem_type == "Classification":
                model = ExplainableBoostingClassifier

        if model is None:
            raise ValueError(f"Model class '{model_class}' is not recognized or supported")

        def ensure_single_element_at_end(lst, x):
            new_lst = [item for item in lst if item != x]
            new_lst.append(x)
            return new_lst

        axes_modelisation = ensure_single_element_at_end(axes_modelisation, list(test_data.columns)[-1])

        # Use the show_optimization_progress function
        best_params, best_cv_score, test_score, fitted_best_model, explanations = show_optimization_progress(
            probleme=problem_type,
            df=test_data[axes_modelisation],
            target_column=list(test_data.columns)[-1],
            model_class=model,
            param_distributions=hyperparameters,
            split_method=split_method,
            test_size=test_size if split_method == 'Random' else None,
            time_column='SURVENANCE' if split_method == 'Backtesting' else None,
            train_years=train_years if split_method == 'Backtesting' else None,
            test_year=test_year if split_method == 'Backtesting' else None,
            n_splits=n_splits,
            n_trials=n_trials,
            scoring=scoring,
            random_state=42,
            custom_fit=custom_fit,
            custom_predict=custom_predict,
            custom_score=custom_score,
            resampling_method=resampling_method,
            explain=False,
            annee_reference=annee_reference
        )

        # The results are now displayed within the show_optimization_progress function
        # You can add any additional visualizations or information here

        # Display model explanations if available
        if explanations:
            st.write("### Detailed Model Explanations")
            st.write("#### Global Feature Importance")
            if isinstance(fitted_best_model, (ExplainableBoostingRegressor, ExplainableBoostingClassifier)):
                st.plotly_chart(explanations['global'])
                st.write("#### Local Feature Importance (for a single prediction)")
                st.pyplot(explanations['local'])
            else:
                st.pyplot(explanations['global'])
                st.write("#### Local Feature Importance (for a single prediction)")
                plotly_chart(explanations['local'])

        st.download_button(
                "Download Model",
                data=pickle.dumps(fitted_best_model),
                file_name="model.pkl",
            )

    if st.button("Back to Test Data Creation"):
        st.session_state.page = 'create_test_data'
        # Clear the test data to free up memory
        if 'test_data' in st.session_state:
            del st.session_state.test_data
