from jointeurs import jointeur_cout, jointeur_frequence
import streamlit as st
import pandas as pd
import json
from pathlib import Path

class DataImporter:
    @st.cache_data
    def import_data(_self, file, sheet_name=None):
        if isinstance(file, str):
            file_path = Path(file)
            file_extension = file_path.suffix.lower()
        else:
            file_extension = Path(file.name).suffix.lower()
            file_path = file

        def convert_object_to_string(df):
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].astype(str)
            return df

        def check_df(df, source):
            if df.empty:
                raise ValueError(f"Imported DataFrame from {source} is empty")
            print(f"Imported data from {source}:")
            print(f"Shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            return df

        if file_extension == '.xlsx':
            df = pd.read_excel(file_path, engine='calamine', sheet_name=sheet_name)
            df = convert_object_to_string(df)
        elif file_extension == '.csv':
            df = pd.read_csv(file_path, dtype='object')
            df = convert_object_to_string(df)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

        df = check_df(df, str(file_path))
        return df

    @st.cache_data
    def get_test_data(_self, prod_files, sinistres_files, garantie, usage, axes_modelisation, use_cout, seuil_mat=None, seuil_crp=None, annee_reference=None, annees_cout=None, capital_assure=1000, inflation_rcmat=None, inflation_rccrp=None):
        # Process the data
        test_data = _self._process_raw_data(prod_files, sinistres_files, garantie, usage, axes_modelisation, seuil_mat, seuil_crp, use_cout, annee_reference, annees_cout, capital_assure, inflation_rcmat, inflation_rccrp)
        return test_data

    def _process_raw_data(self, prod_files, sinistres_files, garantie, usage, axes_modelisation, seuil_mat, seuil_crp, use_cout, annee_reference=None, annees_cout=None, capital_assure=1000, inflation_rcmat=None, inflation_rccrp=None):
        prod_dfs = {}
        for year, file_info in prod_files.items():
            prod_dfs[year] = self.import_data(file_info['file'], file_info['sheet_name'])
            if "ANNEE_POLICE" not in prod_dfs[year].columns:
                prod_dfs[year]["ANNEE_POLICE"] = int(year)
        combined_prod = pd.concat(prod_dfs.values(), ignore_index=True)
        combined_prod = combined_prod.rename(columns={'CNUMPOLIZZA': 'NUM_POLICE'})
        combined_prod.index = list(range(len(combined_prod)))

        sinistres_dfs = {}
        for year, file_info in sinistres_files.items():
            sinistres_dfs[year] = self.import_data(file_info['file'], file_info['sheet_name'])

        if use_cout:
            test_data = jointeur_cout(combined_prod, sinistres_dfs, garantie, usage, annee_reference, axes_modelisation, seuil_mat, seuil_crp, capital_assure, inflation_rcmat, inflation_rccrp)
        else:
            test_data = jointeur_frequence(sinistres_dfs, combined_prod, garantie, usage, axes_modelisation, seuil_mat, seuil_crp)

        return test_data