import pandas as pd
#---------------------------------------------------------------------------------------------------------
# JOINTEUR COUT
#----------------------------------------------------------------------------------------------------------

def creation_sin(sinistres, annee_reference, inflation_MAT, inflation_CRP):
    # Special processing for 2018
    if 2018 in sinistres.keys():
        sin2018 = sinistres[2018][['NUMERO_SINISTRE','NUM_POLICE','TYPE_SINISTRE','TYPE_DOSSIER','SURVENANCE',
                                   'REGLEMENT_INVENTAIRE','RESERVE_INVENTAIRE','GARANTIE']]
        sin2018['From'] = 'VUE_SINISTRE_2018'
        sin2018['GARANTIE_RC'] = (sin2018['GARANTIE'] == 'RC')*1
        sin2018['GARANTIE_BG'] = (sin2018['GARANTIE'] == 'BG')*1
        sin2018['GARANTIE_VOL&INC'] = (sin2018['GARANTIE'].isin(['VOL', 'INC']))*1
        sin2018.rename(columns={'RESERVE_INVENTAIRE': 'RES_RC', 'REGLEMENT_INVENTAIRE': 'REG_RC', 'GARANTIE': 'CHARGE_RC'}, inplace=True)

    # Similar processing for the rest of the years
    cols = ['NUMERO_SINISTRE', 'NUM_POLICE', 'TYPE_SINISTRE', 'TYPE_DOSSIER', 'SURVENANCE','TYPE_DOSSIER_AFIN','IDCDL_AFIN', 'DATE_OUVERTURE', 'CHARGE_RC', 'REG_RC', 'RES_RC', 'GARANTIE']
    sin_dfs = []
    for i in sinistres.keys():
        if i != 2018:
            sin_i = sinistres[i][cols].copy()
            sin_i['From'] = f'VUE_SINISTRE_{i}'
            sin_i['GARANTIE_RC'] = (sin_i['GARANTIE'] == 'RC')*1
            sin_i['GARANTIE_BG'] = (sin_i['GARANTIE'] == 'BG')*1
            sin_i['GARANTIE_VOL&INC'] = (sin_i['GARANTIE'].isin(['VOL', 'INC']))*1
            
            if inflation_MAT is not None or inflation_CRP is not None:
                sin_i['REG_RC'] *= (inflation_MAT.get(i, 1) if inflation_MAT else 1) * (sin_i['TYPE_SINISTRE'] == 'MAT') + \
                                   (inflation_CRP.get(i, 1) if inflation_CRP else 1) * (sin_i['TYPE_SINISTRE'] == 'CRP')
            sin_dfs.append(sin_i)

    sin = pd.concat(sin_dfs)
    
    # Create total regulation and reserve in the reference year
    sin['REG_TOTAL'] = sin.groupby('NUMERO_SINISTRE')['REG_RC'].transform('sum')
    sin[f'RES_{annee_reference}'] = sin['RES_RC'] * (sin['NUMERO_SINISTRE'].isin(sinistres[annee_reference]['NUMERO_SINISTRE']))
    
    # Create total adjusted charge
    sin['CHARGE_TOTAL_AJUSTE'] = sin['REG_TOTAL'] + sin[f'RES_{annee_reference}']
    
    # Apply filter to eliminate stock and non-occurrence cases
    sin = sin[(sin['TYPE_DOSSIER'] == 'EXERCICE') & (sin['SURVENANCE'].isin(sinistres.keys()))].drop(['CHARGE_RC', 'REG_RC', 'RES_RC'], axis=1)
    
    return sin

def jointeur_cout(prod, sinistres, garantie,usage,annee_reference, axes_modelisation,
                  seuil_mat, seuil_crp, capital_assure=1000, inflation_rcmat=None, inflation_rccrp=None):
    # Filter the prod DataFrame
    mask = (prod['TYPE_POLICE'] == "INDIVIDUEL") & \
           (((str(usage) == '210') & (prod['USAGEE'].astype(str) == '210')) | ((str(usage) != '210') & (prod['USAGEE'].astype(str) != '210')))
    filtered_prod = prod[mask]
    
    # Create the sinistre DataFrame
    sin = creation_sin(sinistres, annee_reference, inflation_rcmat, inflation_rccrp)
    
    if 'RC' in garantie:
        sin_subset = sin.loc[(sin['GARANTIE_RC'] == 1) & (sin['TYPE_SINISTRE'] == garantie[3:])].copy()
        threshold = seuil_mat if garantie == "RC MAT" and seuil_mat is not None else seuil_crp
        sin_subset = sin_subset[sin_subset['CHARGE_TOTAL_AJUSTE'].between(0.000001, threshold - 0.0000001)]
        
        sin_subset['Primary_Key'] = sin_subset['SURVENANCE'].astype(str) + '-' + sin_subset['NUM_POLICE'].astype(str)
        Y = 'YM' * (garantie == 'RC MAT') + 'YC' * (garantie == 'RC CRP')
        sin_subset[Y] = sin_subset.groupby('Primary_Key')['CHARGE_TOTAL_AJUSTE'].transform('mean')
        sin_subset.reset_index(drop=True, inplace=True)
    else:  # guarantees BG and VOL&INC
        sin_subset = sin[sin['GARANTIE_' + garantie] == 1].copy()
        sin_subset['Primary_Key'] = sin_subset['SURVENANCE'].astype(str) + '-' + sin_subset['NUM_POLICE'].astype(str)
        Y = 'Y_' + garantie
        sin_subset[Y] = capital_assure * (garantie == "BG")  # to be filled elsewhere by the venal value or the insured capital
    
    sin_subset = sin_subset[['Primary_Key', 'NUM_POLICE', 'SURVENANCE', 'TYPE_SINISTRE','TYPE_DOSSIER_AFIN','IDCDL_AFIN', Y]]

    filtered_prod = filtered_prod[['NUM_POLICE', 'ANNEE_POLICE'] + axes_modelisation + ['VALEUR_VENALE'] * (garantie == 'VOL&INC')]
    
    # Merge and filter
    merged_df = pd.merge(sin_subset,filtered_prod, on='NUM_POLICE', how='inner')
    merged_df = merged_df[merged_df['SURVENANCE'] == merged_df['ANNEE_POLICE']].drop_duplicates()
    if garantie == 'VOL&INC':
        merged_df['VALEUR_VENALE'] = merged_df['VALEUR_VENALE'].apply(lambda x: x / 1000 if x > 1000000 else x)
        merged_df[Y] = merged_df['VALEUR_VENALE']
    if 'GOUVERNORAT' in axes_modelisation:
        gouvernorat_mapping = {"LE KEF": "KEF", "MEHDIA": "MAHDIA", "MANOUBA": "MANNOUBA"}
        merged_df['GOUVERNORAT'] = merged_df['GOUVERNORAT'].replace(gouvernorat_mapping)

    l = list(merged_df.columns)
    l.remove(Y)
    l.remove('ANNEE_POLICE')
    l.append(Y)
    merged_df = merged_df[l]
    return merged_df

#-------------------------------------------------------------------------------------------------------------------------
# JOINTEUR FREQUENCE
#-------------------------------------------------------------------------------------------------------------

def bases_sinistres_par_garantie(dict_vues_sinistre, garantie, seuil_mat, seuil_crp):
    bases = []
    annees = list(dict_vues_sinistre.keys())

    if garantie in ['RC MAT', 'RC CRP']:
        for annee in annees:

            if annee == 2018:
                dict_vues_sinistre[annee] = dict_vues_sinistre[annee][dict_vues_sinistre[annee]['GARANTIE'] == 'RC']
                dict_vues_sinistre[annee]['CHARGE_RC'] = 1

            mask = (
                (dict_vues_sinistre[annee]['SURVENANCE'].isin(annees)) &
                (dict_vues_sinistre[annee]['TYPE_SINISTRE'] == garantie[3:]) &
                (dict_vues_sinistre[annee]['TYPE_DOSSIER'] == 'EXERCICE') &
                (dict_vues_sinistre[annee]['CHARGE_RC'] > 0)
            )
            
            if annee != 2018:
                mask &= (
                    ((dict_vues_sinistre[annee]['CHARGE_RC'] < seuil_mat) & (garantie[3:] == 'MAT')) |
                    ((dict_vues_sinistre[annee]['CHARGE_RC'] < seuil_crp) & (garantie[3:] == 'CRP'))
                )
            
            data = dict_vues_sinistre[annee].loc[mask].copy()
            
            bases.append(data)
    elif garantie == 'BG':
        for annee in annees:
            if annee == 2018:
                dict_vues_sinistre[annee] = dict_vues_sinistre[annee][dict_vues_sinistre[annee]['GARANTIE'] == garantie]
                dict_vues_sinistre[annee][f'CHARGE_{garantie}'] = 1

            mask = (
                (dict_vues_sinistre[annee]['SURVENANCE'].isin(annees)) &
                (dict_vues_sinistre[annee]['TYPE_SINISTRE'] == 'MAT') &
                (dict_vues_sinistre[annee]['TYPE_DOSSIER'] == 'EXERCICE') &
                (dict_vues_sinistre[annee][f'CHARGE_{garantie}'] > 0)
            )
            
            data = dict_vues_sinistre[annee].loc[mask].copy()
            
            bases.append(data)
    elif garantie == 'VOL&INC':
        for annee in annees:
            if annee == 2018:
                dict_vues_sinistre[annee] = dict_vues_sinistre[annee][dict_vues_sinistre[annee]['GARANTIE'].isin(['VOL','INC'])]
                dict_vues_sinistre[annee][f'CHARGE_{garantie}'] = 1

                mask = (
                (dict_vues_sinistre[annee]['SURVENANCE'].isin(annees)) &
                (dict_vues_sinistre[annee]['TYPE_SINISTRE'] == 'MAT') &
                (dict_vues_sinistre[annee]['TYPE_DOSSIER'] == 'EXERCICE') )

            else:
                mask = (
                        (dict_vues_sinistre[annee]['SURVENANCE'].isin(annees)) &
                        (dict_vues_sinistre[annee]['TYPE_SINISTRE'] == 'MAT') &
                        (dict_vues_sinistre[annee]['TYPE_DOSSIER'] == 'EXERCICE') &
                        ((dict_vues_sinistre[annee]['CHARGE_VOL'] > 0)|(dict_vues_sinistre[annee]['CHARGE_INC'] > 0))
                )

            
            data = dict_vues_sinistre[annee].loc[mask].copy()
            data[f'CHARGE_{garantie}'] = 1
            bases.append(data)

    gar = 'RC' if garantie in ['RC MAT', 'RC CRP'] else garantie
    sin_columns = ['NUMERO_SINISTRE', 'NUM_POLICE', 'SURVENANCE', f'CHARGE_{gar}']
    sin_garantie = pd.concat([data[sin_columns] for data in bases], ignore_index=True)
    return sin_garantie

def jointeur_frequence(dict_vues_sinistre, prod,garantie, usage,  axes_modelisation, seuil_mat, seuil_crp):
    sin_garantie = bases_sinistres_par_garantie(dict_vues_sinistre, garantie, seuil_mat, seuil_crp)

    mask_prod = (
        ((str(usage) == '210') & (prod['USAGEE'].astype(str) == '210')) | 
        ((str(usage) != '210') & (prod['USAGEE'].astype(str) != '210')) &
        (prod['ANNEE_POLICE'].isin(dict_vues_sinistre.keys())) &
        (prod['expo'].astype(float) > 0)
    )
    filtered_prod = prod.loc[mask_prod].copy()
    filtered_prod = filtered_prod[filtered_prod['expo'] != 0]
    filtered_prod['NUM_POLICE'] = filtered_prod['NUM_POLICE'].astype(str)
    filtered_prod['Primary_Key'] = filtered_prod['ANNEE_POLICE'].astype(str) + '-' + filtered_prod['NUM_POLICE']

    F_garantie = pd.merge(
        filtered_prod,
        sin_garantie,
        on='NUM_POLICE',
        how='left',
        indicator=True
    )

    F_garantie = F_garantie[
        (F_garantie['SURVENANCE'].isna()) |
        (F_garantie['SURVENANCE'] == F_garantie['ANNEE_POLICE'])
    ]
    gar = 'RC' if garantie in ['RC MAT', 'RC CRP'] else garantie
    F_garantie[f'CHARGE_{gar}'] = F_garantie[f'CHARGE_{gar}'].fillna(0)
    F_garantie[f'ind_{gar}'] = (F_garantie[f'CHARGE_{gar}'] > 0).astype(int)

    columns_to_keep = ['Primary_Key', 'expo'] + axes_modelisation + [f'ind_{gar}']
    F_garantie = F_garantie[columns_to_keep].drop_duplicates()

    F_garantie['nbr_sin_pk'] = F_garantie.groupby('Primary_Key')[f'ind_{gar}'].transform('sum')
    F_garantie[f'freq_{garantie}'] = round(F_garantie['nbr_sin_pk'] / F_garantie['expo'])

    gouvernorat_mapping = {
        "LE KEF": "KEF",
        "MEHDIA": "MAHDIA",
        "MANOUBA": "MANNOUBA"
    }
    F_garantie['GOUVERNORAT'] = F_garantie['GOUVERNORAT'].replace(gouvernorat_mapping)

    return F_garantie
