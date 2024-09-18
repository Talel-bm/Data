hyperparameters_data = {
"LinearRegression": {
        "fit_intercept": {
            "type": "checkbox",
            "args": ["fit_intercept", True],
            "help": "Détermine si l'ordonnée à l'origine doit être calculée pour ce modèle. Si défini à False, aucune ordonnée à l'origine ne sera utilisée dans les calculs (c'est-à-dire que les données sont supposées être centrées)."
        },
        "copy_X": {
            "type": "checkbox",
            "args": ["copy_X", True],
            "help": "Si True, X sera copié ; sinon, il pourrait être écrasé."
        },
        "n_jobs": {
            "type": "text_input",
            "args": ["n_jobs", '0'],
            "help": "Le nombre de tâches à utiliser pour le calcul. Cela n'apportera une accélération qu'en cas de problèmes suffisamment importants, c'est-à-dire si premièrement n_targets > 1 et deuxièmement X est sparse ou si positive est défini à True. None signifie 1 sauf dans un contexte joblib.parallel_backend. -1 signifie utiliser tous les processeurs."
        },
        "positive": {
            "type": "checkbox",
            "args": ["positive", False],
            "help": "Lorsque défini à True, force les coefficients à être positifs. Cette option n'est prise en charge que pour les tableaux denses."
        }
    },
"LogisticRegression": {
        "penalty": {
            "type": "multiselect",
            "args": ["penalty", ['l2', 'l1', 'elasticnet']],
            "help": "Spécifie la norme de la pénalité. 'l2' est la valeur par défaut. 'elasticnet' n'est pris en charge que par le solveur 'saga'."
        },
        "dual": {
            "type": "checkbox",
            "args": ["dual", False],
            "help": "Formulation duale ou primale. La formulation duale n'est implémentée que pour la pénalité l2 avec le solveur liblinear. Préférez dual=False lorsque n_samples > n_features."
        },
        "tol": {
            "type": "text_input",
            "args": ["tol", "0.0001"],
            "help": "Tolérance pour les critères d'arrêt."
        },
        "C": {
            "type": "text_input",
            "args": ["C", "1.0"],
            "help": "Inverse de la force de régularisation ; doit être un flottant positif. Comme dans les machines à vecteurs de support, des valeurs plus petites spécifient une régularisation plus forte."
        },
        "fit_intercept": {
            "type": "checkbox",
            "args": ["fit_intercept", True],
            "help": "Spécifie si une constante (aussi appelée biais ou ordonnée à l'origine) doit être ajoutée à la fonction de décision."
        },
        "intercept_scaling": {
            "type": "text_input",
            "args": ["intercept_scaling", "1.0"],
            "help": "Utile uniquement lorsque le solveur 'liblinear' est utilisé et que self.fit_intercept est défini à True."
        },
        "class_weight": {
            "type": "multiselect",
            "args": ["class_weight", ['balanced']],
            "help": "Poids associés aux classes. Si non spécifié, toutes les classes sont supposées avoir un poids de un. Le mode 'balanced' utilise les valeurs de y pour ajuster automatiquement les poids inversement proportionnels aux fréquences des classes."
        },
        "random_state": {
            "type": "text_input",
            "args": ["random_state","42"],
            "help": "Utilisé lorsque solver == 'sag', 'saga' ou 'liblinear' pour mélanger les données. Voir le Glossaire pour plus de détails."
        },
        "solver": {
            "type": "multiselect",
            "args": ["solver", ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']],
            "help": "Algorithme à utiliser dans le problème d'optimisation. Par défaut, c'est 'lbfgs'. Pour les petits ensembles de données, 'liblinear' est un bon choix, tandis que 'sag' et 'saga' sont plus rapides pour les grands ensembles."
        },
        "max_iter": {
            "type": "text_input",
            "args": ["max_iter", "100"],
            "help": "Nombre maximum d'itérations prises par les solveurs pour converger."
        },
        "multi_class": {
            "type": "multiselect",
            "args": ["multi_class", ['auto', 'ovr', 'multinomial']],
            "help": "Si l'option choisie est 'ovr', alors un problème binaire est ajusté pour chaque étiquette. Pour 'multinomial', la perte minimisée est la perte multinomiale ajustée sur l'ensemble de la distribution de probabilité. 'auto' sélectionne 'ovr' si les données sont binaires, ou si solver='liblinear', et sélectionne sinon 'multinomial'."
        },
        "verbose": {
            "type": "text_input",
            "args": ["verbose", "0"],
            "help": "Pour les solveurs liblinear et lbfgs, définissez verbose à un nombre positif pour la verbosité."
        },
        "warm_start": {
            "type": "checkbox",
            "args": ["warm_start", False],
            "help": "Lorsque défini à True, réutilise la solution de l'appel précédent à fit comme initialisation, sinon, efface simplement la solution précédente."
        },
        "n_jobs": {
            "type": "text_input",
            "args": ["n_jobs","0"],
            "help": "Nombre de cœurs CPU utilisés lors de la parallélisation sur les classes si multi_class='ovr'. Ce paramètre est ignoré lorsque le solveur est défini sur 'liblinear', que 'multi_class' soit spécifié ou non. None signifie 1 sauf dans un contexte joblib.parallel_backend. -1 signifie utiliser tous les processeurs."
        },
        "l1_ratio": {
            "type": "text_input",
            "args": ["l1_ratio","0"],
            "help": "Le paramètre de mélange Elastic-Net, avec 0 <= l1_ratio <= 1. Utilisé uniquement si penalty='elasticnet'. Définir l1_ratio=0 équivaut à utiliser penalty='l2', tandis que définir l1_ratio=1 équivaut à utiliser penalty='l1'."
        }
    },
"Ridge": {
        "alpha": {
            "type": "text_input",
            "args": ["alpha", "1.0"],
            "help": "Coefficient de régularisation. Plus alpha est grand, plus la régularisation est forte. Doit être un nombre positif."
        },
        "fit_intercept": {
            "type": "checkbox",
            "args": ["fit_intercept", True],
            "help": "Si True, le modèle calculera l'ordonnée à l'origine. Sinon, l'ordonnée à l'origine sera fixée à zéro."
        },
        "copy_X": {
            "type": "checkbox",
            "args": ["copy_X", True],
            "help": "Si True, X sera copié; sinon, il pourrait être écrasé."
        },
        "max_iter": {
            "type": "text_input",
            "args": ["max_iter","10"],
            "help": "Nombre maximal d'itérations pour les solveurs conjugate gradient."
        },
        "tol": {
            "type": "text_input",
            "args": ["tol", "0.0001"],
            "help": "Précision du problème de solution."
        },
        "solver": {
            "type": "multiselect",
            "args": ["solver", ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs']],
            "help": "Solveur à utiliser dans le problème de calcul. 'auto' choisit le solveur automatiquement en fonction du type de données."
        },
        "positive": {
            "type": "checkbox",
            "args": ["positive", False],
            "help": "Si True, force les coefficients à être positifs."
        },
        "random_state": {
            "type": "text_input",
            "args": ["random_state","42"],
            "help": "Contrôle la génération de nombres aléatoires pour la mélange des données lorsque solver='sag' ou 'saga'."
        }
    },
"Lasso": {
        "alpha": {
            "type": "text_input",
            "args": ["alpha", "1.0"],
            "help": "Constante qui multiplie le terme L1. Plus alpha est grand, plus la régularisation est forte. Doit être un nombre positif."
        },
        "fit_intercept": {
            "type": "checkbox",
            "args": ["fit_intercept", True],
            "help": "Si True, le modèle calculera l'ordonnée à l'origine. Sinon, les données sont supposées être déjà centrées."
        },
        "precompute": {
            "type": "checkbox",
            "args": ["precompute", False],
            "help": "Si True, pré-calcule les produits de Gram pour accélérer les calculs. Recommandé pour les ensembles de données de taille moyenne ou grande."
        },
        "copy_X": {
            "type": "checkbox",
            "args": ["copy_X", True],
            "help": "Si True, X sera copié; sinon, il pourrait être écrasé."
        },
        "max_iter": {
            "type": "text_input",
            "args": ["max_iter", "1000"],
            "help": "Nombre maximal d'itérations à effectuer."
        },
        "tol": {
            "type": "text_input",
            "args": ["tol", "0.0001"],
            "help": "Tolérance pour la convergence."
        },
        "warm_start": {
            "type": "checkbox",
            "args": ["warm_start", False],
            "help": "Si True, réutilise la solution de l'appel précédent comme initialisation."
        },
        "positive": {
            "type": "checkbox",
            "args": ["positive", False],
            "help": "Si True, force les coefficients à être positifs."
        },
        "random_state": {
            "type": "text_input",
            "args": ["random_state","42"],
            "help": "Contrôle la génération de nombres aléatoires pour la sélection des caractéristiques lorsque selection='random'."
        },
        "selection": {
            "type": "multiselect",
            "args": ["selection", ['cyclic', 'random']],
            "help": "Si 'cyclic', les caractéristiques sont parcourues de manière cyclique. Si 'random', une caractéristique aléatoire est mise à jour à chaque itération."
        }
    },
'ElasticNet': {
    'alpha': {
        'type': 'text_input',
        'args': ['alpha', '1.0'],
        'help': "Constante qui multiplie le terme de pénalité. Plus alpha est grand, plus la régularisation est forte. Doit être un nombre positif."
    },
    'l1_ratio': {
        'type': 'text_input',
        'args': ['l1_ratio', '0.5'],
        'help': "Le ratio de mélange ElasticNet (entre 0 et 1). Pour l1_ratio = 0, la pénalité est une pénalité L2. Pour l1_ratio = 1, c'est une pénalité L1. Pour 0 < l1_ratio < 1, c'est une combinaison des deux."
    },
    'fit_intercept': {
        'type': 'checkbox',
        'args': ['fit_intercept', True],
        'help': "Si True, le modèle calculera l'ordonnée à l'origine. Sinon, les données sont supposées être déjà centrées."
    },
    'precompute': {
        'type': 'checkbox',
        'args': ['precompute', False],
        'help': "Si True, pré-calcule les produits de Gram pour accélérer les calculs. Recommandé pour les ensembles de données de taille moyenne ou grande."
    },
    'max_iter': {
        'type': 'text_input',
        'args': ['max_iter', '1000'],
        'help': "Nombre maximal d'itérations à effectuer dans l'algorithme de résolution."
    },
    'copy_X': {
        'type': 'checkbox',
        'args': ['copy_X', True],
        'help': "Si True, X sera copié; sinon, il pourrait être écrasé."
    },
    'tol': {
        'type': 'text_input',
        'args': ['tol', '0.0001'],
        'help': "Tolérance pour la convergence de l'algorithme."
    },
    'warm_start': {
        'type': 'checkbox',
        'args': ['warm_start', True],
        'help': "Si True, réutilise la solution de l'appel précédent comme initialisation. Peut réduire le temps de calcul."
    },
    'positive': {
        'type': 'checkbox',
        'args': ['positive', True],
        'help': "Si True, force les coefficients à être positifs."
    },
    'random_state': {
        'type': 'text_input',
        'args': ['random_state', '42'],
        'help': "Contrôle la génération de nombres aléatoires pour la sélection des caractéristiques lorsque selection='random'. Utilisez un entier pour des résultats reproductibles."
    },
    'selection': {
        'type': 'multiselect',
        'args': ['selection', {'label': 'selection', 'options': ['cyclic', 'random']}],
        'help': "Si 'cyclic', les caractéristiques sont parcourues de manière cyclique. Si 'random', une caractéristique aléatoire est mise à jour à chaque itération. Le choix 'random' peut être plus rapide pour des ensembles de données avec un grand nombre de caractéristiques."
    }
},
'BayesianRidge': {
    'max_iter': {
        'type': 'text_input',
        'args': ['max_iter', '300'],
        'help': "Nombre maximal d'itérations. Si la convergence n'est pas atteinte, augmentez cette valeur."
    },
    'tol': {
        'type': 'text_input',
        'args': ['tol', '0.001'],
        'help': "Seuil de tolérance pour l'arrêt de l'algorithme. Si la valeur est trop petite, l'algorithme peut prendre plus de temps à converger."
    },
    'alpha_1': {
        'type': 'text_input',
        'args': ['alpha_1', '1e-06'],
        'help': "Paramètre de forme pour la distribution Gamma a priori sur alpha. Doit être supérieur à 0."
    },
    'alpha_2': {
        'type': 'text_input',
        'args': ['alpha_2', '1e-06'],
        'help': "Paramètre inverse d'échelle pour la distribution Gamma a priori sur alpha. Doit être supérieur à 0."
    },
    'lambda_1': {
        'type': 'text_input',
        'args': ['lambda_1', '1e-06'],
        'help': "Paramètre de forme pour la distribution Gamma a priori sur lambda. Doit être supérieur à 0."
    },
    'lambda_2': {
        'type': 'text_input',
        'args': ['lambda_2', '1e-06'],
        'help': "Paramètre inverse d'échelle pour la distribution Gamma a priori sur lambda. Doit être supérieur à 0."
    },
    'alpha_init': {
        'type': 'text_input',
        'args': ['alpha_init', '0'],
        'help': "Valeur initiale pour alpha. Si défini à None, alpha_init est défini à 1 / var(y)."
    },
    'lambda_init': {
        'type': 'text_input',
        'args': ['lambda_init','1'],
        'help': "Valeur initiale pour lambda. Si défini à None, lambda_init est défini à 1."
    },
    'compute_score': {
        'type': 'checkbox',
        'args': ['compute_score', True],
        'help': "Si True, calcule le score de log-marginal likelihood à chaque itération de l'optimisation."
    },
    'fit_intercept': {
        'type': 'checkbox',
        'args': ['fit_intercept', True],
        'help': "Si True, calcule l'ordonnée à l'origine pour ce modèle. Si False, l'ordonnée à l'origine est fixée à zéro."
    },
    'copy_X': {
        'type': 'checkbox',
        'args': ['copy_X', True],
        'help': "Si True, X sera copié; sinon, il pourrait être modifié."
    },
    'verbose': {
        'type': 'checkbox',
        'args': ['verbose', True],
        'help': "Si True, affiche des informations sur la progression de l'algorithme."
    }
},
'SGDRegressor': {
    'loss': {
        'type': 'multiselect',
        'args': ['loss', ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']],
        'help': "La fonction de perte à utiliser. 'squared_error' pour la régression linéaire, 'huber' pour la régression robuste, 'epsilon_insensitive' et 'squared_epsilon_insensitive' pour SVR."
    },
    'penalty': {
        'type': 'multiselect',
        'args': ['penalty',['l2', 'l1', 'elasticnet']],
        'help': "La pénalité (terme de régularisation) à utiliser. 'l2' est la régularisation de Ridge, 'l1' la régularisation Lasso, et 'elasticnet' combine les deux."
    },
    'alpha': {
        'type': 'text_input',
        'args': ['alpha', '0.0001'],
        'help': "Constante qui multiplie le terme de régularisation. Plus alpha est élevé, plus la régularisation est forte."
    },
    'l1_ratio': {
        'type': 'text_input',
        'args': ['l1_ratio', '0.15'],
        'help': "Le ratio Elastic Net mixing parameter. 0 <= l1_ratio <= 1. Utilisé seulement si penalty='elasticnet'."
    },
    'fit_intercept': {
        'type': 'checkbox',
        'args': ['fit_intercept', True],
        'help': "Si True, l'intercept est ajouté à la décision. Si False, l'intercept est fixé à 0."
    },
    'max_iter': {
        'type': 'text_input',
        'args': ['max_iter', '1000'],
        'help': "Le nombre maximum de passages sur les données d'entraînement (aka époques)."
    },
    'tol': {
        'type': 'text_input',
        'args': ['tol', '0.001'],
        'help': "Le critère d'arrêt. Si la perte ne diminue pas d'au moins tol pendant n_iter_no_change époques consécutives, l'entraînement s'arrête."
    },
    'shuffle': {
        'type': 'checkbox',
        'args': ['shuffle', True],
        'help': "Si True, mélange les données d'entraînement après chaque époque."
    },
    'verbose': {
        'type': 'text_input',
        'args': ['verbose', '0'],
        'help': "Le niveau de verbosité."
    },
    'epsilon': {
        'type': 'text_input',
        'args': ['epsilon', '0.1'],
        'help': "Paramètre epsilon dans la perte epsilon-insensitive ou le critère de Huber."
    },
    'random_state': {
        'type': 'text_input',
        'args': ['random_state', '42'],
        'help': "La graine utilisée par le générateur de nombres aléatoires."
    },
    'learning_rate': {
        'type': 'multiselect',
        'args': ['learning_rate', ['constant', 'optimal', 'invscaling', 'adaptive']],
        'help': "Le taux d'apprentissage: 'constant' utilise eta0, 'optimal' 1.0/(alpha * (t+t0)), 'invscaling' eta0 / pow(t, power_t), 'adaptive' diminue quand la perte n'est pas réduite."
    },
    'eta0': {
        'type': 'text_input',
        'args': ['eta0', '0.01'],
        'help': "Le taux d'apprentissage initial pour le taux d'apprentissage 'constant', 'invscaling' ou 'adaptive'."
    },
    'power_t': {
        'type': 'text_input',
        'args': ['power_t', '0.25'],
        'help': "L'exposant pour le taux d'apprentissage 'invscaling'."
    },
    'early_stopping': {
        'type': 'checkbox',
        'args': ['early_stopping', True],
        'help': "Si True, utilise l'arrêt précoce pour terminer l'entraînement lorsque la validation score n'est pas amélioré."
    },
    'validation_fraction': {
        'type': 'text_input',
        'args': ['validation_fraction', '0.1'],
        'help': "La proportion des données d'entraînement à mettre de côté comme ensemble de validation pour l'arrêt précoce."
    },
    'n_iter_no_change': {
        'type': 'text_input',
        'args': ['n_iter_no_change', '5'],
        'help': "Nombre d'itérations avec pas d'amélioration pour décider d'arrêter l'entraînement."
    },
    'warm_start': {
        'type': 'checkbox',
        'args': ['warm_start', True],
        'help': "Si True, réutilise la solution de l'appel précédent pour s'adapter comme initialisation."
    },
    'average': {
        'type': 'chzckbox',
        'args': ['average', False],
        'help': "Si True, calcule la solution moyennée."
    }
},
'PassiveAggressiveRegressor': {
    'C': {
        'type': 'text_input',
        'args': ['C', '1.0'],
        'help': "Paramètre de régularisation. Plus C est grand, moins il y a de régularisation."
    },
    'fit_intercept': {
        'type': 'checkbox',
        'args': ['fit_intercept', True],
        'help': "Si True, l'intercept est ajouté à la décision. Si False, l'intercept est fixé à 0."
    },
    'max_iter': {
        'type': 'text_input',
        'args': ['max_iter', '1000'],
        'help': "Le nombre maximum de passages sur les données d'entraînement (aka époques)."
    },
    'tol': {
        'type': 'text_input',
        'args': ['tol', '0.001'],
        'help': "Le critère d'arrêt. L'entraînement s'arrête quand (loss > previous_loss - tol)."
    },
    'early_stopping': {
        'type': 'checkbox',
        'args': ['early_stopping', True],
        'help': "Si True, utilise l'arrêt précoce pour terminer l'entraînement lorsque la validation score n'est pas amélioré."
    },
    'validation_fraction': {
        'type': 'text_input',
        'args': ['validation_fraction', '0.1'],
        'help': "La proportion des données d'entraînement à mettre de côté comme ensemble de validation pour l'arrêt précoce."
    },
    'n_iter_no_change': {
        'type': 'text_input',
        'args': ['n_iter_no_change', '5'],
        'help': "Nombre d'itérations avec pas d'amélioration pour décider d'arrêter l'entraînement."
    },
    'shuffle': {
        'type': 'checkbox',
        'args': ['shuffle', True],
        'help': "Si True, mélange les données d'entraînement après chaque époque."
    },
    'verbose': {
        'type': 'text_input',
        'args': ['verbose', '0'],
        'help': "Le niveau de verbosité."
    },
    'loss': {
        'type': 'multiselect',
        'args': ['loss', ['epsilon_insensitive', 'squared_epsilon_insensitive']],
        'help': "La fonction de perte à utiliser: 'epsilon_insensitive' est la perte linéaire, et 'squared_epsilon_insensitive' la perte quadratique."
    },
    'epsilon': {
        'type': 'text_input',
        'args': ['epsilon', '0.1'],
        'help': "Paramètre epsilon dans la perte epsilon-insensitive."
    },
    'random_state': {
        'type': 'text_input',
        'args': ['random_state', '42'],
        'help': "La graine utilisée par le générateur de nombres aléatoires."
    },
    'warm_start': {
        'type': 'checkbox',
        'args': ['warm_start', True],
        'help': "Si True, réutilise la solution de l'appel précédent pour s'adapter comme initialisation."
    },
    'average': {
        'type': 'checkbox',
        'args': ['average', False],
        'help': "Si True, calcule la solution moyennée."
    }
},
"RandomForestRegressor": {
    "n_estimators": {
        "type": "text_input",
        "args": ["n_estimators", '100'],
        "help": "Le nombre d'arbres dans la forêt."
    },
    "criterion": {
        "type": "multiselect",
        "args": ["criterion", ['squared_error', 'absolute_error', 'friedman_mse', 'poisson']],
        "help": "La fonction à mesurer pour l'évaluation de la qualité d'un split."
    },
    "max_depth": {
        "type": "text_input",
        "args": ["max_depth", '1'],
        "help": "La profondeur maximale de l'arbre. Si None, les nœuds sont expansés jusqu'à ce que toutes les feuilles soient pures ou que toutes les feuilles contiennent moins de min_samples_split échantillons."
    },
    "min_samples_split": {
        "type": "text_input",
        "args": ["min_samples_split", '2'],
        "help": "Le nombre minimum d'échantillons requis pour diviser un nœud interne."
    },
    "min_samples_leaf": {
        "type": "text_input",
        "args": ["min_samples_leaf", '1'],
        "help": "Le nombre minimum d'échantillons requis pour être à un nœud feuille."
    },
    "min_weight_fraction_leaf": {
        "type": "text_input",
        "args": ["min_weight_fraction_leaf", '0.0'],
        "help": "La fraction pondérée minimale de la somme totale des poids (de tous les échantillons d'entrée) requise pour être à un nœud feuille."
    },
    "max_features": {
        "type": "multiselect",
        "args": ["max_features", ['sqrt', 'log2']],
        "help": "Le nombre de caractéristiques à considérer lors de la recherche du meilleur split."
    },
    "max_leaf_nodes": {
        "type": "text_input",
        "args": ["max_leaf_nodes", '1'],
        "help": "Le nombre maximal de nœuds feuilles dans l'arbre."
    },
    "min_impurity_decrease": {
        "type": "text_input",
        "args": ["min_impurity_decrease", '0.0'],
        "help": "Une division d'un nœud ne sera réalisée que si elle diminue l'impureté du nœud d'au moins cette valeur."
    },
    "bootstrap": {
        "type": "checkbox",
        "args": ["bootstrap", True],
        "help": "Indique si les échantillons doivent être tirés avec remplacement."
    },
    "oob_score": {
        "type": "checkbox",
        "args": ["oob_score", True],
        "help": "Indique si la validation croisée sur les échantillons hors du sac doit être utilisée pour estimer la précision généralisée."
    },
    "n_jobs": {
        "type": "text_input",
        "args": ["n_jobs", '1'],
        "help": "Le nombre de jobs à utiliser pour la construction des arbres. -1 signifie utiliser tous les processeurs."
    },
    "random_state": {
        "type": "text_input",
        "args": ["random_state", '42'],
        "help": "Contrôle le hasard du bootstrapping des échantillons et l'ordre des caractéristiques à considérer lors de la recherche du meilleur split."
    },
    "verbose": {
        "type": "text_input",
        "args": ["verbose", '0'],
        "help": "Contrôle le niveau de verbosité pendant l'ajustement et la prédiction."
    },
    "warm_start": {
        "type": "checkbox",
        "args": ["warm_start", True],
        "help": "Quand il est mis à True, réutilise la solution de l'appel précédent pour ajuster et ajouter plus d'estimateurs à l'ensemble, sinon, commence à partir de zéro."
    },
    "ccp_alpha": {
        "type": "text_input",
        "args": ["ccp_alpha", '0.0'],
        "help": "Paramètre de complexité utilisé pour la taille minimale des coûts. Les sous-arbres les plus importants sont considérés à partir de cette valeur."
    },
    "max_samples": {
        "type": "text_input",
        "args": ["max_samples", '1'],
        "help": "Nombre ou fraction d'échantillons à tirer de X pour former chaque arbre de base."
    },
    "monotonic_cst": {
        "type": "text_input",
        "args": ["monotonic_cst", '1'],
        "help": "Constantes pour les contraintes de monotonie sur les caractéristiques."
    }
},
"ExtraTreesRegressor": {
    "n_estimators": {
        "type": "text_input",
        "args": ["n_estimators", '100'],
        "help": "Le nombre d'arbres dans la forêt."
    },
    "criterion": {
        "type": "multiselect",
        "args": ["criterion", ['squared_error', 'absolute_error', 'friedman_mse', 'poisson']],
        "help": "La fonction à mesurer pour l'évaluation de la qualité d'un split."
    },
    "max_depth": {
        "type": "text_input",
        "args": ["max_depth", '1'],
        "help": "La profondeur maximale de l'arbre. Si None, les nœuds sont expansés jusqu'à ce que toutes les feuilles soient pures ou que toutes les feuilles contiennent moins de min_samples_split échantillons."
    },
    "min_samples_split": {
        "type": "text_input",
        "args": ["min_samples_split", '2'],
        "help": "Le nombre minimum d'échantillons requis pour diviser un nœud interne."
    },
    "min_samples_leaf": {
        "type": "text_input",
        "args": ["min_samples_leaf", '1'],
        "help": "Le nombre minimum d'échantillons requis pour être à un nœud feuille."
    },
    "min_weight_fraction_leaf": {
        "type": "text_input",
        "args": ["min_weight_fraction_leaf", '0.0'],
        "help": "La fraction pondérée minimale de la somme totale des poids (de tous les échantillons d'entrée) requise pour être à un nœud feuille."
    },
    "max_features": {
        "type": "multiselect",
        "args": ["max_features", ['sqrt', 'log2']],
        "help": "Le nombre de caractéristiques à considérer lors de la recherche du meilleur split."
    },
    "max_leaf_nodes": {
        "type": "text_input",
        "args": ["max_leaf_nodes", '1'],
        "help": "Le nombre maximal de nœuds feuilles dans l'arbre."
    },
    "min_impurity_decrease": {
        "type": "text_input",
        "args": ["min_impurity_decrease", '0.0'],
        "help": "Une division d'un nœud ne sera réalisée que si elle diminue l'impureté du nœud d'au moins cette valeur."
    },
    "bootstrap": {
        "type": "checkbox",
        "args": ["bootstrap", True],
        "help": "Indique si les échantillons doivent être tirés avec remplacement."
    },
    "oob_score": {
        "type": "checkbox",
        "args": ["oob_score", True],
        "help": "Indique si la validation croisée sur les échantillons hors du sac doit être utilisée pour estimer la précision généralisée."
    },
    "n_jobs": {
        "type": "text_input",
        "args": ["n_jobs", '1'],
        "help": "Le nombre de jobs à utiliser pour la construction des arbres. -1 signifie utiliser tous les processeurs."
    },
    "random_state": {
        "type": "text_input",
        "args": ["random_state", '42'],
        "help": "Contrôle le hasard du bootstrapping des échantillons et l'ordre des caractéristiques à considérer lors de la recherche du meilleur split."
    },
    "verbose": {
        "type": "text_input",
        "args": ["verbose", '0'],
        "help": "Contrôle le niveau de verbosité pendant l'ajustement et la prédiction."
    },
    "warm_start": {
        "type": "checkbox",
        "args": ["warm_start", True],
        "help": "Quand il est mis à True, réutilise la solution de l'appel précédent pour ajuster et ajouter plus d'estimateurs à l'ensemble, sinon, commence à partir de zéro."
    },
    "ccp_alpha": {
        "type": "text_input",
        "args": ["ccp_alpha", '0.0'],
        "help": "Paramètre de complexité utilisé pour la taille minimale des coûts. Les sous-arbres les plus importants sont considérés à partir de cette valeur."
    },
    "max_samples": {
        "type": "text_input",
        "args": ["max_samples", '1'],
        "help": "Nombre ou fraction d'échantillons à tirer de X pour former chaque arbre de base."
    },
    "monotonic_cst": {
        "type": "text_input",
        "args": ["monotonic_cst", '1'],
        "help": "Constantes pour les contraintes de monotonie sur les caractéristiques."
    }
},
"GradientBoostingRegressor": {
    "loss": {
        "type": "multiselect",
        "args": ["loss", ['squared_error', 'absolute_error', 'huber', 'quantile']],
        "help": "Fonction de perte à optimiser dans chaque étape du boosting."
    },
    "learning_rate": {
        "type": "text_input",
        "args": ["learning_rate", '0'],
        "help": "Taux d'apprentissage réducteur appliqué à chaque étape de boosting. Valeur plus basse signifie un modèle plus robuste mais plus lent."
    },
    "n_estimators": {
        "type": "text_input",
        "args": ["n_estimators", '0'],
        "help": "Le nombre total d'arbres à ajuster."
    },
    "subsample": {
        "type": "text_input",
        "args": ["subsample", '0'],
        "help": "La fraction d'échantillons utilisée pour ajuster les arbres individuels. Réduction des valeurs peut permettre de réduire le surajustement."
    },
    "criterion": {
        "type": "multiselect",
        "args": ["criterion", ['friedman_mse', 'squared_error']],
        "help": "Fonction pour mesurer la qualité d'un split. Par défaut, Friedman MSE qui est plus efficace pour le boosting."
    },
    "min_samples_split": {
        "type": "text_input",
        "args": ["min_samples_split", '0'],
        "help": "Nombre minimum d'échantillons requis pour diviser un nœud."
    },
    "min_samples_leaf": {
        "type": "text_input",
        "args": ["min_samples_leaf", '0'],
        "help": "Nombre minimum d'échantillons requis pour être à un nœud feuille."
    },
    "min_weight_fraction_leaf": {
        "type": "text_input",
        "args": ["min_weight_fraction_leaf", '0'],
        "help": "Fraction pondérée minimale de la somme totale des poids (de tous les échantillons) requise pour être à un nœud feuille."
    },
    "max_depth": {
        "type": "text_input",
        "args": ["max_depth", '0'],
        "help": "Profondeur maximale des arbres individuels. Profondeur élevée peut entraîner un surajustement."
    },
    "min_impurity_decrease": {
        "type": "text_input",
        "args": ["min_impurity_decrease", '0'],
        "help": "Une division d'un nœud ne sera réalisée que si elle diminue l'impureté d'au moins cette valeur."
    },
    "max_features": {
        "type": "multiselect",
        "args": ["max_features", ['sqrt', 'log2']],
        "help": "Nombre de caractéristiques à considérer pour trouver le meilleur split."
    },
    "alpha": {
        "type": "text_input",
        "args": ["alpha", '0'],
        "help": "Quantile utilisé pour la perte de quantile. Seulement applicable si loss='quantile'."
    },
    "max_leaf_nodes": {
        "type": "text_input",
        "args": ["max_leaf_nodes", '0'],
        "help": "Nombre maximum de nœuds feuilles dans chaque arbre."
    },
    "validation_fraction": {
        "type": "text_input",
        "args": ["validation_fraction", '0'],
        "help": "Fraction des données d'apprentissage à réserver pour la validation précoce du boosting."
    },
    "n_iter_no_change": {
        "type": "text_input",
        "args": ["n_iter_no_change", '0'],
        "help": "Nombre de itérations sans amélioration pour arrêter la formation tôt."
    },
    "tol": {
        "type": "text_input",
        "args": ["tol", '0'],
        "help": "Seuil pour l'arrêt précoce du boosting si l'amélioration est inférieure à ce seuil."
    },
    "ccp_alpha": {
        "type": "text_input",
        "args": ["ccp_alpha", '0'],
        "help": "Paramètre de complexité utilisé pour la taille minimale des coûts. Les sous-arbres les plus importants sont considérés à partir de cette valeur."
    }
},
"RandomForestClassifier": {
    "n_estimators": {
        "type": "text_input",
        "args": ["n_estimators", '0'],
        "help": "Nombre d'arbres dans la forêt."
    },
    "criterion": {
        "type": "multiselect",
        "args": ["criterion", ['gini', 'entropy', 'log_loss']],
        "help": "Fonction pour mesurer la qualité des splits. Gini pour l'impureté de Gini, Entropie pour l'information gain, et Log Loss pour la log-vraisemblance."
    },
    "max_depth": {
        "type": "text_input",
        "args": ["max_depth", '0'],
        "help": "Profondeur maximale de l'arbre. None signifie que les nœuds sont expansés jusqu'à ce qu'ils contiennent moins de min_samples_split échantillons."
    },
    "min_samples_split": {
        "type": "text_input",
        "args": ["min_samples_split", '0'],
        "help": "Nombre minimum d'échantillons requis pour diviser un nœud interne."
    },
    "min_samples_leaf": {
        "type": "text_input",
        "args": ["min_samples_leaf", '0'],
        "help": "Nombre minimum d'échantillons requis pour être à un nœud feuille."
    },
    "min_weight_fraction_leaf": {
        "type": "text_input",
        "args": ["min_weight_fraction_leaf", '0'],
        "help": "Fraction pondérée minimale de la somme totale des poids (de tous les échantillons) requise pour être à un nœud feuille."
    },
    "max_features": {
        "type": "multiselect",
        "args": ["max_features", ['sqrt', 'log2']],
        "help": "Nombre de caractéristiques à considérer lors de la recherche du meilleur split."
    },
    "max_leaf_nodes": {
        "type": "text_input",
        "args": ["max_leaf_nodes", '0'],
        "help": "Nombre maximum de nœuds feuilles dans l'arbre."
    },
    "min_impurity_decrease": {
        "type": "text_input",
        "args": ["min_impurity_decrease", '0'],
        "help": "Une division d'un nœud ne sera réalisée que si elle diminue l'impureté d'au moins cette valeur."
    },
    "bootstrap": {
        "type": "checkbox",
        "args": ["bootstrap", False],
        "help": "Indique si les échantillons doivent être tirés avec remplacement."
    },
    "oob_score": {
        "type": "checkbox",
        "args": ["oob_score", False],
        "help": "Indique si la validation croisée sur les échantillons hors du sac doit être utilisée pour estimer la précision généralisée."
    },
    "n_jobs": {
        "type": "text_input",
        "args": ["n_jobs", '0'],
        "help": "Nombre de jobs à utiliser pour la construction des arbres. -1 signifie utiliser tous les processeurs."
    },
    "verbose": {
        "type": "text_input",
        "args": ["verbose", '0'],
        "help": "Contrôle le niveau de verbosité pendant l'ajustement et la prédiction."
    },
    "warm_start": {
        "type": "checkbox",
        "args": ["warm_start", False],
        "help": "Quand il est mis à True, réutilise la solution de l'appel précédent pour ajuster et ajouter plus d'estimateurs à l'ensemble, sinon, commence à partir de zéro."
    },
    "class_weight": {
        "type": "multiselect",
        "args": ["class_weight", ['balanced', 'balanced_subsample']],
        "help": "Poids associés aux classes dans la fonction de coût. Utilisez 'balanced' pour ajuster automatiquement les poids en fonction de la fréquence des classes."
    },
    "ccp_alpha": {
        "type": "text_input",
        "args": ["ccp_alpha", '0'],
        "help": "Paramètre de complexité utilisé pour la taille minimale des coûts. Les sous-arbres les plus importants sont considérés à partir de cette valeur."
    },
    "max_samples": {
        "type": "text_input",
        "args": ["max_samples", '0'],
        "help": "Nombre ou fraction d'échantillons à tirer de X pour former chaque arbre de base."
    },
    "monotonic_cst": {
        "type": "text_input",
        "args": ["monotonic_cst", '0'],
        "help": "Constantes pour les contraintes de monotonie sur les caractéristiques."
    }
},
"ExtraTreesClassifier": {
    "n_estimators": {
        "type": "text_input",
        "args": ["n_estimators", '0'],
        "help": "Nombre d'arbres dans la forêt extra-trees."
    },
    "criterion": {
        "type": "multiselect",
        "args": ["criterion", ['gini', 'entropy', 'log_loss']],
        "help": "Fonction pour mesurer la qualité des splits. Gini pour l'impureté de Gini, Entropie pour l'information gain, et Log Loss pour la log-vraisemblance."
    },
    "max_depth": {
        "type": "text_input",
        "args": ["max_depth", '0'],
        "help": "Profondeur maximale de l'arbre. None signifie que les nœuds sont expansés jusqu'à ce qu'ils contiennent moins de min_samples_split échantillons."
    },
    "min_samples_split": {
        "type": "text_input",
        "args": ["min_samples_split", '0'],
        "help": "Nombre minimum d'échantillons requis pour diviser un nœud interne."
    },
    "min_samples_leaf": {
        "type": "text_input",
        "args": ["min_samples_leaf", '0'],
        "help": "Nombre minimum d'échantillons requis pour être à un nœud feuille."
    },
    "min_weight_fraction_leaf": {
        "type": "text_input",
        "args": ["min_weight_fraction_leaf", '0'],
        "help": "Fraction pondérée minimale de la somme totale des poids (de tous les échantillons) requise pour être à un nœud feuille."
    },
    "max_features": {
        "type": "multiselect",
        "args": ["max_features", ['sqrt', 'log2']],
        "help": "Nombre de caractéristiques à considérer lors de la recherche du meilleur split."
    },
    "max_leaf_nodes": {
        "type": "text_input",
        "args": ["max_leaf_nodes", '0'],
        "help": "Nombre maximum de nœuds feuilles dans l'arbre."
    },
    "min_impurity_decrease": {
        "type": "text_input",
        "args": ["min_impurity_decrease", '0'],
        "help": "Une division d'un nœud ne sera réalisée que si elle diminue l'impureté d'au moins cette valeur."
    },
    "bootstrap": {
        "type": "checkbox",
        "args": ["bootstrap", False],
        "help": "Indique si les échantillons doivent être tirés avec remplacement."
    },
    "oob_score": {
        "type": "checkbox",
        "args": ["oob_score", False],
        "help": "Indique si la validation croisée sur les échantillons hors du sac doit être utilisée pour estimer la précision généralisée."
    },
    "n_jobs": {
        "type": "text_input",
        "args": ["n_jobs", '0'],
        "help": "Nombre de jobs à utiliser pour la construction des arbres. -1 signifie utiliser tous les processeurs."
    },
    "verbose": {
        "type": "text_input",
        "args": ["verbose", '0'],
        "help": "Contrôle le niveau de verbosité pendant l'ajustement et la prédiction."
    },
    "warm_start": {
        "type": "checkbox",
        "args": ["warm_start", False],
        "help": "Quand il est mis à True, réutilise la solution de l'appel précédent pour ajuster et ajouter plus d'estimateurs à l'ensemble, sinon, commence à partir de zéro."
    },
    "class_weight": {
        "type": "multiselect",
        "args": ["class_weight", ['balanced', 'balanced_subsample']],
        "help": "Poids associés aux classes dans la fonction de coût. Utilisez 'balanced' pour ajuster automatiquement les poids en fonction de la fréquence des classes."
    },
    "ccp_alpha": {
        "type": "text_input",
        "args": ["ccp_alpha", '0'],
        "help": "Paramètre de complexité utilisé pour la taille minimale des coûts. Les sous-arbres les plus importants sont considérés à partir de cette valeur."
    },
    "max_samples": {
        "type": "text_input",
        "args": ["max_samples", '0'],
        "help": "Nombre ou fraction d'échantillons à tirer de X pour former chaque arbre de base."
    },
    "monotonic_cst": {
        "type": "text_input",
        "args": ["monotonic_cst", '0'],
        "help": "Constantes pour les contraintes de monotonie sur les caractéristiques."
    }
},
"GradientBoostingClassifier": {
    "loss": {
        "type": "multiselect",
        "args": ["loss", ['log_loss', 'exponential']],
        "help": "Fonction de perte à optimiser à chaque étape du boosting. 'log_loss' pour la perte logistique et 'exponential' pour la perte exponentielle."
    },
    "learning_rate": {
        "type": "text_input",
        "args": ["learning_rate", '0'],
        "help": "Taux d'apprentissage réducteur appliqué à chaque étape de boosting. Une valeur plus basse signifie un modèle plus robuste mais plus lent."
    },
    "n_estimators": {
        "type": "text_input",
        "args": ["n_estimators", '0'],
        "help": "Nombre total d'arbres à ajuster."
    },
    "subsample": {
        "type": "text_input",
        "args": ["subsample", '0'],
        "help": "Fraction des échantillons utilisée pour ajuster les arbres individuels. Réduction des valeurs peut permettre de réduire le surajustement."
    },
    "criterion": {
        "type": "multiselect",
        "args": ["criterion", ['friedman_mse', 'squared_error']],
        "help": "Fonction pour mesurer la qualité d'un split. 'friedman_mse' est par défaut plus efficace pour le boosting."
    },
    "min_samples_split": {
        "type": "text_input",
        "args": ["min_samples_split", '0'],
        "help": "Nombre minimum d'échantillons requis pour diviser un nœud."
    },
    "min_samples_leaf": {
        "type": "text_input",
        "args": ["min_samples_leaf", '0'],
        "help": "Nombre minimum d'échantillons requis pour être à un nœud feuille."
    },
    "min_weight_fraction_leaf": {
        "type": "text_input",
        "args": ["min_weight_fraction_leaf", '0'],
        "help": "Fraction pondérée minimale de la somme totale des poids (de tous les échantillons) requise pour être à un nœud feuille."
    },
    "max_depth": {
        "type": "text_input",
        "args": ["max_depth", '0'],
        "help": "Profondeur maximale des arbres individuels. Une profondeur élevée peut entraîner un surajustement."
    },
    "min_impurity_decrease": {
        "type": "text_input",
        "args": ["min_impurity_decrease", '0'],
        "help": "Une division d'un nœud ne sera réalisée que si elle diminue l'impureté d'au moins cette valeur."
    },
    "max_features": {
        "type": "multiselect",
        "args": ["max_features", ['sqrt', 'log2']],
        "help": "Nombre de caractéristiques à considérer lors de la recherche du meilleur split."
    },
    "verbose": {
        "type": "text_input",
        "args": ["verbose", '0'],
        "help": "Contrôle le niveau de verbosité pendant l'ajustement et la prédiction."
    },
    "max_leaf_nodes": {
        "type": "text_input",
        "args": ["max_leaf_nodes", '0'],
        "help": "Nombre maximum de nœuds feuilles dans chaque arbre."
    },
    "warm_start": {
        "type": "checkbox",
        "args": ["warm_start", False],
        "help": "Quand il est mis à True, réutilise la solution de l'appel précédent pour ajuster et ajouter plus d'estimateurs à l'ensemble, sinon, commence à partir de zéro."
    },
    "validation_fraction": {
        "type": "text_input",
        "args": ["validation_fraction", '0'],
        "help": "Fraction des données d'apprentissage à réserver pour la validation précoce du boosting."
    },
    "n_iter_no_change": {
        "type": "text_input",
        "args": ["n_iter_no_change", '0'],
        "help": "Nombre d'itérations sans amélioration pour arrêter la formation tôt."
    },
    "tol": {
        "type": "text_input",
        "args": ["tol", '0'],
        "help": "Seuil pour l'arrêt précoce du boosting si l'amélioration est inférieure à ce seuil."
    },
    "ccp_alpha": {
        "type": "text_input",
        "args": ["ccp_alpha", '0'],
        "help": "Paramètre de complexité utilisé pour la taille minimale des coûts. Les sous-arbres les plus importants sont considérés à partir de cette valeur."
    }
},
"SVR": {
    "kernel": {
        "type": "multiselect",
        "args": ["kernel", ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']],
        "help": "Type de fonction noyau à utiliser pour le modèle SVR. 'linear' est un noyau linéaire, 'poly' est un noyau polynomial, 'rbf' est un noyau gaussien, 'sigmoid' est un noyau sigmoïde, et 'precomputed' signifie que le noyau est pré-calculé."
    },
    "degree": {
        "type": "text_input",
        "args": ["degree", '0'],
        "help": "Degré du noyau polynomial ('poly'). Ignoré par les autres noyaux."
    },
    "gamma": {
        "type": "multiselect",
        "args": ["gamma", ['scale', 'auto']],
        "help": "Coefficient gamma pour le noyau RBF, 'poly', et 'sigmoid'. 'scale' est par défaut 1 / (n_features * X.var()), 'auto' est 1 / n_features."
    },
    "coef0": {
        "type": "text_input",
        "args": ["coef0", '0'],
        "help": "Terme indépendant dans le noyau polynomial et sigmoïde. Ignoré par les autres noyaux."
    },
    "tol": {
        "type": "text_input",
        "args": ["tol", '0'],
        "help": "Tolérance pour le critère d'arrêt."
    },
    "C": {
        "type": "text_input",
        "args": ["C", '0'],
        "help": "Paramètre de régularisation. La force de la régularisation est inversement proportionnelle à C. Doit être strictement positif."
    },
    "epsilon": {
        "type": "text_input",
        "args": ["epsilon", '0'],
        "help": "Précision à laquelle les points doivent se trouver pour ne pas contribuer à la perte dans l'epsilon-Tube."
    },
    "shrinking": {
        "type": "checkbox",
        "args": ["shrinking", False],
        "help": "Si True, utilise l'heuristique de shrinking."
    },
    "cache_size": {
        "type": "text_input",
        "args": ["cache_size", '0'],
        "help": "Taille du cache (en Mo) pour le noyau."
    },
    "verbose": {
        "type": "checkbox",
        "args": ["verbose", False],
        "help": "Active la sortie de la console détaillée pendant l'ajustement. Uniquement activé si tolérance et max_iter sont spécifiés."
    },
    "max_iter": {
        "type": "text_input",
        "args": ["max_iter", '0'],
        "help": "Nombre maximum d'itérations. -1 pour pas de limite."
    }
},
"KNeighborsClassifier": {
    "n_neighbors": {
        "type": "text_input",
        "args": ["n_neighbors", '0'],
        "help": "Nombre de voisins à utiliser par défaut pour les requêtes de voisinage."
    },
    "weights": {
        "type": "multiselect",
        "args": ["weights", ['uniform', 'distance']],
        "help": "'uniform' attribue un poids uniforme à tous les voisins. 'distance' pondère les voisins par l'inverse de leur distance."
    },
    "algorithm": {
        "type": "multiselect",
        "args": ["algorithm", ['auto', 'ball_tree', 'kd_tree', 'brute']],
        "help": "Algorithme utilisé pour calculer les voisins. 'auto' sélectionne automatiquement l'algorithme optimal en fonction des valeurs des paramètres."
    },
    "leaf_size": {
        "type": "text_input",
        "args": ["leaf_size", '0'],
        "help": "Taille de la feuille passée à BallTree ou KDTree. Influence la vitesse de construction et la mémoire requise pour les arbres."
    },
    "p": {
        "type": "text_input",
        "args": ["p", '0'],
        "help": "Paramètre de puissance pour la distance de Minkowski. p=1 est équivalent à la distance de Manhattan, et p=2 à la distance euclidienne."
    },
    "metric": {
        "type": "text_input",
        "args": ["metric", '0'],
        "help": "Métrique utilisée pour la distance entre les points. Par défaut, c'est la distance euclidienne."
    },
    "metric_params": {
        "type": "text_area",
        "args": ["metric_params", '{}'],
        "help": "Paramètres supplémentaires pour la métrique spécifiée. Aucun paramètre supplémentaire n'est utilisé par défaut."
    },
    "n_jobs": {
        "type": "text_input",
        "args": ["n_jobs", '0'],
        "help": "Nombre de jobs parallèles pour la recherche des voisins. -1 signifie utiliser tous les processeurs disponibles."
    }
},
"KNeighborsRegressor": {
    "n_neighbors": {
        "type": "text_input",
        "args": ["n_neighbors", '0'],
        "help": "Nombre de voisins à utiliser par défaut pour les requêtes de voisinage."
    },
    "weights": {
        "type": "multiselect",
        "args": ["weights", ['uniform', 'distance']],
        "help": "'uniform' attribue un poids uniforme à tous les voisins. 'distance' pondère les voisins par l'inverse de leur distance."
    },
    "algorithm": {
        "type": "multiselect",
        "args": ["algorithm", ['auto', 'ball_tree', 'kd_tree', 'brute']],
        "help": "Algorithme utilisé pour calculer les voisins. 'auto' sélectionne automatiquement l'algorithme optimal en fonction des valeurs des paramètres."
    },
    "leaf_size": {
        "type": "text_input",
        "args": ["leaf_size", '0'],
        "help": "Taille de la feuille passée à BallTree ou KDTree. Influence la vitesse de construction et la mémoire requise pour les arbres."
    },
    "p": {
        "type": "text_input",
        "args": ["p", '0'],
        "help": "Paramètre de puissance pour la distance de Minkowski. p=1 est équivalent à la distance de Manhattan, et p=2 à la distance euclidienne."
    },
    "metric": {
        "type": "text_input",
        "args": ["metric", '0'],
        "help": "Métrique utilisée pour la distance entre les points. Par défaut, c'est la distance euclidienne."
    },
    "metric_params": {
        "type": "text_area",
        "args": ["metric_params", '{}'],
        "help": "Paramètres supplémentaires pour la métrique spécifiée. Aucun paramètre supplémentaire n'est utilisé par défaut."
    },
    "n_jobs": {
        "type": "text_input",
        "args": ["n_jobs", '0'],
        "help": "Nombre de jobs parallèles pour la recherche des voisins. -1 signifie utiliser tous les processeurs disponibles."
    }
},
"AdaBoostClassifier": {
    "n_estimators": {
        "type": "text_input",
        "args": ["n_estimators", '0'],
        "help": "Nombre maximum d'estimateurs faibles (arbres de décision) à entraîner."
    },
    "learning_rate": {
        "type": "text_input",
        "args": ["learning_rate", '0'],
        "help": "Poids appliqué à chaque classificateur faible au cours de l'apprentissage. Plus il est faible, plus l'algorithme est conservateur."
    },
    "algorithm": {
        "type": "multiselect",
        "args": ["algorithm", ['SAMME', 'SAMME.R']],
        "help": "'SAMME' pour le boost d'algorithmes discrets. 'SAMME.R' utilise les probabilités et est plus rapide."
    },
    "random_state": {
        "type": "text_input",
        "args": ["random_state", '42'],
        "help": "Contrôle la randomisation de l'algorithme pour la reproductibilité."
    }
},
"AdaBoostRegressor": {
    "n_estimators": {
        "type": "text_input",
        "args": ["n_estimators", '0'],
        "help": "Nombre maximum d'estimateurs faibles (arbres de décision) à entraîner."
    },
    "learning_rate": {
        "type": "text_input",
        "args": ["learning_rate", '0'],
        "help": "Poids appliqué à chaque régresseur faible au cours de l'apprentissage. Plus il est faible, plus l'algorithme est conservateur."
    },
    "loss": {
        "type": "multiselect",
        "args": ["loss", ['linear', 'square', 'exponential']],
        "help": "Fonction de perte à minimiser lors du réajustement des erreurs. 'linear' est la perte linéaire, 'square' est la perte quadratique, et 'exponential' pèse les erreurs exponentiellement."
    },
    "random_state": {
        "type": "text_input",
        "args": ["random_state", '42'],
        "help": "Contrôle la randomisation de l'algorithme pour la reproductibilité."
    }
},
"MLPClassifier": {
    "hidden_layer_sizes": {
        "type": "text_input",
        "args": ["hidden_layer_sizes", '1'],
        "help": "Nombre de neurones dans chaque couche cachée. Par exemple, (100,) pour une seule couche de 100 neurones."
    },
    "activation": {
        "type": "multiselect",
        "args": ["activation", ['identity', 'logistic', 'tanh', 'relu']],
        "help": "Fonction d'activation pour les neurones de la couche cachée."
    },
    "solver": {
        "type": "multiselect",
        "args": ["solver", ['lbfgs', 'sgd', 'adam']],
        "help": "Algorithme utilisé pour optimiser les poids des neurones."
    },
    "alpha": {
        "type": "text_input",
        "args": ["alpha", '0'],
        "help": "Paramètre de régularisation L2 pour éviter le surapprentissage."
    },
    "batch_size": {
        "type": "text_input",
        "args": ["batch_size", '0'],
        "help": "Nombre de points de données dans chaque lot pour le calcul du gradient."
    },
    "learning_rate": {
        "type": "multiselect",
        "args": ["learning_rate", ['constant', 'invscaling', 'adaptive']],
        "help": "Stratégie pour ajuster le taux d'apprentissage au fil des itérations."
    },
    "learning_rate_init": {
        "type": "text_input",
        "args": ["learning_rate_init", '0'],
        "help": "Taux d'apprentissage initial pour l'optimisation du poids."
    },
    "power_t": {
        "type": "text_input",
        "args": ["power_t", '0'],
        "help": "Exposant pour la mise à l'échelle du taux d'apprentissage lors de l'utilisation de l'invscaling."
    },
    "max_iter": {
        "type": "text_input",
        "args": ["max_iter", '0'],
        "help": "Nombre maximum d'itérations pour l'optimisation."
    },
    "shuffle": {
        "type": "checkbox",
        "args": ["shuffle", False],
        "help": "Indique si les données doivent être mélangées à chaque itération."
    },
    "random_state": {
        "type": "text_input",
        "args": ["random_state", '42'],
        "help": "Contrôle la randomisation pour la reproductibilité."
    },
    "tol": {
        "type": "text_input",
        "args": ["tol", '0'],
        "help": "Seuil pour l'arrêt précoce lorsque la perte cesse de diminuer."
    },
    "verbose": {
        "type": "checkbox",
        "args": ["verbose", False],
        "help": "Active ou désactive la sortie détaillée pendant l'entraînement."
    },
    "warm_start": {
        "type": "checkbox",
        "args": ["warm_start", False],
        "help": "Réutilise la solution de l'entraînement précédent pour l'initialisation."
    },
    "momentum": {
        "type": "text_input",
        "args": ["momentum", '0'],
        "help": "Terme de momentum utilisé lors de l'optimisation par descente de gradient stochastique."
    },
    "nesterovs_momentum": {
        "type": "checkbox",
        "args": ["nesterovs_momentum", False],
        "help": "Indique si l'impulsion de Nesterov doit être utilisée."
    },
    "early_stopping": {
        "type": "checkbox",
        "args": ["early_stopping", False],
        "help": "Arrête l'entraînement tôt si la performance cesse de s'améliorer sur l'ensemble de validation."
    },
    "validation_fraction": {
        "type": "text_input",
        "args": ["validation_fraction", '0'],
        "help": "Fraction des données d'entraînement utilisée comme ensemble de validation pour l'arrêt précoce."
    },
    "beta_1": {
        "type": "text_input",
        "args": ["beta_1", '0'],
        "help": "Paramètre pour l'estimation du premier moment pour l'algorithme Adam."
    },
    "beta_2": {
        "type": "text_input",
        "args": ["beta_2", '0'],
        "help": "Paramètre pour l'estimation du second moment pour l'algorithme Adam."
    },
    "epsilon": {
        "type": "text_input",
        "args": ["epsilon", '0'],
        "help": "Valeur pour stabiliser la division lors de la mise à jour des poids."
    },
    "n_iter_no_change": {
        "type": "text_input",
        "args": ["n_iter_no_change", '0'],
        "help": "Nombre d'itérations sans amélioration avant de déclencher un arrêt précoce."
    }
},
"MLPRegressor": {
    "hidden_layer_sizes": {
        "type": "text_input",
        "args": ["hidden_layer_sizes", '1'],
        "help": "Nombre de neurones dans chaque couche cachée. Par exemple, (100,) pour une seule couche de 100 neurones."
    },
    "activation": {
        "type": "multiselect",
        "args": ["activation", ['identity', 'logistic', 'tanh', 'relu']],
        "help": "Fonction d'activation pour les neurones de la couche cachée."
    },
    "solver": {
        "type": "multiselect",
        "args": ["solver", ['lbfgs', 'sgd', 'adam']],
        "help": "Algorithme utilisé pour optimiser les poids des neurones."
    },
    "alpha": {
        "type": "text_input",
        "args": ["alpha", '0'],
        "help": "Paramètre de régularisation L2 pour éviter le surapprentissage."
    },
    "batch_size": {
        "type": "text_input",
        "args": ["batch_size", '0'],
        "help": "Nombre de points de données dans chaque lot pour le calcul du gradient."
    },
    "learning_rate": {
        "type": "multiselect",
        "args": ["learning_rate", ['constant', 'invscaling', 'adaptive']],
        "help": "Stratégie pour ajuster le taux d'apprentissage au fil des itérations."
    },
    "learning_rate_init": {
        "type": "text_input",
        "args": ["learning_rate_init", '0'],
        "help": "Taux d'apprentissage initial pour l'optimisation du poids."
    },
    "power_t": {
        "type": "text_input",
        "args": ["power_t", '0'],
        "help": "Exposant pour la mise à l'échelle du taux d'apprentissage lors de l'utilisation de l'invscaling."
    },
    "max_iter": {
        "type": "text_input",
        "args": ["max_iter", '0'],
        "help": "Nombre maximum d'itérations pour l'optimisation."
    },
    "shuffle": {
        "type": "checkbox",
        "args": ["shuffle", False],
        "help": "Indique si les données doivent être mélangées à chaque itération."
    },
    "random_state": {
        "type": "text_input",
        "args": ["random_state", '42'],
        "help": "Contrôle la randomisation pour la reproductibilité."
    },
    "tol": {
        "type": "text_input",
        "args": ["tol", '0'],
        "help": "Seuil pour l'arrêt précoce lorsque la perte cesse de diminuer."
    },
    "verbose": {
        "type": "checkbox",
        "args": ["verbose", False],
        "help": "Active ou désactive la sortie détaillée pendant l'entraînement."
    },
    "warm_start": {
        "type": "checkbox",
        "args": ["warm_start", False],
        "help": "Réutilise la solution de l'entraînement précédent pour l'initialisation."
    },
    "momentum": {
        "type": "text_input",
        "args": ["momentum", '0'],
        "help": "Terme de momentum utilisé lors de l'optimisation par descente de gradient stochastique."
    },
    "nesterovs_momentum": {
        "type": "checkbox",
        "args": ["nesterovs_momentum", False],
        "help": "Indique si l'impulsion de Nesterov doit être utilisée."
    },
    "early_stopping": {
        "type": "checkbox",
        "args": ["early_stopping", False],
        "help": "Arrête l'entraînement tôt si la performance cesse de s'améliorer sur l'ensemble de validation."
    },
    "validation_fraction": {
        "type": "text_input",
        "args": ["validation_fraction", '0'],
        "help": "Fraction des données d'entraînement utilisée comme ensemble de validation pour l'arrêt précoce."
    },
    "beta_1": {
        "type": "text_input",
        "args": ["beta_1", '0'],
        "help": "Paramètre pour l'estimation du premier moment pour l'algorithme Adam."
    },
    "beta_2": {
        "type": "text_input",
        "args": ["beta_2", '0'],
        "help": "Paramètre pour l'estimation du second moment pour l'algorithme Adam."
    },
    "epsilon": {
        "type": "text_input",
        "args": ["epsilon", '0'],
        "help": "Valeur pour stabiliser la division lors de la mise à jour des poids."
    },
    "n_iter_no_change": {
        "type": "text_input",
        "args": ["n_iter_no_change", '0'],
        "help": "Nombre d'itérations sans amélioration avant de déclencher un arrêt précoce."
    }
},
"GaussianProcessRegressor": {
    "alpha": {
        "type": "text_input",
        "args": ["alpha", '0'],
        "help": "Valeur ajoutée à la diagonale de la matrice de covariance pour améliorer la stabilité numérique."
    },
    "optimizer": {
        "type": "multiselect",
        "args": ["optimizer", ['fmin_l_bfgs_b']],
        "help": "Algorithme d'optimisation utilisé pour maximiser la log-vraisemblance marginale."
    },
    "n_restarts_optimizer": {
        "type": "text_input",
        "args": ["n_restarts_optimizer", '0'],
        "help": "Nombre de redémarrages pour l'optimisation pour éviter de se coincer dans des minima locaux."
    },
    "normalize_y": {
        "type": "checkbox",
        "args": ["normalize_y", False],
        "help": "Indique si les cibles doivent être normalisées avant l'entraînement."
    },
    "copy_X_train": {
        "type": "checkbox",
        "args": ["copy_X_train", False],
        "help": "Indique si une copie des données d'entraînement doit être conservée."
    },
    "n_targets": {
        "type": "text_input",
        "args": ["n_targets", '0'],
        "help": "Nombre de cibles de sortie. Si 1, il s'agit d'une régression univariée."
    },
    "random_state": {
        "type": "text_input",
        "args": ["random_state", '42'],
        "help": "Contrôle la randomisation pour la reproductibilité."
    }
},
"GaussianProcessClassifier": {
    "optimizer": {
        "type": "multiselect",
        "args": ["optimizer", ['fmin_l_bfgs_b']],
        "help": "Algorithme d'optimisation utilisé pour maximiser la log-vraisemblance marginale."
    },
    "n_restarts_optimizer": {
        "type": "text_input",
        "args": ["n_restarts_optimizer", '0'],
        "help": "Nombre de redémarrages pour l'optimisation pour éviter de se coincer dans des minima locaux."
    },
    "max_iter_predict": {
        "type": "text_input",
        "args": ["max_iter_predict", '0'],
        "help": "Nombre maximum d'itérations lors de la prédiction des probabilités."
    },
    "warm_start": {
        "type": "checkbox",
        "args": ["warm_start", False],
        "help": "Réutilise la solution de l'entraînement précédent pour l'initialisation."
    },
    "copy_X_train": {
        "type": "checkbox",
        "args": ["copy_X_train", False],
        "help": "Indique si une copie des données d'entraînement doit être conservée."
    },
    "random_state": {
        "type": "text_input",
        "args": ["random_state", '42'],
        "help": "Contrôle la randomisation pour la reproductibilité."
    },
    "multi_class": {
        "type": "multiselect",
        "args": ["multi_class", ['one_vs_rest', 'one_vs_one']],
        "help": "Mode de classification multi-classes. 'one_vs_rest' entraîne un classificateur binaire pour chaque classe contre toutes les autres. 'one_vs_one' entraîne un classificateur binaire pour chaque paire de classes."
    },
    "n_jobs": {
        "type": "text_input",
        "args": ["n_jobs", '0'],
        "help": "Nombre de tâches en parallèle à exécuter. -1 signifie utiliser tous les processeurs disponibles."
    }
},
"SGDClassifier": {
    "loss": {
        "type": "multiselect",
        "args": ["loss", ['hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']],
        "help": "Fonction de perte à utiliser. Par exemple, 'hinge' pour un SVM linéaire, 'log_loss' pour la régression logistique."
    },
    "penalty": {
        "type": "multiselect",
        "args": ["penalty", ['l2', 'l1', 'elasticnet']],
        "help": "Régularisation à utiliser. 'l2' est la norme de Tikhonov, 'l1' est la régularisation de Lasso, 'elasticnet' est une combinaison des deux."
    },
    "alpha": {
        "type": "text_input",
        "args": ["alpha", '0'],
        "help": "Coefficient de régularisation pour éviter le surapprentissage."
    },
    "l1_ratio": {
        "type": "text_input",
        "args": ["l1_ratio", '0'],
        "help": "Ratio entre l1 et l2 lorsqu'on utilise 'elasticnet'."
    },
    "fit_intercept": {
        "type": "checkbox",
        "args": ["fit_intercept", False],
        "help": "Indique si un terme d'interception doit être calculé. Si False, pas d'interception dans le modèle."
    },
    "max_iter": {
        "type": "text_input",
        "args": ["max_iter", '0'],
        "help": "Nombre maximum d'itérations pour l'optimisation."
    },
    "tol": {
        "type": "text_input",
        "args": ["tol", '0'],
        "help": "Tolérance pour l'arrêt lors de l'optimisation."
    },
    "shuffle": {
        "type": "checkbox",
        "args": ["shuffle", False],
        "help": "Indique si les données doivent être mélangées après chaque itération."
    },
    "verbose": {
        "type": "text_input",
        "args": ["verbose", '0'],
        "help": "Contrôle le niveau de verbosité. Plus la valeur est élevée, plus l'algorithme affichera d'informations."
    },
    "epsilon": {
        "type": "text_input",
        "args": ["epsilon", '0'],
        "help": "Seuil pour l'insensibilité à la perte de type Huber ou epsilon_insensitive."
    },
    "n_jobs": {
        "type": "text_input",
        "args": ["n_jobs", '0'],
        "help": "Nombre de tâches en parallèle à exécuter. -1 signifie utiliser tous les processeurs disponibles."
    },
    "random_state": {
        "type": "text_input",
        "args": ["random_state", '42'],
        "help": "Contrôle la randomisation pour la reproductibilité."
    },
    "learning_rate": {
        "type": "multiselect",
        "args": ["learning_rate", ['optimal', 'constant', 'invscaling', 'adaptive']],
        "help": "Stratégie pour ajuster le taux d'apprentissage au fil des itérations."
    },
    "eta0": {
        "type": "text_input",
        "args": ["eta0", '0'],
        "help": "Taux d'apprentissage initial lorsque 'learning_rate' est 'constant', 'invscaling' ou 'adaptive'."
    },
    "power_t": {
        "type": "text_input",
        "args": ["power_t", '0'],
        "help": "Exposant pour la mise à l'échelle du taux d'apprentissage lorsque 'learning_rate' est 'invscaling'."
    },
    "early_stopping": {
        "type": "checkbox",
        "args": ["early_stopping", False],
        "help": "Arrête l'entraînement tôt si la performance cesse de s'améliorer sur l'ensemble de validation."
    },
    "validation_fraction": {
        "type": "text_input",
        "args": ["validation_fraction", '0'],
        "help": "Fraction des données d'entraînement utilisée comme ensemble de validation pour l'arrêt précoce."
    },
    "n_iter_no_change": {
        "type": "text_input",
        "args": ["n_iter_no_change", '0'],
        "help": "Nombre d'itérations sans amélioration avant de déclencher un arrêt précoce."
    },
    "class_weight": {
        "type": "multiselect",
        "args": ["class_weight", ['balanced']],
        "help": "Ajuste les poids des classes pour compenser les déséquilibres dans les données d'entraînement."
    },
    "warm_start": {
        "type": "checkbox",
        "args": ["warm_start", False],
        "help": "Réutilise la solution de l'entraînement précédent pour l'initialisation."
    },
    "average": {
        "type": "checkbox",
        "args": ["average", False],
        "help": "Ajuste l'utilisation de moyennes pondérées des coefficients tout au long de l'entraînement."
    }
},
'NuSVC': {
    'nu': {
        'type': 'text_input',
        'args': ['nu', '0'],
        'help': "Une limite supérieure sur la fraction d'erreurs d'entraînement et une limite inférieure sur la fraction de vecteurs de support. Doit être dans l'intervalle (0, 1]."
    },
    'kernel': {
        'type': 'multiselect',
        'args': ['kernel', ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']],
        'help': "Spécifie le type de noyau à utiliser dans l'algorithme. Peut être 'linear', 'poly', 'rbf', 'sigmoid', ou 'precomputed'."
    },
    'degree': {
        'type': 'text_input',
        'args': ['degree', '0'],
        'help': "Degré de la fonction de noyau polynomial ('poly'). Ignoré par tous les autres noyaux."
    },
    'gamma': {
        'type': 'multiselect',
        'args': ['gamma', ['scale', 'auto']],
        'help': "Coefficient de noyau pour 'rbf', 'poly', et 'sigmoid'. Si gamma='scale' (par défaut) est passé, alors il utilise 1 / (n_features * X.var()) comme valeur de gamma, si 'auto', utilise 1 / n_features."
    },
    'coef0': {
        'type': 'text_input',
        'args': ['coef0', '0'],
        'help': "Terme indépendant dans la fonction de noyau. Il n'est significatif que dans 'poly' et 'sigmoid'."
    },
    'shrinking': {
        'type': 'checkbox',
        'args': ['shrinking', False],
        'help': "Indique si l'heuristique de réduction doit être utilisée."
    },
    'probability': {
        'type': 'checkbox',
        'args': ['probability', False],
        'help': "Indique si les estimations de probabilité doivent être activées. Cela doit être activé avant d'appeler fit, et ralentira cette méthode."
    },
    'tol': {
        'type': 'text_input',
        'args': ['tol', '0'],
        'help': "Tolérance pour le critère d'arrêt."
    },
    'cache_size': {
        'type': 'text_input',
        'args': ['cache_size', '0'],
        'help': "Spécifie la taille du cache du noyau (en Mo)."
    },
    'class_weight': {
        'type': 'multiselect',
        'args': ['class_weight', ['balanced']],
        'help': "Définit le paramètre C de la classe i à class_weight[i] * C pour SVC. Si non donné, toutes les classes sont supposées avoir un poids de un. Le mode 'balanced' utilise les valeurs de y pour ajuster automatiquement les poids de manière inversement proportionnelle aux fréquences des classes dans les données d'entrée."
    },
    'verbose': {
        'type': 'checkbox',
        'args': ['verbose', False],
        'help': "Active la sortie détaillée. Notez que ce paramètre est spécifique à l'implémentation libsvm et peut ne pas être supporté par d'autres implémentations SVM."
    },
    'max_iter': {
        'type': 'text_input',
        'args': ['max_iter', '0'],
        'help': "Limite stricte sur les itérations dans le solveur, ou -1 pour aucune limite."
    },
    'decision_function_shape': {
        'type': 'multiselect',
        'args': ['decision_function_shape', ['ovo', 'ovr']],
        'help': "Indique s'il faut retourner une fonction de décision one-vs-rest ('ovr') de forme (n_samples, n_classes) comme tous les autres classificateurs, ou la fonction de décision one-vs-one ('ovo') d'origine de libsvm qui a la forme (n_samples, n_classes * (n_classes - 1) / 2). Cependant, notez que one-vs-one ('ovo') est toujours utilisé comme stratégie multi-classes pour entraîner les modèles ; ce paramètre n'affecte que la forme de la fonction de décision."
    },
    'break_ties': {
        'type': 'checkbox',
        'args': ['break_ties', False],
        'help': "Si vrai, decision_function_shape='ovr', et le nombre de classes > 2, predict brisera les égalités selon les valeurs de confiance de decision_function ; sinon, la première classe parmi les classes à égalité est retournée. Veuillez noter que briser les égalités (c'est-à-dire définir break_ties=True) n'est généralement pas recommandé, sauf s'il y a un besoin spécifique pour cela (voir ce problème pour plus de détails)."
    },
    'random_state': {
        'type': 'text_input',
        'args': ['random_state', '42'],
        'help': "La graine du générateur de nombres pseudo-aléatoires à utiliser lors du mélange des données pour les estimations de probabilité. Si int, random_state est la graine utilisée par le générateur de nombres aléatoires ; Si RandomState instance, random_state est le générateur de nombres aléatoires ; Si None, le générateur de nombres aléatoires est l'instance RandomState utilisée par np.random."
    }
},
'NuSVR': {
    'nu': {
        'type': 'text_input',
        'args': ['nu', '0'],
        'help': "Une limite supérieure sur la fraction d'erreurs d'entraînement et une limite inférieure sur la fraction de vecteurs de support. Doit être dans l'intervalle (0, 1]."
    },
    'kernel': {
        'type': 'multiselect',
        'args': ['kernel', ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']],
        'help': "Spécifie le type de noyau à utiliser dans l'algorithme. Peut être 'linear', 'poly', 'rbf', 'sigmoid', ou 'precomputed'."
    },
    'degree': {
        'type': 'text_input',
        'args': ['degree', '0'],
        'help': "Degré de la fonction de noyau polynomial ('poly'). Ignoré par tous les autres noyaux."
    },
    'gamma': {
        'type': 'multiselect',
        'args': ['gamma', ['scale', 'auto']],
        'help': "Coefficient de noyau pour 'rbf', 'poly', et 'sigmoid'. Si gamma='scale' (par défaut) est passé, alors il utilise 1 / (n_features * X.var()) comme valeur de gamma, si 'auto', utilise 1 / n_features."
    },
    'coef0': {
        'type': 'text_input',
        'args': ['coef0', '0'],
        'help': "Terme indépendant dans la fonction de noyau. Il n'est significatif que dans 'poly' et 'sigmoid'."
    },
    'tol': {
        'type': 'text_input',
        'args': ['tol', '0'],
        'help': "Tolérance pour le critère d'arrêt."
    },
    'C': {
        'type': 'text_input',
        'args': ['C', '0'],
        'help': "Paramètre de régularisation. La force de la régularisation est inversement proportionnelle à C. Doit être strictement positif. Le paramètre de pénalité C de la fonction de perte épsilon-insensible est utilisé."
    },
    'epsilon': {
        'type': 'text_input',
        'args': ['epsilon', '0'],
        'help': "Spécifie la tolérance d'erreur de la fonction de perte epsilon-insensible utilisée dans la régression SVR. Doit être strictement positif."
    },
    'shrinking': {
        'type': 'checkbox',
        'args': ['shrinking', False],
        'help': "Indique si l'heuristique de réduction doit être utilisée."
    },
    'cache_size': {
        'type': 'text_input',
        'args': ['cache_size', '0'],
        'help': "Spécifie la taille du cache du noyau (en Mo)."
    },
    'verbose': {
        'type': 'checkbox',
        'args': ['verbose', False],
        'help': "Active la sortie détaillée. Notez que ce paramètre est spécifique à l'implémentation libsvm et peut ne pas être supporté par d'autres implémentations SVM."
    },
    'max_iter': {
        'type': 'text_input',
        'args': ['max_iter', '0'],
        'help': "Limite stricte sur les itérations dans le solveur, ou -1 pour aucune limite."
    },
    'random_state': {
        'type': 'text_input',
        'args': ['random_state', '42'],
        'help': "La graine du générateur de nombres pseudo-aléatoires à utiliser lors du mélange des données pour les estimations de probabilité. Si int, random_state est la graine utilisée par le générateur de nombres aléatoires ; Si RandomState instance, random_state est le générateur de nombres aléatoires ; Si None, le générateur de nombres aléatoires est l'instance RandomState utilisée par np.random."
    }
},
'RadiusNeighborsClassifier': {
    'radius': {
        'type': 'text_input',
        'args': ['radius', '0'],
        'help': "Rayon à l'intérieur duquel les voisins sont considérés. Doit être strictement positif."
    },
    'weights': {
        'type': 'multiselect',
        'args': ['weights', ['uniform', 'distance']],
        'help': "Type de pondération à utiliser. 'uniform' attribue le même poids à chaque voisin, tandis que 'distance' attribue des poids proportionnels à l'inverse de la distance."
    },
    'algorithm': {
        'type': 'multiselect',
        'args': ['algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']],
        'help': "Algorithme utilisé pour calculer les voisins les plus proches. 'auto' choisira automatiquement l'algorithme le plus approprié en fonction des données d'entrée."
    },
    'leaf_size': {
        'type': 'text_input',
        'args': ['leaf_size', '0'],
        'help': "Taille de la feuille passée aux arbres BallTree ou KDTree. Cela peut affecter la vitesse de construction et de requête, ainsi que la mémoire requise pour stocker l'arbre. Doit être strictement positif."
    },
    'p': {
        'type': 'text_input',
        'args': ['p', '0'],
        'help': "Paramètre de puissance pour la métrique de Minkowski. Lorsque p = 1, cela correspond à la distance de Manhattan, et lorsque p = 2, cela correspond à la distance euclidienne."
    },
    'metric': {
        'type': 'text_input',
        'args': ['metric', '0'],
        'help': "Métrique de distance à utiliser pour la recherche de voisins. Peut être une chaîne de caractères ou une fonction de distance personnalisée."
    },
    'outlier_label': {
        'type': 'multiselect',
        'args': ['outlier_label', ['most_frequent']],
        'help': "Étiquette à attribuer aux points qui n'ont pas de voisins dans le rayon spécifié. 'most_frequent' attribue l'étiquette la plus fréquente parmi les voisins."
    },
    'metric_params': {
        'type': 'text_area',
        'args': ['metric_params', '{}'],
        'help': "Paramètres supplémentaires pour la métrique de distance. Doit être un dictionnaire."
    },
    'n_jobs': {
        'type': 'text_input',
        'args': ['n_jobs', '0'],
        'help': "Nombre de tâches à utiliser pour le calcul. -1 signifie utiliser tous les processeurs."
    }
},
'RadiusNeighborsRegressor': {
    'radius': {
        'type': 'text_input',
        'args': ['radius', '0'],
        'help': "Rayon à l'intérieur duquel les voisins sont considérés. Doit être strictement positif."
    },
    'weights': {
        'type': 'multiselect',
        'args': ['weights', ['uniform', 'distance']],
        'help': "Type de pondération à utiliser. 'uniform' attribue le même poids à chaque voisin, tandis que 'distance' attribue des poids proportionnels à l'inverse de la distance."
    },
    'algorithm': {
        'type': 'multiselect',
        'args': ['algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']],
        'help': "Algorithme utilisé pour calculer les voisins les plus proches. 'auto' choisira automatiquement l'algorithme le plus approprié en fonction des données d'entrée."
    },
    'leaf_size': {
        'type': 'text_input',
        'args': ['leaf_size', '0'],
        'help': "Taille de la feuille passée aux arbres BallTree ou KDTree. Cela peut affecter la vitesse de construction et de requête, ainsi que la mémoire requise pour stocker l'arbre. Doit être strictement positif."
    },
    'p': {
        'type': 'text_input',
        'args': ['p', '0'],
        'help': "Paramètre de puissance pour la métrique de Minkowski. Lorsque p = 1, cela correspond à la distance de Manhattan, et lorsque p = 2, cela correspond à la distance euclidienne."
    },
    'metric': {
        'type': 'text_input',
        'args': ['metric', '0'],
        'help': "Métrique de distance à utiliser pour la recherche de voisins. Peut être une chaîne de caractères ou une fonction de distance personnalisée."
    },
    'metric_params': {
        'type': 'text_area',
        'args': ['metric_params', '{}'],
        'help': "Paramètres supplémentaires pour la métrique de distance. Doit être un dictionnaire."
    },
    'n_jobs': {
        'type': 'text_input',
        'args': ['n_jobs', '0'],
        'help': "Nombre de tâches à utiliser pour le calcul. -1 signifie utiliser tous les processeurs."
    }
},
'GLM': {
    'family': {
        'type': 'multiselect',
        'args': ['family', ['gaussian', 'binomial', 'poisson', 'gamma', 'inverse_gaussian', 'tweedie']],
        'help': "Type de distribution à utiliser pour le modèle de régression généralisée. Peut être 'gaussian' (normale), 'binomial' (binaire), 'poisson' (comptage), 'gamma' (positive continue), 'inverse_gaussian' (positive continue), ou 'tweedie' (composé de Poisson)."
    }
},
'CatBoostRegressor': {
    'iterations': {
        'type': 'text_input',
        'args': ['iterations', '100'],
        'help': "Nombre d'itérations à effectuer. Plus le nombre est élevé, plus le modèle sera entraîné longtemps."
    },
    'learning_rate': {
        'type': 'text_input',
        'args': ['learning_rate', '0.1'],
        'help': "Taux d'apprentissage utilisé pour l'entraînement. Un taux d'apprentissage plus élevé peut accélérer l'entraînement mais peut aussi entraîner une convergence moins stable."
    },
    'depth': {
        'type': 'text_input',
        'args': ['depth', '6'],
        'help': "Profondeur des arbres de décision. Une profondeur plus élevée peut capturer des relations plus complexes mais peut aussi entraîner un surapprentissage."
    },
    'l2_leaf_reg': {
        'type': 'text_input',
        'args': ['l2_leaf_reg', '3.0'],
        'help': "Coefficient de régularisation L2 appliqué aux feuilles des arbres. Une valeur plus élevée peut aider à prévenir le surapprentissage."
    },
    'model_size_reg': {
        'type': 'text_input',
        'args': ['model_size_reg', '0.5'],
        'help': "Coefficient de régularisation pour la taille du modèle. Une valeur plus élevée peut réduire la taille du modèle."
    },
    'rsm': {
        'type': 'text_input',
        'args': ['rsm', '1.0'],
        'help': "Coefficient de régularisation pour la taille des feuilles. Une valeur plus élevée peut réduire la taille des feuilles."
    },
    'loss_function': {
        'type': 'multiselect',
        'args': ['loss_function', ['RMSE', 'MAE', 'Quantile', 'LogLinQuantile', 'Poisson', 'MAPE', 'Lq']],
        'help': "Fonction de perte à utiliser pour l'entraînement. Peut être 'RMSE', 'MAE', 'Quantile', 'LogLinQuantile', 'Poisson', 'MAPE', ou 'Lq'."
    },
    'border_count': {
        'type': 'text_input',
        'args': ['border_count', '32'],
        'help': "Nombre de bords à utiliser pour la quantification des caractéristiques."
    },
    'feature_border_type': {
        'type': 'multiselect',
        'args': ['feature_border_type', ['Uniform', 'UniformAndQuantiles', 'GreedyLogSum', 'MaxLogSum', 'Median']],
        'help': "Type de bordure à utiliser pour la quantification des caractéristiques."
    },
    'per_float_feature_quantization': {
        'type': 'text_input',
        'args': ['per_float_feature_quantization', '0'],
        'help': "Quantification des caractéristiques flottantes."
    },
    'input_borders': {
        'type': 'text_input',
        'args': ['input_borders', '0'],
        'help': "Bordures d'entrée pour la quantification des caractéristiques."
    },
    'output_borders': {
        'type': 'text_input',
        'args': ['output_borders', '0'],
        'help': "Bordures de sortie pour la quantification des caractéristiques."
    },
    'fold_permutation_block': {
        'type': 'text_input',
        'args': ['fold_permutation_block', '0'],
        'help': "Taille du bloc de permutation pour le pliage."
    },
    'od_pval': {
        'type': 'text_input',
        'args': ['od_pval', '0.0'],
        'help': "Valeur p pour la détection des valeurs aberrantes."
    },
    'od_wait': {
        'type': 'text_input',
        'args': ['od_wait', '0'],
        'help': "Nombre d'itérations à attendre avant de commencer la détection des valeurs aberrantes."
    },
    'od_type': {
        'type': 'multiselect',
        'args': ['od_type', ['IncToDec', 'Iter']],
        'help': "Type de détection des valeurs aberrantes."
    },
    'nan_mode': {
        'type': 'multiselect',
        'args': ['nan_mode', ['Min', 'Max', 'Forbidden']],
        'help': "Mode de gestion des valeurs manquantes (NaN)."
    },
    'counter_calc_method': {
        'type': 'multiselect',
        'args': ['counter_calc_method', ['Full', 'SkipTest']],
        'help': "Méthode de calcul des compteurs."
    },
    'leaf_estimation_iterations': {
        'type': 'text_input',
        'args': ['leaf_estimation_iterations', '1'],
        'help': "Nombre d'itérations pour l'estimation des feuilles."
    },
    'leaf_estimation_method': {
        'type': 'multiselect',
        'args': ['leaf_estimation_method', ['Newton', 'Gradient']],
        'help': "Méthode d'estimation des feuilles."
    },
    'thread_count': {
        'type': 'text_input',
        'args': ['thread_count', '0'],
        'help': "Nombre de threads à utiliser pour le calcul. -1 signifie utiliser tous les processeurs."
    },
    'random_seed': {
        'type': 'text_input',
        'args': ['random_seed', '0'],
        'help': "Graine pour le générateur de nombres aléatoires."
    },
    'use_best_model': {
        'type': 'checkbox',
        'args': ['use_best_model', False],
        'help': "Indique si le meilleur modèle doit être utilisé."
    },
    'best_model_min_trees': {
        'type': 'text_input',
        'args': ['best_model_min_trees', '0'],
        'help': "Nombre minimum d'arbres pour considérer le meilleur modèle."
    },
    'verbose': {
        'type': 'checkbox',
        'args': ['verbose', False],
        'help': "Active la sortie détaillée."
    },
    'silent': {
        'type': 'checkbox',
        'args': ['silent', False],
        'help': "Désactive la sortie détaillée."
    },
    'logging_level': {
        'type': 'multiselect',
        'args': ['logging_level', ['Silent', 'Verbose', 'Info', 'Debug']],
        'help': "Niveau de journalisation."
    },
    'metric_period': {
        'type': 'text_input',
        'args': ['metric_period', '0'],
        'help': "Période de calcul des métriques."
    },
    'ctr_leaf_count_limit': {
        'type': 'text_input',
        'args': ['ctr_leaf_count_limit', '0'],
        'help': "Limite du nombre de feuilles pour les compteurs de taux de clic (CTR)."
    },
    'store_all_simple_ctr': {
        'type': 'checkbox',
        'args': ['store_all_simple_ctr', False],
        'help': "Indique si tous les compteurs de taux de clic simples doivent être stockés."
    },
    'max_ctr_complexity': {
        'type': 'text_input',
        'args': ['max_ctr_complexity', '0'],
        'help': "Complexité maximale des compteurs de taux de clic."
    },
    'has_time': {
        'type': 'checkbox',
        'args': ['has_time', False],
        'help': "Indique si les données contiennent une dimension temporelle."
    },
    'allow_const_label': {
        'type': 'checkbox',
        'args': ['allow_const_label', False],
        'help': "Indique si les étiquettes constantes sont autorisées."
    },
    'one_hot_max_size': {
        'type': 'text_input',
        'args': ['one_hot_max_size', '0'],
        'help': "Taille maximale pour l'encodage one-hot."
    },
    'random_strength': {
        'type': 'text_input',
        'args': ['random_strength', '1.0'],
        'help': "Force de la randomisation."
    },
    'name': {
        'type': 'text_input',
        'args': ['name', '0'],
        'help': "Nom du modèle."
    },
    'ignored_features': {
        'type': 'text_input',
        'args': ['ignored_features', '0'],
        'help': "Caractéristiques à ignorer."
    },
    'train_dir': {
        'type': 'text_input',
        'args': ['train_dir', '0'],
        'help': "Répertoire d'entraînement."
    },
    'custom_metric': {
        'type': 'text_input',
        'args': ['custom_metric', '0'],
        'help': "Métrique personnalisée à utiliser."
    },
    'eval_metric': {
        'type': 'text_input',
        'args': ['eval_metric', '0'],
        'help': "Métrique d'évaluation à utiliser."
    },
    'bagging_temperature': {
        'type': 'text_input',
        'args': ['bagging_temperature', '0.0'],
        'help': "Température de bagging."
    },
    'save_snapshot': {
        'type': 'checkbox',
        'args': ['save_snapshot', False],
        'help': "Indique si un snapshot doit être sauvegardé."
    },
    'snapshot_file': {
        'type': 'text_input',
        'args': ['snapshot_file', '0'],
        'help': "Fichier de snapshot."
    },
    'snapshot_interval': {
        'type': 'text_input',
        'args': ['snapshot_interval', '0'],
        'help': "Intervalle de sauvegarde des snapshots."
    },
    'fold_len_multiplier': {
        'type': 'text_input',
        'args': ['fold_len_multiplier', '2.0'],
        'help': "Multiplicateur de longueur de pli."
    },
    'used_ram_limit': {
        'type': 'text_input',
        'args': ['used_ram_limit', '0'],
        'help': "Limite de RAM utilisée."
    },
    'gpu_ram_part': {
        'type': 'text_input',
        'args': ['gpu_ram_part', '0.95'],
        'help': "Partie de la RAM GPU à utiliser."
    },
    'pinned_memory_size': {
        'type': 'text_input',
        'args': ['pinned_memory_size', '0'],
        'help': "Taille de la mémoire épinglée."
    },
    'allow_writing_files': {
        'type': 'checkbox',
        'args': ['allow_writing_files', False],
        'help': "Indique si l'écriture de fichiers est autorisée."
    },
    'final_ctr_computation_mode': {
        'type': 'multiselect',
        'args': ['final_ctr_computation_mode', ['Default', 'Skip']],
        'help': "Mode de calcul final des compteurs de taux de clic."
    },
    'approx_on_full_history': {
        'type': 'checkbox',
        'args': ['approx_on_full_history', False],
        'help': "Indique si l'approximation sur l'historique complet doit être utilisée."
    },
    'boosting_type': {
        'type': 'multiselect',
        'args': ['boosting_type', ['Ordered', 'Plain']],
        'help': "Type de boosting à utiliser."
    },
    'simple_ctr': {
        'type': 'text_input',
        'args': ['simple_ctr', '0'],
        'help': "Compteurs de taux de clic simples."
    },
    'combinations_ctr': {
        'type': 'text_input',
        'args': ['combinations_ctr', '0'],
        'help': "Compteurs de taux de clic de combinaisons."
    },
    'per_feature_ctr': {
        'type': 'text_input',
        'args': ['per_feature_ctr', '0'],
        'help': "Compteurs de taux de clic par caractéristique."
    },
    'ctr_target_border_count': {
        'type': 'text_input',
        'args': ['ctr_target_border_count', '0'],
        'help': "Nombre de bordures cibles pour les compteurs de taux de clic."
    },
    'task_type': {
        'type': 'multiselect',
        'args': ['task_type', ['CPU', 'GPU']],
        'help': "Type de tâche à utiliser (CPU ou GPU)."
    },
    'device_config': {
        'type': 'text_input',
        'args': ['device_config', '0'],
        'help': "Configuration des dispositifs."
    },
    'devices': {
        'type': 'text_input',
        'args': ['devices', '0'],
        'help': "Dispositifs à utiliser."
    },
    'bootstrap_type': {
        'type': 'multiselect',
        'args': ['bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS', 'No']],
        'help': "Type de bootstrap à utiliser."
    },
    'subsample': {
        'type': 'text_input',
        'args': ['subsample', '1'],
        'help': "Fraction des données à utiliser pour l'entraînement."
    },
    'sampling_unit': {
        'type': 'multiselect',
        'args': ['sampling_unit', ['Object', 'Group']],
        'help': "Unité d'échantillonnage."
    },
    'dev_score_calc_obj_block_size': {
        'type': 'text_input',
        'args': ['dev_score_calc_obj_block_size', '0'],
        'help': "Taille du bloc d'objets pour le calcul du score de développement."
    },
    'max_depth': {
        'type': 'text_input',
        'args': ['max_depth', '1'],
        'help': "Profondeur maximale des arbres de décision."
    }
},
'CatBoostClassifier': {
    'iterations': {
        'type': 'text_input',
        'args': ['iterations', '0'],
        'help': "Nombre d'itérations à effectuer. Plus le nombre est élevé, plus le modèle sera entraîné longtemps."
    },
    'learning_rate': {
        'type': 'text_input',
        'args': ['learning_rate', '0'],
        'help': "Taux d'apprentissage utilisé pour l'entraînement. Un taux d'apprentissage plus élevé peut accélérer l'entraînement mais peut aussi entraîner une convergence moins stable."
    },
    'depth': {
        'type': 'text_input',
        'args': ['depth', '0'],
        'help': "Profondeur des arbres de décision. Une profondeur plus élevée peut capturer des relations plus complexes mais peut aussi entraîner un surapprentissage."
    },
    'l2_leaf_reg': {
        'type': 'text_input',
        'args': ['l2_leaf_reg', '0'],
        'help': "Coefficient de régularisation L2 appliqué aux feuilles des arbres. Une valeur plus élevée peut aider à prévenir le surapprentissage."
    },
    'model_size_reg': {
        'type': 'text_input',
        'args': ['model_size_reg', '0'],
        'help': "Coefficient de régularisation pour la taille du modèle. Une valeur plus élevée peut réduire la taille du modèle."
    },
    'rsm': {
        'type': 'text_input',
        'args': ['rsm', '0'],
        'help': "Coefficient de régularisation pour la taille des feuilles. Une valeur plus élevée peut réduire la taille des feuilles."
    },
    'loss_function': {
        'type': 'multiselect',
        'args': ['loss_function', ['Logloss', 'CrossEntropy', 'MultiClass', 'RMSE', 'Quantile', 'Poisson', 'MAE', 'Huber', 'QueryRMSE', 'QueryCrossEntropy']],
        'help': "Fonction de perte à utiliser pour l'entraînement. Peut être 'Logloss', 'CrossEntropy', 'MultiClass', 'RMSE', 'Quantile', 'Poisson', 'MAE', 'Huber', 'QueryRMSE', ou 'QueryCrossEntropy'."
    },
    'border_count': {
        'type': 'text_input',
        'args': ['border_count', '0'],
        'help': "Nombre de bords à utiliser pour la quantification des caractéristiques."
    },
    'feature_border_type': {
        'type': 'multiselect',
        'args': ['feature_border_type', ['Median', 'Uniform', 'GreedyLogSum', 'MaxLogSum', 'MinEntropy']],
        'help': "Type de bordure à utiliser pour la quantification des caractéristiques."
    },
    'per_float_feature_quantization': {
        'type': 'text_input',
        'args': ['per_float_feature_quantization', '0'],
        'help': "Quantification des caractéristiques flottantes."
    },
    'input_borders': {
        'type': 'text_input',
        'args': ['input_borders', '0'],
        'help': "Bordures d'entrée pour la quantification des caractéristiques."
    },
    'output_borders': {
        'type': 'text_input',
        'args': ['output_borders', '0'],
        'help': "Bordures de sortie pour la quantification des caractéristiques."
    },
    'fold_permutation_block': {
        'type': 'text_input',
        'args': ['fold_permutation_block', '0'],
        'help': "Taille du bloc de permutation pour le pliage."
    },
    'od_pval': {
        'type': 'text_input',
        'args': ['od_pval', '0'],
        'help': "Valeur p pour la détection des valeurs aberrantes."
    },
    'od_wait': {
        'type': 'text_input',
        'args': ['od_wait', '0'],
        'help': "Nombre d'itérations à attendre avant de commencer la détection des valeurs aberrantes."
    },
    'od_type': {
        'type': 'multiselect',
        'args': ['od_type', ['IncToDec', 'Iter']],
        'help': "Type de détection des valeurs aberrantes."
    },
    'nan_mode': {
        'type': 'multiselect',
        'args': ['nan_mode', ['Min', 'Max', 'Poison']],
        'help': "Mode de gestion des valeurs manquantes (NaN)."
    },
    'counter_calc_method': {
        'type': 'multiselect',
        'args': ['counter_calc_method', ['FullDocument', 'Percanon']],
        'help': "Méthode de calcul des compteurs."
    },
    'leaf_estimation_iterations': {
        'type': 'text_input',
        'args': ['leaf_estimation_iterations', '0'],
        'help': "Nombre d'itérations pour l'estimation des feuilles."
    },
    'leaf_estimation_method': {
        'type': 'multiselect',
        'args': ['leaf_estimation_method', ['Gradient', 'Newton']],
        'help': "Méthode d'estimation des feuilles."
    },
    'thread_count': {
        'type': 'text_input',
        'args': ['thread_count', '0'],
        'help': "Nombre de threads à utiliser pour le calcul. -1 signifie utiliser tous les processeurs."
    },
    'random_seed': {
        'type': 'text_input',
        'args': ['random_seed', '0'],
        'help': "Graine pour le générateur de nombres aléatoires."
    },
    'use_best_model': {
        'type': 'checkbox',
        'args': ['use_best_model', False],
        'help': "Indique si le meilleur modèle doit être utilisé."
    },
    'verbose': {
        'type': 'text_input',
        'args': ['verbose', '0'],
        'help': "Active la sortie détaillée."
    },
    'logging_level': {
        'type': 'multiselect',
        'args': ['logging_level', ['Silent', 'Fatal', 'Error', 'Warning', 'Info', 'Debug', 'Verbose']],
        'help': "Niveau de journalisation."
    },
    'metric_period': {
        'type': 'text_input',
        'args': ['metric_period', '0'],
        'help': "Période de calcul des métriques."
    },
    'ctr_leaf_count_limit': {
        'type': 'text_input',
        'args': ['ctr_leaf_count_limit', '0'],
        'help': "Limite du nombre de feuilles pour les compteurs de taux de clic (CTR)."
    },
    'store_all_simple_ctr': {
        'type': 'checkbox',
        'args': ['store_all_simple_ctr', False],
        'help': "Indique si tous les compteurs de taux de clic simples doivent être stockés."
    },
    'max_ctr_complexity': {
        'type': 'text_input',
        'args': ['max_ctr_complexity', '0'],
        'help': "Complexité maximale des compteurs de taux de clic."
    },
    'has_time': {
        'type': 'checkbox',
        'args': ['has_time', False],
        'help': "Indique si les données contiennent une dimension temporelle."
    },
    'allow_const_label': {
        'type': 'checkbox',
        'args': ['allow_const_label', False],
        'help': "Indique si les étiquettes constantes sont autorisées."
    },
    'classes_count': {
        'type': 'text_input',
        'args': ['classes_count', '0'],
        'help': "Nombre de classes dans le problème de classification."
    },
    'class_weights': {
        'type': 'text_input',
        'args': ['class_weights', '0'],
        'help': "Poids des classes pour la classification."
    },
    'auto_class_weights': {
        'type': 'multiselect',
        'args': ['auto_class_weights', ['0', 'Balanced', 'Balanced_0']],
        'help': "Poids automatiques des classes."
    },
    'one_hot_max_size': {
        'type': 'text_input',
        'args': ['one_hot_max_size', '0'],
        'help': "Taille maximale pour l'encodage one-hot."
    },
    'random_strength': {
        'type': 'text_input',
        'args': ['random_strength', '0'],
        'help': "Force de la randomisation."
    },
    'name': {
        'type': 'text_input',
        'args': ['name', '0'],
        'help': "Nom du modèle."
    },
    'ignored_features': {
        'type': 'text_input',
        'args': ['ignored_features', '0'],
        'help': "Caractéristiques à ignorer."
    },
    'train_dir': {
        'type': 'text_input',
        'args': ['train_dir', '0'],
        'help': "Répertoire d'entraînement."
    },
    'custom_loss': {
        'type': 'text_input',
        'args': ['custom_loss', '0'],
        'help': "Fonction de perte personnalisée à utiliser."
    },
    'custom_metric': {
        'type': 'text_input',
        'args': ['custom_metric', '0'],
        'help': "Métrique personnalisée à utiliser."
    },
    'eval_metric': {
        'type': 'text_input',
        'args': ['eval_metric', '0'],
        'help': "Métrique d'évaluation à utiliser."
    },
    'bagging_temperature': {
        'type': 'text_input',
        'args': ['bagging_temperature', '0'],
        'help': "Température de bagging."
    },
    'save_snapshot': {
        'type': 'checkbox',
        'args': ['save_snapshot', False],
        'help': "Indique si un snapshot doit être sauvegardé."
    },
    'snapshot_file': {
        'type': 'text_input',
        'args': ['snapshot_file', '0'],
        'help': "Fichier de snapshot."
    },
    'snapshot_interval': {
        'type': 'text_input',
        'args': ['snapshot_interval', '0'],
        'help': "Intervalle de sauvegarde des snapshots."
    },
    'fold_len_multiplier': {
        'type': 'text_input',
        'args': ['fold_len_multiplier', '0'],
        'help': "Multiplicateur de longueur de pli."
    },
    'used_ram_limit': {
        'type': 'text_input',
        'args': ['used_ram_limit', '0'],
        'help': "Limite de RAM utilisée."
    },
    'gpu_ram_part': {
        'type': 'text_input',
        'args': ['gpu_ram_part', '0'],
        'help': "Partie de la RAM GPU à utiliser."
    },
    'allow_writing_files': {
        'type': 'checkbox',
        'args': ['allow_writing_files', False],
        'help': "Indique si l'écriture de fichiers est autorisée."
    },
    'final_ctr_computation_mode': {
        'type': 'multiselect',
        'args': ['final_ctr_computation_mode', ['Default', 'CatFeatures', 'BinarizedTarget']],
        'help': "Mode de calcul final des compteurs de taux de clic."
    },
    'approx_on_full_history': {
        'type': 'checkbox',
        'args': ['approx_on_full_history', False],
        'help': "Indique si l'approximation sur l'historique complet doit être utilisée."
    },
    'boosting_type': {
        'type': 'multiselect',
        'args': ['boosting_type', ['Ordered', 'Plain']],
        'help': "Type de boosting à utiliser."
    },
    'simple_ctr': {
        'type': 'text_input',
        'args': ['simple_ctr', '0'],
        'help': "Compteurs de taux de clic simples."
    },
    'combinations_ctr': {
        'type': 'text_input',
        'args': ['combinations_ctr', '0'],
        'help': "Compteurs de taux de clic de combinaisons."
    },
    'per_feature_ctr': {
        'type': 'text_input',
        'args': ['per_feature_ctr', '0'],
        'help': "Compteurs de taux de clic par caractéristique."
    },
    'task_type': {
        'type': 'multiselect',
        'args': ['task_type', ['CPU', 'GPU']],
        'help': "Type de tâche à utiliser (CPU ou GPU)."
    },
    'device_config': {
        'type': 'text_input',
        'args': ['device_config', '0'],
        'help': "Configuration des dispositifs."
    },
    'devices': {
        'type': 'text_input',
        'args': ['devices', '0'],
        'help': "Dispositifs à utiliser."
    },
    'bootstrap_type': {
        'type': 'multiselect',
        'args': ['bootstrap_type', ['Bayesian', 'Bernoulli', 'Poisson']],
        'help': "Type de bootstrap à utiliser."
    },
    'subsample': {
        'type': 'text_input',
        'args': ['subsample', '0'],
        'help': "Fraction des données à utiliser pour l'entraînement."
    },
    'sampling_unit': {
        'type': 'multiselect',
        'args': ['sampling_unit', ['Objects', 'Objects_surrogate']],
        'help': "Unité d'échantillonnage."
    },
    'dev_score_calc_obj_block_size': {
        'type': 'text_input',
        'args': ['dev_score_calc_obj_block_size', '0'],
        'help': "Taille du bloc d'objets pour le calcul du score de développement."
    },
    'max_depth': {
        'type': 'text_input',
        'args': ['max_depth', '0'],
        'help': "Profondeur maximale des arbres de décision."
    },
    'n_estimators': {
        'type': 'text_input',
        'args': ['n_estimators', '0'],
        'help': "Nombre d'estimateurs à utiliser."
    },
    'num_boost_round': {
        'type': 'text_input',
        'args': ['num_boost_round', '0'],
        'help': "Nombre de tours de boosting."
    },
    'num_trees': {
        'type': 'text_input',
        'args': ['num_trees', '0'],
        'help': "Nombre d'arbres à utiliser."
    },
    'colsample_bylevel': {
        'type': 'text_input',
        'args': ['colsample_bylevel', '0'],
        'help': "Fraction de colonnes à échantillonner par niveau."
    },
    'random_state': {
        'type': 'text_input',
        'args': ['random_state', '42'],
        'help': "Graine pour le générateur de nombres aléatoires."
    },
    'reg_lambda': {
        'type': 'text_input',
        'args': ['reg_lambda', '0'],
        'help': "Coefficient de régularisation L2."
    },
    'objective': {
        'type': 'multiselect',
        'args': ['objective', ['Logloss', 'CrossEntropy', 'MultiClass', 'RMSE', 'Quantile', 'Poisson', 'MAE', 'Huber', 'QueryRMSE', 'QueryCrossEntropy']],
        'help': "Fonction objectif à utiliser pour l'entraînement."
    },
    'eta': {
        'type': 'text_input',
        'args': ['eta', '0'],
        'help': "Taux d'apprentissage."
    },
    'max_bin': {
        'type': 'text_input',
        'args': ['max_bin', '0'],
        'help': "Nombre maximal de bins pour la quantification des caractéristiques."
    },
    'scale_pos_weight': {
        'type': 'text_input',
        'args': ['scale_pos_weight', '0'],
        'help': "Poids des échantillons positifs."
    },
    'gpu_cat_features_storage': {
        'type': 'multiselect',
        'args': ['gpu_cat_features_storage', ['GpuRam', 'CpuRam']],
        'help': "Stockage des caractéristiques catégorielles sur GPU."
    },
    'data_partition': {
        'type': 'multiselect',
        'args': ['data_partition', ['FeatureParallel', 'DocParallel']],
        'help': "Partitionnement des données."
    },
    'metadata': {
        'type': 'text_input',
        'args': ['metadata', '0'],
        'help': "Métadonnées."
    },
    'early_stopping_rounds': {
        'type': 'text_input',
        'args': ['early_stopping_rounds', '0'],
        'help': "Nombre de tours sans amélioration pour arrêter l'entraînement précocement."
    },
    'cat_features': {
        'type': 'text_input',
        'args': ['cat_features', '0'],
        'help': "Caractéristiques catégorielles."
    },
    'grow_policy': {
        'type': 'multiselect',
        'args': ['grow_policy', ['Depthwise', 'Lossguide']],
        'help': "Politique de croissance des arbres."
    },
    'min_data_in_leaf': {
        'type': 'text_input',
        'args': ['min_data_in_leaf', '0'],
        'help': "Nombre minimum de données dans une feuille."
    },
    'min_child_samples': {
        'type': 'text_input',
        'args': ['min_child_samples', '0'],
        'help': "Nombre minimum d'échantillons enfants."
    },
    'max_leaves': {
        'type': 'text_input',
        'args': ['max_leaves', '0'],
        'help': "Nombre maximum de feuilles."
    },
    'num_leaves': {
        'type': 'text_input',
        'args': ['num_leaves', '0'],
        'help': "Nombre de feuilles."
    },
    'score_function': {
        'type': 'multiselect',
        'args': ['score_function', ['Correlation', 'NewtonOnFullGradient', 'NewtonRidgeRegression']],
        'help': "Fonction de score à utiliser."
    },
    'leaf_estimation_backtracking': {
        'type': 'multiselect',
        'args': ['leaf_estimation_backtracking', ['No', 'AnyImprovement', 'Armijo']],
        'help': "Méthode de retour en arrière pour l'estimation des feuilles."
    },
    'ctr_history_unit': {
        'type': 'multiselect',
        'args': ['ctr_history_unit', ['Default', 'DocId', 'FeatureId']],
        'help': "Unité d'historique des compteurs de taux de clic."
    },
    'monotone_constraints': {
        'type': 'text_input',
        'args': ['monotone_constraints', '0'],
        'help': "Contraintes monotones."
    },
    'feature_weights': {
        'type': 'text_input',
        'args': ['feature_weights', '0'],
        'help': "Poids des caractéristiques."
    },
    'penalties_coefficient': {
        'type': 'text_input',
        'args': ['penalties_coefficient', '0'],
        'help': "Coefficient de pénalités."
    },
    'first_feature_use_penalties': {
        'type': 'checkbox',
        'args': ['first_feature_use_penalties', False],
        'help': "Indique si les pénalités doivent être utilisées pour la première utilisation des caractéristiques."
    },
    'model_shrink_rate': {
        'type': 'text_input',
        'args': ['model_shrink_rate', '0'],
        'help': "Taux de réduction du modèle."
    },
    'model_shrink_mode': {
        'type': 'multiselect',
        'args': ['model_shrink_mode', ['Constant', 'Freeze']],
        'help': "Mode de réduction du modèle."
    },
    'langevin': {
        'type': 'checkbox',
        'args': ['langevin', False],
        'help': "Indique si l'algorithme de Langevin doit être utilisé."
    },
    'diffusion_temperature': {
        'type': 'text_input',
        'args': ['diffusion_temperature', '0'],
        'help': "Température de diffusion."
    },
    'posterior_sampling': {
        'type': 'checkbox',
        'args': ['posterior_sampling', False],
        'help': "Indique si l'échantillonnage postérieur doit être utilisé."
    },
    'boost_from_average': {
        'type': 'checkbox',
        'args': ['boost_from_average', False],
        'help': "Indique si le boosting doit commencer à partir de la moyenne."
    },
    'text_features': {
        'type': 'text_input',
        'args': ['text_features', '0'],
        'help': "Caractéristiques textuelles."
    },
    'tokenizers': {
        'type': 'text_input',
        'args': ['tokenizers', '0'],
        'help': "Tokenizers à utiliser."
    },
    'dictionaries': {
        'type': 'text_input',
        'args': ['dictionaries', '0'],
        'help': "Dictionnaires à utiliser."
    },
    'feature_calcers': {
        'type': 'text_input',
        'args': ['feature_calcers', '0'],
        'help': "Calculateurs de caractéristiques à utiliser."
    },
    'text_processing': {
        'type': 'text_input',
        'args': ['text_processing', '0'],
        'help': "Traitement du texte."
    },
    'fixed_binary_splits': {
        'type': 'checkbox',
        'args': ['fixed_binary_splits', False],
        'help': "Indique si les divisions binaires fixes doivent être utilisées."
    }
},
'LGBMRegressor': {
    'boosting_type': {
        'type': 'multiselect',
        'args': ['boosting_type', ['gbdt', 'dart', 'rf']],
        'help': "Type de boosting à utiliser. Peut être 'gbdt' (Gradient Boosting Decision Tree), 'dart' (Dropouts meet Multiple Additive Regression Trees), ou 'rf' (Random Forest)."
    },
    'num_leaves': {
        'type': 'text_input',
        'args': ['num_leaves', '31'],
        'help': "Nombre maximum de feuilles dans un arbre. Une valeur plus élevée peut capturer des relations plus complexes mais peut aussi entraîner un surapprentissage."
    },
    'max_depth': {
        'type': 'text_input',
        'args': ['max_depth', '-1'],
        'help': "Profondeur maximale des arbres de décision. Une valeur de -1 signifie qu'il n'y a pas de limite de profondeur."
    },
    'learning_rate': {
        'type': 'text_input',
        'args': ['learning_rate', '0.1'],
        'help': "Taux d'apprentissage utilisé pour l'entraînement. Un taux d'apprentissage plus élevé peut accélérer l'entraînement mais peut aussi entraîner une convergence moins stable."
    },
    'n_estimators': {
        'type': 'text_input',
        'args': ['n_estimators', '100'],
        'help': "Nombre d'estimateurs (arbres) à utiliser. Plus le nombre est élevé, plus le modèle sera entraîné longtemps."
    },
    'subsample_for_bin': {
        'type': 'text_input',
        'args': ['subsample_for_bin', '0'],
        'help': "Nombre d'échantillons à utiliser pour la construction des bins. Une valeur plus élevée peut améliorer la précision mais augmentera le temps de calcul."
    },
    'min_split_gain': {
        'type': 'text_input',
        'args': ['min_split_gain', '0.0'],
        'help': "Gain minimum requis pour effectuer une division. Une valeur plus élevée peut aider à prévenir le surapprentissage."
    },
    'min_child_weight': {
        'type': 'text_input',
        'args': ['min_child_weight', '0.001'],
        'help': "Poids minimum des échantillons dans une feuille. Une valeur plus élevée peut aider à prévenir le surapprentissage."
    },
    'min_child_samples': {
        'type': 'text_input',
        'args': ['min_child_samples', '20'],
        'help': "Nombre minimum d'échantillons requis pour être dans une feuille. Une valeur plus élevée peut aider à prévenir le surapprentissage."
    },
    'subsample': {
        'type': 'text_input',
        'args': ['subsample', '1.0'],
        'help': "Fraction des échantillons à utiliser pour l'entraînement de chaque arbre. Une valeur inférieure à 1.0 peut aider à prévenir le surapprentissage."
    },
    'subsample_freq': {
        'type': 'text_input',
        'args': ['subsample_freq', '0'],
        'help': "Fréquence de sous-échantillonnage. Une valeur de 0 signifie qu'il n'y a pas de sous-échantillonnage."
    },
    'colsample_bytree': {
        'type': 'text_input',
        'args': ['colsample_bytree', '1.0'],
        'help': "Fraction des caractéristiques à utiliser pour l'entraînement de chaque arbre. Une valeur inférieure à 1.0 peut aider à prévenir le surapprentissage."
    },
    'reg_alpha': {
        'type': 'text_input',
        'args': ['reg_alpha', '0.0'],
        'help': "Coefficient de régularisation L1. Une valeur plus élevée peut aider à prévenir le surapprentissage."
    },
    'reg_lambda': {
        'type': 'text_input',
        'args': ['reg_lambda', '0.0'],
        'help': "Coefficient de régularisation L2. Une valeur plus élevée peut aider à prévenir le surapprentissage."
    },
    'random_state': {
        'type': 'text_input',
        'args': ['random_state', '42'],
        'help': "Graine pour le générateur de nombres aléatoires. Utilisé pour garantir la reproductibilité des résultats."
    },
    'n_jobs': {
        'type': 'text_input',
        'args': ['n_jobs', '0'],
        'help': "Nombre de tâches à utiliser pour le calcul. -1 signifie utiliser tous les processeurs."
    },
    'importance_type': {
        'type': 'multiselect',
        'args': ['importance_type', ['split', 'gain']],
        'help': "Type d'importance des caractéristiques à utiliser. Peut être 'split' (nombre de fois où une caractéristique est utilisée pour diviser) ou 'gain' (gain total de la division utilisant la caractéristique)."
    }
},
'LGBMClassifier': {
    'boosting_type': {
        'type': 'multiselect',
        'args': ['boosting_type', ['gbdt', 'dart', 'rf']],
        'help': "Type de boosting à utiliser. Peut être 'gbdt' (Gradient Boosting Decision Tree), 'dart' (Dropouts meet Multiple Additive Regression Trees), ou 'rf' (Random Forest)."
    },
    'num_leaves': {
        'type': 'text_input',
        'args': ['num_leaves', '31'],
        'help': "Nombre maximum de feuilles dans un arbre. Une valeur plus élevée peut capturer des relations plus complexes mais peut aussi entraîner un surapprentissage."
    },
    'max_depth': {
        'type': 'text_input',
        'args': ['max_depth', '-1'],
        'help': "Profondeur maximale des arbres de décision. Une valeur de -1 signifie qu'il n'y a pas de limite de profondeur."
    },
    'learning_rate': {
        'type': 'text_input',
        'args': ['learning_rate', '0.1'],
        'help': "Taux d'apprentissage utilisé pour l'entraînement. Un taux d'apprentissage plus élevé peut accélérer l'entraînement mais peut aussi entraîner une convergence moins stable."
    },
    'n_estimators': {
        'type': 'text_input',
        'args': ['n_estimators', '100'],
        'help': "Nombre d'estimateurs (arbres) à utiliser. Plus le nombre est élevé, plus le modèle sera entraîné longtemps."
    },
    'subsample_for_bin': {
        'type': 'text_input',
        'args': ['subsample_for_bin', '0'],
        'help': "Nombre d'échantillons à utiliser pour la construction des bins. Une valeur plus élevée peut améliorer la précision mais augmentera le temps de calcul."
    },
    'objective': {
        'type': 'multiselect',
        'args': ['objective', ['binary', 'multiclass']],
        'help': "Type d'objectif à utiliser pour la classification. Peut être 'binary' pour la classification binaire ou 'multiclass' pour la classification multi-classes."
    },
    'class_weight': {
        'type': 'multiselect',
        'args': ['class_weight', ['0', 'balanced']],
        'help': "Poids des classes pour la classification. 'balanced' ajuste automatiquement les poids en fonction de la fréquence des classes."
    },
    'min_split_gain': {
        'type': 'text_input',
        'args': ['min_split_gain', '0.0'],
        'help': "Gain minimum requis pour effectuer une division. Une valeur plus élevée peut aider à prévenir le surapprentissage."
    },
    'min_child_weight': {
        'type': 'text_input',
        'args': ['min_child_weight', '0.001'],
        'help': "Poids minimum des échantillons dans une feuille. Une valeur plus élevée peut aider à prévenir le surapprentissage."
    },
    'min_child_samples': {
        'type': 'text_input',
        'args': ['min_child_samples', '20'],
        'help': "Nombre minimum d'échantillons requis pour être dans une feuille. Une valeur plus élevée peut aider à prévenir le surapprentissage."
    },
    'subsample': {
        'type': 'text_input',
        'args': ['subsample', '1.0'],
        'help': "Fraction des échantillons à utiliser pour l'entraînement de chaque arbre. Une valeur inférieure à 1.0 peut aider à prévenir le surapprentissage."
    },
    'subsample_freq': {
        'type': 'text_input',
        'args': ['subsample_freq', '0'],
        'help': "Fréquence de sous-échantillonnage. Une valeur de 0 signifie qu'il n'y a pas de sous-échantillonnage."
    },
    'colsample_bytree': {
        'type': 'text_input',
        'args': ['colsample_bytree', '1.0'],
        'help': "Fraction des caractéristiques à utiliser pour l'entraînement de chaque arbre. Une valeur inférieure à 1.0 peut aider à prévenir le surapprentissage."
    },
    'reg_alpha': {
        'type': 'text_input',
        'args': ['reg_alpha', '0.0'],
        'help': "Coefficient de régularisation L1. Une valeur plus élevée peut aider à prévenir le surapprentissage."
    },
    'reg_lambda': {
        'type': 'text_input',
        'args': ['reg_lambda', '0.0'],
        'help': "Coefficient de régularisation L2. Une valeur plus élevée peut aider à prévenir le surapprentissage."
    },
    'random_state': {
        'type': 'text_input',
        'args': ['random_state', '42'],
        'help': "Graine pour le générateur de nombres aléatoires. Utilisé pour garantir la reproductibilité des résultats."
    },
    'n_jobs': {
        'type': 'text_input',
        'args': ['n_jobs', '0'],
        'help': "Nombre de tâches à utiliser pour le calcul. -1 signifie utiliser tous les processeurs."
    },
    'importance_type': {
        'type': 'multiselect',
        'args': ['importance_type', ['split', 'gain']],
        'help': "Type d'importance des caractéristiques à utiliser. Peut être 'split' (nombre de fois où une caractéristique est utilisée pour diviser) ou 'gain' (gain total de la division utilisant la caractéristique)."
    },
    'is_unbalance': {
        'type': 'checkbox',
        'args': ['is_unbalance', False],
        'help': "Indique si le modèle doit être entraîné en tenant compte du déséquilibre des classes."
    },
    'scale_pos_weight': {
        'type': 'text_input',
        'args': ['scale_pos_weight', '0'],
        'help': "Poids des échantillons positifs. Utilisé pour équilibrer les classes dans le cas de données déséquilibrées."
    }
},
'ExplainableBoostingRegressor': {
    'max_bins': {
        'type': 'text_input',
        'args': ['max_bins', '1024'],
        'help': "Nombre maximum de bins pour la discrétisation des caractéristiques. Une valeur plus élevée peut améliorer la précision mais augmentera le temps de calcul."
    },
    'max_interaction_bins': {
        'type': 'text_input',
        'args': ['max_interaction_bins', '32'],
        'help': "Nombre maximum de bins pour les interactions entre les caractéristiques. Une valeur plus élevée peut améliorer la précision mais augmentera le temps de calcul."
    },
    'interactions': {
        'type': 'text_input',
        'args': ['interactions', '0.9'],
        'help': "Proportion des interactions entre les caractéristiques à considérer. Une valeur plus élevée peut améliorer la précision mais augmentera le temps de calcul."
    },
    'exclude': {
        'type': 'text_input',
        'args': ['exclude', '0'],
        'help': "Caractéristiques à exclure du modèle. Peut être une liste de noms de caractéristiques."
    },
    'validation_size': {
        'type': 'text_input',
        'args': ['validation_size', '0.15'],
        'help': "Proportion des données à utiliser pour la validation. Une valeur plus élevée peut améliorer la précision de la validation mais réduira la taille de l'ensemble d'entraînement."
    },
    'outer_bags': {
        'type': 'text_input',
        'args': ['outer_bags', '14'],
        'help': "Nombre de sacs externes à utiliser pour le bagging. Une valeur plus élevée peut améliorer la précision mais augmentera le temps de calcul."
    },
    'inner_bags': {
        'type': 'text_input',
        'args': ['inner_bags', '0'],
        'help': "Nombre de sacs internes à utiliser pour le bagging. Une valeur plus élevée peut améliorer la précision mais augmentera le temps de calcul."
    },
    'learning_rate': {
        'type': 'text_input',
        'args': ['learning_rate', '0.01'],
        'help': "Taux d'apprentissage utilisé pour l'entraînement. Un taux d'apprentissage plus élevé peut accélérer l'entraînement mais peut aussi entraîner une convergence moins stable."
    },
    'greedy_ratio': {
        'type': 'text_input',
        'args': ['greedy_ratio', '1.5'],
        'help': "Ratio de l'algorithme glouton à utiliser pour la sélection des caractéristiques. Une valeur plus élevée peut améliorer la précision mais augmentera le temps de calcul."
    },
    'cyclic_progress': {
        'type': 'checkbox',
        'args': ['cyclic_progress', True],
        'help': "Indique si la progression cyclique doit être utilisée pour l'entraînement."
    },
    'smoothing_rounds': {
        'type': 'text_input',
        'args': ['smoothing_rounds', '200'],
        'help': "Nombre de tours de lissage à effectuer. Une valeur plus élevée peut améliorer la précision mais augmentera le temps de calcul."
    },
    'interaction_smoothing_rounds': {
        'type': 'text_input',
        'args': ['interaction_smoothing_rounds', '50'],
        'help': "Nombre de tours de lissage des interactions à effectuer. Une valeur plus élevée peut améliorer la précision mais augmentera le temps de calcul."
    },
    'max_rounds': {
        'type': 'text_input',
        'args': ['max_rounds', '25000'],
        'help': "Nombre maximum de tours d'entraînement. Une valeur plus élevée peut améliorer la précision mais augmentera le temps de calcul."
    },
    'early_stopping_rounds': {
        'type': 'text_input',
        'args': ['early_stopping_rounds', '50'],
        'help': "Nombre de tours sans amélioration pour arrêter l'entraînement précocement."
    },
    'early_stopping_tolerance': {
        'type': 'text_input',
        'args': ['early_stopping_tolerance', '1e-05'],
        'help': "Tolérance pour l'arrêt précoce. Une valeur plus élevée peut arrêter l'entraînement plus tôt mais peut aussi entraîner une sous-optimisation."
    },
    'min_samples_leaf': {
        'type': 'text_input',
        'args': ['min_samples_leaf', '2'],
        'help': "Nombre minimum d'échantillons requis pour être dans une feuille. Une valeur plus élevée peut aider à prévenir le surapprentissage."
    },
    'min_hessian': {
        'type': 'text_input',
        'args': ['min_hessian', '0.0001'],
        'help': "Valeur minimale de la hessienne pour effectuer une division. Une valeur plus élevée peut aider à prévenir le surapprentissage."
    },
    'max_leaves': {
        'type': 'text_input',
        'args': ['max_leaves', '3'],
        'help': "Nombre maximum de feuilles dans un arbre. Une valeur plus élevée peut capturer des relations plus complexes mais peut aussi entraîner un surapprentissage."
    },
    'monotone_constraints': {
        'type': 'text_input',
        'args': ['monotone_constraints', '0'],
        'help': "Contraintes monotones à appliquer aux caractéristiques. Peut être une liste de contraintes."
    },
    'objective': {
        'type': 'multiselect',
        'args': ['objective', ['rmse', 'poisson_deviance', 'tweedie_deviance:variance_power=1.5', 'gamma_deviance', 'pseudo_huber:delta=1.0', 'rmse_log']],
        'help': "Fonction objectif à utiliser pour l'entraînement. Peut être 'rmse', 'poisson_deviance', 'tweedie_deviance', 'gamma_deviance', 'pseudo_huber', ou 'rmse_log'."
    },
    'n_jobs': {
        'type': 'text_input',
        'args': ['n_jobs', '-2'],
        'help': "Nombre de tâches à utiliser pour le calcul. -1 signifie utiliser tous les processeurs."
    },
    'random_state': {
        'type': 'text_input',
        'args': ['random_state', '42'],
        'help': "Graine pour le générateur de nombres aléatoires. Utilisé pour garantir la reproductibilité des résultats."
    }
},
'ExplainableBoostingClassifier': {
    'max_bins': {
        'type': 'text_input',
        'args': ['max_bins', '1024'],
        'help': "Nombre maximum de bins pour la discrétisation des caractéristiques. Une valeur plus élevée peut améliorer la précision mais augmentera le temps de calcul."
    },
    'max_interaction_bins': {
        'type': 'text_input',
        'args': ['max_interaction_bins', '32'],
        'help': "Nombre maximum de bins pour les interactions entre les caractéristiques. Une valeur plus élevée peut améliorer la précision mais augmentera le temps de calcul."
    },
    'interactions': {
        'type': 'text_input',
        'args': ['interactions', '0.9'],
        'help': "Proportion des interactions entre les caractéristiques à considérer. Une valeur plus élevée peut améliorer la précision mais augmentera le temps de calcul."
    },
    'exclude': {
        'type': 'text_input',
        'args': ['exclude', '0'],
        'help': "Caractéristiques à exclure du modèle. Peut être une liste de noms de caractéristiques."
    },
    'validation_size': {
        'type': 'text_input',
        'args': ['validation_size', '0.15'],
        'help': "Proportion des données à utiliser pour la validation. Une valeur plus élevée peut améliorer la précision de la validation mais réduira la taille de l'ensemble d'entraînement."
    },
    'outer_bags': {
        'type': 'text_input',
        'args': ['outer_bags', '14'],
        'help': "Nombre de sacs externes à utiliser pour le bagging. Une valeur plus élevée peut améliorer la précision mais augmentera le temps de calcul."
    },
    'inner_bags': {
        'type': 'text_input',
        'args': ['inner_bags', '0'],
        'help': "Nombre de sacs internes à utiliser pour le bagging. Une valeur plus élevée peut améliorer la précision mais augmentera le temps de calcul."
    },
    'learning_rate': {
        'type': 'text_input',
        'args': ['learning_rate', '0.01'],
        'help': "Taux d'apprentissage utilisé pour l'entraînement. Un taux d'apprentissage plus élevé peut accélérer l'entraînement mais peut aussi entraîner une convergence moins stable."
    },
    'greedy_ratio': {
        'type': 'text_input',
        'args': ['greedy_ratio', '1.5'],
        'help': "Ratio de l'algorithme glouton à utiliser pour la sélection des caractéristiques. Une valeur plus élevée peut améliorer la précision mais augmentera le temps de calcul."
    },
    'cyclic_progress': {
        'type': 'checkbox',
        'args': ['cyclic_progress', True],
        'help': "Indique si la progression cyclique doit être utilisée pour l'entraînement."
    },
    'smoothing_rounds': {
        'type': 'text_input',
        'args': ['smoothing_rounds', '200'],
        'help': "Nombre de tours de lissage à effectuer. Une valeur plus élevée peut améliorer la précision mais augmentera le temps de calcul."
    },
    'interaction_smoothing_rounds': {
        'type': 'text_input',
        'args': ['interaction_smoothing_rounds', '50'],
        'help': "Nombre de tours de lissage des interactions à effectuer. Une valeur plus élevée peut améliorer la précision mais augmentera le temps de calcul."
    },
    'max_rounds': {
        'type': 'text_input',
        'args': ['max_rounds', '25000'],
        'help': "Nombre maximum de tours d'entraînement. Une valeur plus élevée peut améliorer la précision mais augmentera le temps de calcul."
    },
    'early_stopping_rounds': {
        'type': 'text_input',
        'args': ['early_stopping_rounds', '50'],
        'help': "Nombre de tours sans amélioration pour arrêter l'entraînement précocement."
    },
    'early_stopping_tolerance': {
        'type': 'text_input',
        'args': ['early_stopping_tolerance', '1e-05'],
        'help': "Tolérance pour l'arrêt précoce. Une valeur plus élevée peut arrêter l'entraînement plus tôt mais peut aussi entraîner une sous-optimisation."
    },
    'min_samples_leaf': {
        'type': 'text_input',
        'args': ['min_samples_leaf', '2'],
        'help': "Nombre minimum d'échantillons requis pour être dans une feuille. Une valeur plus élevée peut aider à prévenir le surapprentissage."
    },
    'min_hessian': {
        'type': 'text_input',
        'args': ['min_hessian', '0.0001'],
        'help': "Valeur minimale de la hessienne pour effectuer une division. Une valeur plus élevée peut aider à prévenir le surapprentissage."
    },
    'max_leaves': {
        'type': 'text_input',
        'args': ['max_leaves', '3'],
        'help': "Nombre maximum de feuilles dans un arbre. Une valeur plus élevée peut capturer des relations plus complexes mais peut aussi entraîner un surapprentissage."
    },
    'monotone_constraints': {
        'type': 'text_input',
        'args': ['monotone_constraints', '0'],
        'help': "Contraintes monotones à appliquer aux caractéristiques. Peut être une liste de contraintes."
    },
    'n_jobs': {
        'type': 'text_input',
        'args': ['n_jobs', '-2'],
        'help': "Nombre de tâches à utiliser pour le calcul. -1 signifie utiliser tous les processeurs."
    },
    'random_state': {
        'type': 'text_input',
        'args': ['random_state', '42'],
        'help': "Graine pour le générateur de nombres aléatoires. Utilisé pour garantir la reproductibilité des résultats."
    }
},
'XGBRegressor': {
    'n_estimators': {
        'type': 'text_input',
        'args': ['n_estimators', '100'],
        'help': "Nombre d'arbres à construire lors de l'entraînement. Plus la valeur est élevée, plus le modèle sera entraîné longtemps."
    },
    'max_depth': {
        'type': 'text_input',
        'args': ['max_depth', '6'],
        'help': "Profondeur maximale des arbres de décision. Une valeur plus élevée peut capturer des relations plus complexes mais peut aussi entraîner un surapprentissage."
    },
    'max_leaves': {
        'type': 'text_input',
        'args': ['max_leaves', '0'],
        'help': "Nombre maximum de feuilles dans un arbre. Une valeur plus élevée peut capturer des relations plus complexes mais peut aussi entraîner un surapprentissage."
    },
    'max_bin': {
        'type': 'text_input',
        'args': ['max_bin', '256'],
        'help': "Nombre maximum de bins pour la discrétisation des caractéristiques. Une valeur plus élevée peut améliorer la précision mais augmentera le temps de calcul."
    },
    'grow_policy': {
        'type': 'multiselect',
        'args': ['grow_policy', ['grow_policy', 'depthwise']],
        'help': "Politique de croissance des arbres. Peut être 'grow_policy' (croissance par niveau) ou 'depthwise' (croissance en profondeur)."
    },
    'learning_rate': {
        'type': 'text_input',
        'args': ['learning_rate', '0.3'],
        'help': "Taux d'apprentissage utilisé pour l'entraînement. Un taux d'apprentissage plus élevé peut accélérer l'entraînement mais peut aussi entraîner une convergence moins stable."
    },
    'verbosity': {
        'type': 'text_input',
        'args': ['verbosity', '1'],
        'help': "Niveau de verbosité pour la sortie de débogage. Plus la valeur est élevée, plus la sortie contiendra de détails."
    },
    'booster': {
        'type': 'multiselect',
        'args': ['booster', ['gbtree', 'gblinear','dart']],
        'help': "Type de booster à utiliser. Peut être 'gbtree' (arbres de décision), 'gblinear' (régression linéaire) ou 'dart' (Dropouts meet Multiple Additive Regression Trees)."
    },
    'n_jobs': {
        'type': 'text_input',
        'args': ['n_jobs', '-1'],
        'help': "Nombre de tâches à utiliser pour le calcul. -1 signifie utiliser tous les processeurs."
    },
    'gamma': {
        'type': 'text_input',
        'args': ['gamma', '0'],
        'help': "Paramètre de régularisation utilisé pour contrôler la complexité du modèle. Une valeur plus élevée peut aider à prévenir le surapprentissage."
    },
    'min_child_weight': {
        'type': 'text_input',
        'args': ['min_child_weight', '1'],
        'help': "Poids minimum des échantillons dans une feuille. Une valeur plus élevée peut aider à prévenir le surapprentissage."
    },
    'max_delta_step': {
        'type': 'text_input',
        'args': ['max_delta_step', '0'],
        'help': "Taille maximale de la mise à jour de la fonction de perte. Une valeur plus élevée peut améliorer la précision mais augmentera le temps de calcul."
    },
    'subsample': {
        'type': 'text_input',
        'args': ['subsample', '1'],
        'help': "Fraction des échantillons à utiliser pour l'entraînement de chaque arbre. Une valeur inférieure à 1.0 peut aider à prévenir le surapprentissage."
    },
    'sampling_method': {
        'type': 'multiselect',
        'args': ['sampling_method', ['uniform','gradient_based']],
        'help': "Méthode de sous-échantillonnage à utiliser. Peut être 'uniform' (sous-échantillonnage uniforme) ou 'gradient_based' (sous-échantillonnage basé sur les gradients)."
    },
    'colsample_bytree': {
        'type': 'text_input',
        'args': ['colsample_bytree', '1'],
        'help': "Fraction des caractéristiques à utiliser pour l'entraînement de chaque arbre. Une valeur inférieure à 1.0 peut aider à prévenir le surapprentissage."
    },
    'colsample_bylevel': {
        'type': 'text_input',
        'args': ['colsample_bylevel', '1'],
        'help': "Fraction des caractéristiques à utiliser pour l'entraînement de chaque niveau. Une valeur inférieure à 1.0 peut aider à prévenir le surapprentissage."
    },
    'colsample_bynode': {
        'type': 'text_input',
        'args': ['colsample_bynode', '1'],
        'help': "Fraction des caractéristiques à utiliser pour l'entraînement de chaque nœud. Une valeur inférieure à 1.0 peut aider à prévenir le surapprentissage."
    },
    'reg_alpha': {
        'type': 'text_input',
        'args': ['reg_alpha', '0'],
        'help': "Coefficient de régularisation L1. Une valeur plus élevée peut aider à prévenir le surapprentissage."
    },
    'reg_lambda': {
        'type': 'text_input',
        'args': ['reg_lambda', '1'],
        'help': "Coefficient de régularisation L2. Une valeur plus élevée peut aider à prévenir le surapprentissage."
    },
    'scale_pos_weight': {
        'type': 'text_input',
        'args': ['scale_pos_weight', '1'],
        'help': "Poids des échantillons positifs. Utilisé pour équilibrer les classes dans le cas de données déséquilibrées."
    },
    'base_score': {
        'type': 'text_input',
        'args': ['base_score', '0.5'],
        'help': "Score de base pour les prédictions. Utilisé pour équilibrer les scores de prédiction."
    },
    'random_state': {
        'type': 'text_input',
        'args': ['random_state', '0'],
        'help': "Graine pour le générateur de nombres aléatoires. Utilisé pour garantir la reproductibilité des résultats."
    },
    'missing': {
        'type': 'multiselect',
        'args': ['missing',['missing', 'nan']],
        'help': "Mode de gestion des valeurs manquantes. Peut être 'missing' (ignorer les valeurs manquantes) ou 'nan' (traiter les valeurs manquantes comme des valeurs numériques)."
    },
    'num_parallel_tree': {
        'type': 'text_input',
        'args': ['num_parallel_tree', '1'],
        'help': "Nombre d'arbres à construire en parallèle. Plus la valeur est élevée, plus le modèle sera entraîné rapidement mais peut aussi entraîner une utilisation plus élevée des ressources."
    },
    'device': {
        'type': 'multiselect',
        'args': ['device', ['cpu','cuda','gpu']],
        'help': "Dispositif de calcul à utiliser pour l'entraînement. Peut être 'cpu' (processeur), 'cuda' (GPU Nvidia) ou 'gpu' (GPU)."
    },
    'validate_parameters': {
        'type': 'checkbox',
        'args': ['validate_parameters', True],
        'help': "Indique si les paramètres doivent être validés après l'entraînement."
    },
    'enable_categorical': {
        'type': 'checkbox',
        'args': ['enable_categorical', False],
        'help': "Indique si la prise en charge des caractéristiques catégorielles doit être activée."
    },
    'max_cat_to_onehot': {
        'type': 'text_input',
        'args': ['max_cat_to_onehot', '4'],
        'help': "Nombre maximum de caractéristiques catégorielles à convertir en une seule caractéristique one-hot. Une valeur plus élevée peut améliorer la précision mais augmentera le temps de calcul."
    },
    'max_cat_threshold': {
        'type': 'text_input',
        'args': ['max_cat_threshold', '64'],
        'help': "Seuil de fréquence pour la conversion des caractéristiques catégorielles en une seule caractéristique one-hot. Une valeur plus élevée peut améliorer la précision mais augmentera le temps de calcul."
    },
    'early_stopping_rounds': {
        'type': 'text_input',
        'args': ['early_stopping_rounds', '10'],
        'help': "Nombre de tours sans amélioration pour arrêter l'entraînement précocement."
    }
},
'XGBClassifier': {
    'n_estimators': {
        'type': 'text_input',
        'args': ['n_estimators', '100'],
        'help': "Nombre d'arbres à construire lors de l'entraînement. Plus la valeur est élevée, plus le modèle sera entraîné longtemps."
    },
    'max_depth': {
        'type': 'text_input',
        'args': ['max_depth', '6'],
        'help': "Profondeur maximale des arbres de décision. Une valeur plus élevée peut capturer des relations plus complexes mais peut aussi entraîner un surapprentissage."
    },
    'max_leaves': {
        'type': 'text_input',
        'args': ['max_leaves', '0'],
        'help': "Nombre maximum de feuilles dans un arbre. Une valeur plus élevée peut capturer des relations plus complexes mais peut aussi entraîner un surapprentissage."
    },
    'max_bin': {
        'type': 'text_input',
        'args': ['max_bin', '256'],
        'help': "Nombre maximum de bins pour la discrétisation des caractéristiques. Une valeur plus élevée peut améliorer la précision mais augmentera le temps de calcul."
    },
    'grow_policy': {
        'type': 'multiselect',
        'args': ['grow_policy', ['grow_policy', 'depthwise']],
        'help': "Politique de croissance des arbres. Peut être 'grow_policy' (croissance par niveau) ou 'depthwise' (croissance en profondeur)."
    },
    'learning_rate': {
        'type': 'text_input',
        'args': ['learning_rate', '0.3'],
        'help': "Taux d'apprentissage utilisé pour l'entraînement. Un taux d'apprentissage plus élevé peut accélérer l'entraînement mais peut aussi entraîner une convergence moins stable."
    },
    'verbosity': {
        'type': 'text_input',
        'args': ['verbosity', '1'],
        'help': "Niveau de verbosité pour la sortie de débogage. Plus la valeur est élevée, plus la sortie contiendra de détails."
    },
    'booster': {
        'type': 'multiselect',
        'args': ['booster', ['gbtree', 'gblinear','dart']],
        'help': "Type de booster à utiliser. Peut être 'gbtree' (arbres de décision), 'gblinear' (régression linéaire) ou 'dart' (Dropouts meet Multiple Additive Regression Trees)."
    },
    'n_jobs': {
        'type': 'text_input',
        'args': ['n_jobs', '-1'],
        'help': "Nombre de tâches à utiliser pour le calcul. -1 signifie utiliser tous les processeurs."
    },
    'gamma': {
        'type': 'text_input',
        'args': ['gamma', '0'],
        'help': "Paramètre de régularisation utilisé pour contrôler la complexité du modèle. Une valeur plus élevée peut aider à prévenir le surapprentissage."
    },
    'min_child_weight': {
        'type': 'text_input',
        'args': ['min_child_weight', '1'],
        'help': "Poids minimum des échantillons dans une feuille. Une valeur plus élevée peut aider à prévenir le surapprentissage."
    },
    'max_delta_step': {
        'type': 'text_input',
        'args': ['max_delta_step', '0'],
        'help': "Taille maximale de la mise à jour de la fonction de perte. Une valeur plus élevée peut améliorer la précision mais augmentera le temps de calcul."
    },
    'subsample': {
        'type': 'text_input',
        'args': ['subsample', '1'],
        'help': "Fraction des échantillons à utiliser pour l'entraînement de chaque arbre. Une valeur inférieure à 1.0 peut aider à prévenir le surapprentissage."
    },
    'sampling_method': {
        'type': 'multiselect',
        'args': ['sampling_method', ['uniform','gradient_based']],
        'help': "Méthode de sous-échantillonnage à utiliser. Peut être 'uniform' (sous-échantillonnage uniforme) ou 'gradient_based' (sous-échantillonnage basé sur les gradients)."
    },
    'colsample_bytree': {
        'type': 'text_input',
        'args': ['colsample_bytree', '1'],
        'help': "Fraction des caractéristiques à utiliser pour l'entraînement de chaque arbre. Une valeur inférieure à 1.0 peut aider à prévenir le surapprentissage."
    },
    'colsample_bylevel': {
        'type': 'text_input',
        'args': ['colsample_bylevel', '1'],
        'help': "Fraction des caractéristiques à utiliser pour l'entraînement de chaque niveau. Une valeur inférieure à 1.0 peut aider à prévenir le surapprentissage."
    },
    'colsample_bynode': {
        'type': 'text_input',
        'args': ['colsample_bynode', '1'],
        'help': "Fraction des caractéristiques à utiliser pour l'entraînement de chaque nœud. Une valeur inférieure à 1.0 peut aider à prévenir le surapprentissage."
    },
    'reg_alpha': {
        'type': 'text_input',
        'args': ['reg_alpha', '0'],
        'help': "Coefficient de régularisation L1. Une valeur plus élevée peut aider à prévenir le surapprentissage."
    },
    'reg_lambda': {
        'type': 'text_input',
        'args': ['reg_lambda', '1'],
        'help': "Coefficient de régularisation L2. Une valeur plus élevée peut aider à prévenir le surapprentissage."
    },
    'scale_pos_weight': {
        'type': 'text_input',
        'args': ['scale_pos_weight', '1'],
        'help': "Poids des échantillons positifs. Utilisé pour équilibrer les classes dans le cas de données déséquilibrées."
    },
    'base_score': {
        'type': 'text_input',
        'args': ['base_score', '0.5'],
        'help': "Score de base pour les prédictions. Utilisé pour équilibrer les scores de prédiction."
    },
    'random_state': {
        'type': 'text_input',
        'args': ['random_state', '0'],
        'help': "Graine pour le générateur de nombres aléatoires. Utilisé pour garantir la reproductibilité des résultats."
    },
    'missing': {
        'type': 'multiselect',
        'args': ['missing',['missing', 'nan']],
        'help': "Mode de gestion des valeurs manquantes. Peut être 'missing' (ignorer les valeurs manquantes) ou 'nan' (traiter les valeurs manquantes comme des valeurs numériques)."
    },
    'num_parallel_tree': {
        'type': 'text_input',
        'args': ['num_parallel_tree', '1'],
        'help': "Nombre d'arbres à construire en parallèle. Plus la valeur est élevée, plus le modèle sera entraîné rapidement mais peut aussi entraîner une utilisation plus élevée des ressources."
    },
    'device': {
        'type': 'multiselect',
        'args': ['device', ['cpu','cuda','gpu']],
        'help': "Dispositif de calcul à utiliser pour l'entraînement. Peut être 'cpu' (processeur), 'cuda' (GPU Nvidia) ou 'gpu' (GPU)."
    },
    'validate_parameters': {
        'type': 'checkbox',
        'args': ['validate_parameters', True],
        'help': "Indique si les paramètres doivent être validés après l'entraînement."
    },
    'enable_categorical': {
        'type': 'checkbox',
        'args': ['enable_categorical', False],
        'help': "Indique si la prise en charge des caractéristiques catégorielles doit être activée."
    },
    'max_cat_to_onehot': {
        'type': 'text_input',
        'args': ['max_cat_to_onehot', '4'],
        'help': "Nombre maximum de caractéristiques catégorielles à convertir en une seule caractéristique one-hot. Une valeur plus élevée peut améliorer la précision mais augmentera le temps de calcul."
    },
    'max_cat_threshold': {
        'type': 'text_input',
        'args': ['max_cat_threshold', '64'],
        'help': "Seuil de fréquence pour la conversion des caractéristiques catégorielles en une seule caractéristique one-hot. Une valeur plus élevée peut améliorer la précision mais augmentera le temps de calcul."
    },
    'early_stopping_rounds': {
        'type': 'text_input',
        'args': ['early_stopping_rounds', '10'],
        'help': "Nombre de tours sans amélioration pour arrêter l'entraînement précocement."
    }
}
}

model_explanation = {
"LinearRegression" : """
Le modèle de régression linéaire prédit la variable dépendante $y$ en fonction des variables indépendantes $X$. Le modèle peut être exprimé mathématiquement comme suit :

$$
\\hat{y} = \\mathbf{X}\\boldsymbol{\\beta} + \\boldsymbol{\\epsilon}
$$

Où :
- $\\hat{y}$ est la valeur prédite.
- $\\mathbf{X}$ est la matrice des variables explicatives (variables indépendantes), de dimensions $n \\times p$, où $n$ est le nombre d'observations et $p$ est le nombre de caractéristiques.
- $\\boldsymbol{\\beta}$ est le vecteur des coefficients (poids) associés aux caractéristiques, de dimensions $p \\times 1$.
- $\\boldsymbol{\\epsilon}$ est le terme d'erreur ou les résidus, qui représentent la différence entre les valeurs observées et les valeurs prédites.

Sous sa forme développée, pour une seule observation, le modèle peut être écrit comme suit :

$$
\\hat{y}_i = \\beta_0 + \\beta_1 x_{i1} + \\beta_2 x_{i2} + \\cdots + \\beta_p x_{ip} + \\epsilon_i
$$

Où :
- $\\hat{y}_i$ est la valeur prédite pour la $i$-ème observation.
- $\\beta_0$ est l'intercept (terme constant).
- $\\beta_1, \\beta_2, \\dots, \\beta_p$ sont les coefficients pour chaque caractéristique $x_{i1}, x_{i2}, \\dots, x_{ip}$.
- $\\epsilon_i$ est le résidu pour la $i$-ème observation.
""",
"LogisticRegression":"""
Le modèle de régression logistique est utilisé pour prédire la probabilité qu'une observation appartienne à une classe particulière, en fonction des variables explicatives $X$. Le modèle peut être exprimé mathématiquement comme suit :

$$
P(y = 1 | \mathbf{X}) = \\frac{1}{1 + e^{-(\\mathbf{X}\\boldsymbol{\\beta})}}
$$

Où :
- $P(y = 1 | \\mathbf{X})$ est la probabilité que l'événement $y = 1$ se produise, étant donné les variables explicatives $X$.
- $\\mathbf{X}$ est la matrice des variables explicatives, de dimensions $n \\times p$, où $n$ est le nombre d'observations et $p$ est le nombre de caractéristiques.
- $\\boldsymbol{\\beta}$ est le vecteur des coefficients associés aux caractéristiques, de dimensions $p \\times 1$.

Le modèle peut également être exprimé sous la forme log-odds (logarithme du rapport des cotes) :

$$
\\log\\left(\\frac{P(y = 1 | \\mathbf{X})}{1 - P(y = 1 | \\mathbf{X})}\\right) = \\mathbf{X}\\boldsymbol{\\beta}
$$

Où :
- $\\log\\left(\\frac{P(y = 1 | \\mathbf{X})}{1 - P(y = 1 | \\mathbf{X})}\\right)$ est le logarithme du rapport des cotes, qui est une transformation linéaire des variables explicatives.
- $\\mathbf{X}\\boldsymbol{\\beta}$ représente la combinaison linéaire des variables explicatives et de leurs coefficients respectifs.

Ce modèle est utilisé principalement pour les problèmes de classification binaire.
""",
"Ridge":"""
Le modèle de régression Ridge est une extension de la régression linéaire classique, qui ajoute un terme de régularisation L2 pour pénaliser la taille des coefficients, dans le but de réduire le surapprentissage (overfitting). Le modèle peut être exprimé mathématiquement comme suit :

$$
\\hat{y} = \\mathbf{X}\\boldsymbol{\\beta} + \\boldsymbol{\\epsilon}
$$

Où :
- $\\hat{y}$ est la valeur prédite.
- $\\mathbf{X}$ est la matrice des variables explicatives (variables indépendantes), de dimensions $n \\times p$, où $n$ est le nombre d'observations et $p$ est le nombre de caractéristiques.
- $\\boldsymbol{\\beta}$ est le vecteur des coefficients (poids) associés aux caractéristiques, de dimensions $p \\times 1$.
- $\\boldsymbol{\\epsilon}$ est le terme d'erreur ou les résidus, représentant la différence entre les valeurs observées et les valeurs prédites.

Dans la régression Ridge, la fonction de coût (ou fonction de perte) à minimiser est la suivante :

$$
J(\\boldsymbol{\\beta}) = \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2 + \\lambda \\sum_{j=1}^{p} \\beta_j^2
$$

Où :
- $J(\\boldsymbol{\\beta})$ est la fonction de coût à minimiser.
- $\\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2$ est la somme des erreurs quadratiques entre les valeurs observées $y_i$ et les valeurs prédites $\\hat{y}_i$.
- $\\lambda$ est le paramètre de régularisation qui contrôle l'importance de la pénalisation L2.
- $\\sum_{j=1}^{p} \\beta_j^2$ est la somme des carrés des coefficients $\\beta_j$, ajoutée pour la régularisation.

La régularisation Ridge aide à prévenir le surapprentissage en imposant une contrainte sur les valeurs des coefficients, les forçant à rester petits, ce qui peut améliorer la généralisation du modèle.
""",
"Lasso":"""
Le modèle de régression Lasso est une variante de la régression linéaire qui ajoute un terme de régularisation L1. Cette régularisation favorise des modèles plus simples en réduisant certains coefficients à zéro, ce qui peut être interprété comme une forme de sélection de variables. Le modèle peut être exprimé mathématiquement comme suit :

$$
\\hat{y} = \\mathbf{X}\\boldsymbol{\\beta} + \\boldsymbol{\\epsilon}
$$

Où :
- $\\hat{y}$ est la valeur prédite.
- $\\mathbf{X}$ est la matrice des variables explicatives (variables indépendantes), de dimensions $n \\times p$, où $n$ est le nombre d'observations et $p$ est le nombre de caractéristiques.
- $\\boldsymbol{\\beta}$ est le vecteur des coefficients (poids) associés aux caractéristiques, de dimensions $p \\times 1$.
- $\\boldsymbol{\\epsilon}$ est le terme d'erreur ou les résidus, représentant la différence entre les valeurs observées et les valeurs prédites.

Dans la régression Lasso, la fonction de coût (ou fonction de perte) à minimiser est la suivante :

$$
J(\\boldsymbol{\\beta}) = \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2 + \\lambda \\sum_{j=1}^{p} |\\beta_j|
$$

Où :
- $J(\\boldsymbol{\\beta})$ est la fonction de coût à minimiser.
- $\\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2$ est la somme des erreurs quadratiques entre les valeurs observées $y_i$ et les valeurs prédites $\\hat{y}_i$.
- $\\lambda$ est le paramètre de régularisation qui contrôle l'importance de la pénalisation L1.
- $\\sum_{j=1}^{p} |\\beta_j|$ est la somme des valeurs absolues des coefficients $\\beta_j$, ajoutée pour la régularisation.

La régularisation Lasso aide à prévenir le surapprentissage et peut entraîner la suppression (réduction à zéro) de certains coefficients, ce qui rend le modèle plus interprétable en sélectionnant uniquement les caractéristiques les plus importantes.
""",
"ElasticNet": """
Le modèle ElasticNet est une méthode de régression linéaire qui combine les régularisations L1 (comme dans Lasso) et L2 (comme dans Ridge). Cette combinaison permet de bénéficier à la fois des propriétés de sélection de variables du Lasso et de la stabilisation des coefficients apportée par Ridge. Le modèle peut être exprimé mathématiquement comme suit :

$$
\\hat{y} = \\mathbf{X}\\boldsymbol{\\beta} + \\boldsymbol{\\epsilon}
$$

Où :
- $\\hat{y}$ est la valeur prédite.
- $\\mathbf{X}$ est la matrice des variables explicatives (variables indépendantes), de dimensions $n \\times p$, où $n$ est le nombre d'observations et $p$ est le nombre de caractéristiques.
- $\\boldsymbol{\\beta}$ est le vecteur des coefficients (poids) associés aux caractéristiques, de dimensions $p \\times 1$.
- $\\boldsymbol{\\epsilon}$ est le terme d'erreur ou les résidus, représentant la différence entre les valeurs observées et les valeurs prédites.

Dans la régression ElasticNet, la fonction de coût (ou fonction de perte) à minimiser est la suivante :

$$
J(\\boldsymbol{\\beta}) = \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2 + \\lambda_1 \\sum_{j=1}^{p} |\\beta_j| + \\lambda_2 \\sum_{j=1}^{p} \\beta_j^2
$$

Où :
- $J(\\boldsymbol{\\beta})$ est la fonction de coût à minimiser.
- $\\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2$ est la somme des erreurs quadratiques entre les valeurs observées $y_i$ et les valeurs prédites $\\hat{y}_i$.
- $\\lambda_1$ est le paramètre de régularisation qui contrôle l'importance de la pénalisation L1 (terme Lasso).
- $\\lambda_2$ est le paramètre de régularisation qui contrôle l'importance de la pénalisation L2 (terme Ridge).
- $\\sum_{j=1}^{p} |\\beta_j|$ est la somme des valeurs absolues des coefficients $\\beta_j$ (terme L1).
- $\\sum_{j=1}^{p} \\beta_j^2$ est la somme des carrés des coefficients $\\beta_j$ (terme L2).

Le modèle ElasticNet est particulièrement utile lorsque le nombre de variables explicatives est grand et que certaines de ces variables sont fortement corrélées. Il permet une sélection de variables tout en maintenant une certaine robustesse dans la prédiction.
""",
"BayesianRidge":"""
Le modèle Bayesian Ridge est une forme de régression linéaire qui applique un cadre bayésien pour estimer les coefficients. Contrairement à la régression Ridge classique, ce modèle introduit une distribution de probabilité sur les coefficients, ce qui permet de capturer l'incertitude associée aux estimations. Le modèle peut être exprimé mathématiquement comme suit :

$$
\\hat{y} = \\mathbf{X}\\boldsymbol{\\beta} + \\boldsymbol{\\epsilon}
$$

Où :
- $\\hat{y}$ est la valeur prédite.
- $\\mathbf{X}$ est la matrice des variables explicatives (variables indépendantes), de dimensions $n \\times p$, où $n$ est le nombre d'observations et $p$ est le nombre de caractéristiques.
- $\\boldsymbol{\\beta}$ est le vecteur des coefficients (poids) associés aux caractéristiques, de dimensions $p \\times 1$.
- $\\boldsymbol{\\epsilon}$ est le terme d'erreur ou les résidus, représentant la différence entre les valeurs observées et les valeurs prédites.

Dans le cadre bayésien, les coefficients $\\boldsymbol{\\beta}$ sont modélisés comme des variables aléatoires avec une distribution normale :

$$
\\boldsymbol{\\beta} \sim \\mathcal{N}(\\mathbf{0}, \\lambda^{-1} \\mathbf{I})
$$

Où :
- $\\mathcal{N}(\\mathbf{0}, \\lambda^{-1} \\mathbf{I})$ est la distribution normale centrée sur $\\mathbf{0}$ avec une matrice de covariance $\\lambda^{-1} \\mathbf{I}$.
- $\\lambda$ est le paramètre de précision (inverse de la variance) qui contrôle la régularisation.

Le modèle suppose également que le terme d'erreur $\\boldsymbol{\\epsilon}$ suit une distribution normale :

$$
\\boldsymbol{\\epsilon} \sim \\mathcal{N}(\\mathbf{0}, \\alpha^{-1} \\mathbf{I})
$$

Où :
- $\\alpha$ est un autre paramètre de précision qui contrôle l'écart-type des résidus.

En combinant ces deux distributions, le modèle Bayesian Ridge estime les paramètres $\\alpha$ et $\\lambda$ à partir des données et utilise ces estimations pour inférer les coefficients $\\boldsymbol{\\beta}$, en tenant compte de l'incertitude.

Le modèle Bayesian Ridge est particulièrement utile lorsque l'on souhaite obtenir des intervalles de confiance pour les coefficients ou lorsque l'on travaille avec des données complexes et incertaines.
""",
"SGDRegressor":r"""
Le SGDRegressor (Stochastic Gradient Descent Regressor) est un modèle de régression linéaire qui utilise la descente de gradient stochastique pour optimiser ses paramètres. Il est particulièrement utile pour les grands ensembles de données et l'apprentissage en ligne.
Le modèle peut être exprimé mathématiquement comme suit :
$$
\hat{y} = \mathbf{X}\boldsymbol{\beta}
$$
Où :

- $\hat{y}$ est la valeur prédite.
- $\mathbf{X}$ est la matrice des variables explicatives.
- $\boldsymbol{\beta}$ est le vecteur des coefficients à optimiser.

L'algorithme SGD met à jour les coefficients de manière itérative en utilisant la règle suivante :
$$
\boldsymbol{\beta}_{t+1} = \boldsymbol{\beta}_t - \eta \nabla L(\boldsymbol{\beta}_t)
$$
Où :

- $\boldsymbol{\beta}_t$ sont les coefficients à l'itération $t$.
- $\eta$ est le taux d'apprentissage.
- $\nabla L(\boldsymbol{\beta}_t)$ est le gradient de la fonction de perte par rapport aux coefficients.

Pour la régression, la fonction de perte typique est l'erreur quadratique moyenne :
$$
L(\boldsymbol{\beta}) = \frac{1}{2n} \sum_{i=1}^n (y_i - \mathbf{x}_i^T\boldsymbol{\beta})^2
$$
Le SGDRegressor peut également utiliser différentes fonctions de perte et de régularisation, comme la régularisation L1 (Lasso) ou L2 (Ridge), exprimées respectivement par :
$$
L1: \lambda \sum_{j=1}^p |\beta_j| \quad \text{et} \quad L2: \lambda \sum_{j=1}^p \beta_j^2
$$
Où $\lambda$ est le paramètre de régularisation.
""",
"PassiveAggressiveRegressor":r"""
Le PassiveAggressiveRegressor est un algorithme d'apprentissage en ligne pour la régression. Il est conçu pour être "passif" lorsque la prédiction est correcte, mais "agressif" lorsqu'une erreur de prédiction se produit.
Le modèle peut être exprimé mathématiquement comme suit :
$$
\hat{y} = \mathbf{x}^T\boldsymbol{\beta}
$$
Où :

- $\hat{y}$ est la valeur prédite.
- $\mathbf{x}$ est le vecteur des caractéristiques d'entrée.
- $\boldsymbol{\beta}$ est le vecteur des coefficients du modèle.

L'algorithme met à jour les coefficients de manière itérative en utilisant la règle suivante :
$$
\boldsymbol{\beta}_{t+1} = \boldsymbol{\beta}_t + \tau_t \mathbf{x}_t
$$
Où :

- $\boldsymbol{\beta}_t$ sont les coefficients à l'itération $t$.
- $\tau_t$ est le taux de mise à jour à l'itération $t$.
- $\mathbf{x}_t$ est le vecteur des caractéristiques à l'itération $t$.

Le taux de mise à jour $\tau_t$ est calculé comme suit :
$$
\tau_t = \min\left(C, \frac{l_t}{\|\mathbf{x}_t\|^2}\right)
$$
Où :

- $C$ est le paramètre d'agressivité (un hyperparamètre du modèle).
- $l_t$ est la perte à l'itération $t$.
- $\|\mathbf{x}_t\|^2$ est la norme au carré du vecteur des caractéristiques.

La perte $l_t$ est définie comme :
$$
l_t = \max(0, |y_t - \hat{y}_t| - \epsilon)
$$
Où :

- $y_t$ est la vraie valeur à l'itération $t$.
- $\hat{y}_t$ est la valeur prédite à l'itération $t$.
- $\epsilon$ est la marge d'erreur tolérée (un autre hyperparamètre du modèle).

Le modèle reste "passif" (pas de mise à jour) si la perte est nulle, c'est-à-dire si la prédiction est dans la marge d'erreur tolérée. Sinon, il devient "agressif" et met à jour les coefficients proportionnellement à l'erreur commise.
""",
"RandomForestRegressor":r"""
Le RandomForestRegressor est un modèle d'ensemble qui combine plusieurs arbres de décision pour effectuer des prédictions de régression. Il utilise la technique du bagging (bootstrap aggregating) pour améliorer la stabilité et la précision des prédictions.
Le modèle peut être exprimé mathématiquement comme suit :
$$
\hat{y} = \frac{1}{B} \sum_{b=1}^B f_b(\mathbf{x})
$$
Où :

- $\hat{y}$ est la valeur prédite finale.
- $B$ est le nombre total d'arbres dans la forêt.
- $f_b(\mathbf{x})$ est la prédiction du $b$-ème arbre pour le vecteur d'entrée $\mathbf{x}$.

Chaque arbre $f_b$ est construit à partir d'un échantillon bootstrap $\mathcal{D}_b$ tiré avec remplacement de l'ensemble de données original $\mathcal{D}$. La taille de $\mathcal{D}_b$ est généralement la même que celle de $\mathcal{D}$.
Pour chaque nœud de l'arbre, un sous-ensemble aléatoire de $m$ caractéristiques est sélectionné parmi les $p$ caractéristiques disponibles, où typiquement $m = \sqrt{p}$ pour la régression.
La division à chaque nœud est choisie pour minimiser l'erreur quadratique moyenne (MSE) :
$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
$$
Où $n$ est le nombre d'échantillons dans le nœud, $y_i$ sont les vraies valeurs et $\hat{y}_i$ sont les prédictions.
L'importance des caractéristiques peut être estimée en calculant la diminution moyenne de l'impureté (MDI) pour chaque caractéristique $j$ :
$$
\text{MDI}j = \frac{1}{B} \sum{b=1}^B \sum_{t \in T_b} p(t) \Delta i(s_t, j)
$$
Où :

- $T_b$ est l'ensemble des nœuds de l'arbre $b$.
- $p(t)$ est la proportion d'échantillons atteignant le nœud $t$.
- $\Delta i(s_t, j)$ est la diminution de l'impureté due à la division $s_t$ sur la caractéristique $j$.

Le RandomForestRegressor offre plusieurs avantages, notamment une bonne gestion du surapprentissage, une capacité à capturer des relations non linéaires, et une robustesse aux valeurs aberrantes et au bruit dans les données.
""",
"ExtraTreesRegressor":r"""
L'ExtraTreesRegressor (Extremely Randomized Trees Regressor) est un modèle d'ensemble similaire au RandomForestRegressor, mais avec une randomisation accrue dans la construction des arbres. Cette approche vise à réduire davantage la variance du modèle au prix d'une légère augmentation du biais.
Le modèle peut être exprimé mathématiquement comme suit :
$$
\hat{y} = \frac{1}{B} \sum_{b=1}^B f_b(\mathbf{x})
$$
Où :

- $\hat{y}$ est la valeur prédite finale.
- $B$ est le nombre total d'arbres dans l'ensemble.
- $f_b(\mathbf{x})$ est la prédiction du $b$-ème arbre pour le vecteur d'entrée $\mathbf{x}$.

Les principales différences avec le RandomForestRegressor sont :

Échantillonnage des données : ExtraTrees utilise l'ensemble complet des données pour chaque arbre, sans échantillonnage bootstrap.
Sélection des seuils : Pour chaque caractéristique considérée, au lieu de chercher le meilleur seuil possible, ExtraTrees génère des seuils aléatoires et choisit le meilleur parmi ceux-ci.

La division à chaque nœud est choisie pour maximiser la réduction de la variance :
$$
\Delta V = V(S) - \frac{n_l}{n} V(S_l) - \frac{n_r}{n} V(S_r)
$$
Où :

- $V(S)$ est la variance de l'ensemble $S$ au nœud courant.
- $S_l$ et $S_r$ sont les sous-ensembles gauche et droit après la division.
- $n$, $n_l$, et $n_r$ sont les nombres d'échantillons dans $S$, $S_l$, et $S_r$ respectivement.

La variance est calculée comme suit :
$$
V(S) = \frac{1}{n} \sum_{i=1}^n (y_i - \bar{y})^2
$$
Où $\bar{y}$ est la moyenne des valeurs cibles dans l'ensemble $S$.
L'importance des caractéristiques peut être estimée de manière similaire au RandomForestRegressor :
$$
\text{Importance}j = \frac{1}{B} \sum{b=1}^B \sum_{t \in T_b} p(t) \Delta V(s_t, j)
$$
Où $\Delta V(s_t, j)$ est la réduction de variance due à la division $s_t$ sur la caractéristique $j$.
L'ExtraTreesRegressor offre souvent une meilleure généralisation que le RandomForestRegressor grâce à sa randomisation accrue, tout en conservant une bonne performance prédictive et une faible sensibilité au bruit dans les données.
""",
"GradientBoostingClassifier":r"""
Le GradientBoostingClassifier est un modèle d'ensemble qui combine des arbres de décision faibles en une série séquentielle pour former un prédicteur puissant. Il utilise la technique du gradient boosting pour minimiser la fonction de perte.
Le modèle peut être exprimé mathématiquement comme suit :
$$
F_M(\mathbf{x}) = F_0(\mathbf{x}) + \sum_{m=1}^M \gamma_m h_m(\mathbf{x})
$$
Où :

- $F_M(\mathbf{x})$ est le modèle final après $M$ itérations.
- $F_0(\mathbf{x})$ est le modèle initial (généralement une constante).
- $h_m(\mathbf{x})$ est le $m$-ème arbre de décision faible.
- $\gamma_m$ est le coefficient de pondération pour le $m$-ème arbre.

Pour la classification binaire, la prédiction est donnée par :
$$
\hat{y} = \text{sign}(F_M(\mathbf{x}))
$$
Le processus d'entraînement minimise une fonction de perte $L(y, F(\mathbf{x}))$. Pour la classification binaire, on utilise souvent la log-perte :

$$
L(y, F(\mathbf{x})) = \log(1 + e^{-2yF(\mathbf{x})})
$$

À chaque itération $m$, le modèle ajoute un nouvel arbre $h_m(\mathbf{x})$ qui minimise :

$$
\sum_{i=1}^n L(y_i, F_{m-1}(\mathbf{x}_i) + h_m(\mathbf{x}_i))
$$

Cela est approximé en ajustant $h_m(\mathbf{x})$ aux pseudo-résidus négatifs :

$$
r_{im} = -\left[\frac{\partial L(y_i, F(\mathbf{x}i))}{\partial F(\mathbf{x}i)}\right]{F(\mathbf{x})=F{m-1}(\mathbf{x})}
$$

Le taux d'apprentissage $\eta$ (learning rate) est introduit pour contrôler la contribution de chaque arbre :
$$
F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \gamma_m h_m(\mathbf{x})
$$
L'importance des caractéristiques peut être estimée en sommant les améliorations de la perte pour chaque caractéristique $j$ sur tous les arbres :
$$
I_j^2 = \sum_{m=1}^M \sum_{t \in T_m} i_t^2 \mathbb{1}(v(t) = j)
$$
Où $i_t^2$ est l'amélioration de la perte pour le nœud $t$, et $v(t)$ est la caractéristique utilisée pour la division au nœud $t$.
Le GradientBoostingClassifier offre une haute précision et gère bien les interactions complexes entre les caractéristiques, mais peut être sensible au surapprentissage si ses hyperparamètres ne sont pas correctement réglés.
""",
"GradientBoostingRegressor": r"""
Le GradientBoostingRegressor est un modèle d'ensemble qui utilise la technique du gradient boosting pour la régression. Il combine séquentiellement des arbres de décision faibles pour former un prédicteur puissant, en minimisant une fonction de perte.
Le modèle peut être exprimé mathématiquement comme suit :
$$
F_M(\mathbf{x}) = F_0(\mathbf{x}) + \sum_{m=1}^M \gamma_m h_m(\mathbf{x})
$$
Où :

- $F_M(\mathbf{x})$ est le modèle final après $M$ itérations.
- $F_0(\mathbf{x})$ est le modèle initial (généralement la moyenne des valeurs cibles).
- $h_m(\mathbf{x})$ est le $m$-ème arbre de décision faible.
- $\gamma_m$ est le coefficient de pondération pour le $m$-ème arbre.

Pour la régression, la prédiction est directement donnée par $F_M(\mathbf{x})$.
Le processus d'entraînement minimise une fonction de perte $L(y, F(\mathbf{x}))$. Pour la régression, on utilise souvent l'erreur quadratique moyenne (MSE) :
$$
L(y, F(\mathbf{x})) = \frac{1}{2}(y - F(\mathbf{x}))^2
$$
À chaque itération $m$, le modèle ajoute un nouvel arbre $h_m(\mathbf{x})$ qui minimise :
$$
\sum_{i=1}^n L(y_i, F_{m-1}(\mathbf{x}_i) + h_m(\mathbf{x}_i))
$$
Cela est approximé en ajustant $h_m(\mathbf{x})$ aux résidus négatifs :
$$
r_{im} = -\left[\frac{\partial L(y_i, F(\mathbf{x}i))}{\partial F(\mathbf{x}i)}\right]{F(\mathbf{x})=F{m-1}(\mathbf{x})}
$$
Pour la MSE, les résidus sont simplement :
$$
r_{im} = y_i - F_{m-1}(\mathbf{x}_i)
$$
Le taux d'apprentissage $\eta$ (learning rate) est introduit pour contrôler la contribution de chaque arbre :
$$
F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \gamma_m h_m(\mathbf{x})
$$
L'importance des caractéristiques peut être estimée en sommant les améliorations de la perte pour chaque caractéristique $j$ sur tous les arbres :
$$
I_j^2 = \sum_{m=1}^M \sum_{t \in T_m} i_t^2 \mathbb{1}(v(t) = j)
$$
Où $i_t^2$ est l'amélioration de la perte pour le nœud $t$, et $v(t)$ est la caractéristique utilisée pour la division au nœud $t$.
Le GradientBoostingRegressor offre plusieurs avantages :

Haute précision prédictive
Gestion des interactions complexes entre les caractéristiques
Robustesse aux valeurs aberrantes (avec des fonctions de perte appropriées)
Capacité à capturer des relations non linéaires

Cependant, il peut être sensible au surapprentissage si ses hyperparamètres (comme le nombre d'arbres et la profondeur maximale) ne sont pas correctement réglés.
""",
"RandomForestClassifier":r"""
Le RandomForestClassifier est un modèle d'ensemble qui combine plusieurs arbres de décision pour effectuer des tâches de classification. Il utilise la technique du bagging (bootstrap aggregating) pour améliorer la stabilité et la précision des prédictions.
Le modèle peut être exprimé mathématiquement comme suit :
$$
\hat{y} = \text{mode}\{f_b(\mathbf{x})\}_{b=1}^B
$$
Où :

- $\hat{y}$ est la classe prédite finale.
- $B$ est le nombre total d'arbres dans la forêt.
- $f_b(\mathbf{x})$ est la prédiction du $b$-ème arbre pour le vecteur d'entrée $\mathbf{x}$.
- $\text{mode}\{\cdot\}$ représente la classe la plus fréquente (vote majoritaire).

Chaque arbre $f_b$ est construit à partir d'un échantillon bootstrap $\mathcal{D}_b$ tiré avec remplacement de l'ensemble de données original $\mathcal{D}$. La taille de $\mathcal{D}_b$ est généralement la même que celle de $\mathcal{D}$.
Pour chaque nœud de l'arbre, un sous-ensemble aléatoire de $m$ caractéristiques est sélectionné parmi les $p$ caractéristiques disponibles, où typiquement $m = \sqrt{p}$ pour la classification.
La division à chaque nœud est choisie pour maximiser la réduction de l'impureté, souvent mesurée par l'indice de Gini ou l'entropie. Pour l'indice de Gini :
$$
\text{Gini} = 1 - \sum_{k=1}^K p_k^2
$$
Où $K$ est le nombre de classes et $p_k$ est la proportion d'échantillons de la classe $k$ dans le nœud.
La probabilité de classe pour une nouvelle observation $\mathbf{x}$ est calculée comme la moyenne des probabilités prédites par chaque arbre :
$$
P(y = k | \mathbf{x}) = \frac{1}{B} \sum_{b=1}^B P_b(y = k | \mathbf{x})
$$
L'importance des caractéristiques peut être estimée en calculant la diminution moyenne de l'impureté (MDI) pour chaque caractéristique $j$ :
$$
\text{MDI}j = \frac{1}{B} \sum{b=1}^B \sum_{t \in T_b} p(t) \Delta i(s_t, j)
$$
Où :

- $T_b$ est l'ensemble des nœuds de l'arbre $b$.
- $p(t)$ est la proportion d'échantillons atteignant le nœud $t$.
- $\Delta i(s_t, j)$ est la diminution de l'impureté due à la division $s_t$ sur la caractéristique $j$.

Le RandomForestClassifier offre plusieurs avantages :

Bonne gestion du surapprentissage
Capacité à capturer des relations non linéaires
Robustesse aux valeurs aberrantes et au bruit dans les données
Estimation de l'importance des caractéristiques
Possibilité de parallélisation pour un entraînement plus rapide

Il est particulièrement efficace pour les problèmes de classification complexes avec de nombreuses caractéristiques et des interactions non linéaires.
""",
"ExtraTreesClassifier":r"""
L'ExtraTreesClassifier (Extremely Randomized Trees Classifier) est un modèle d'ensemble similaire au RandomForestClassifier, mais avec une randomisation accrue dans la construction des arbres. Cette approche vise à réduire davantage la variance du modèle au prix d'une légère augmentation du biais.
Le modèle peut être exprimé mathématiquement comme suit :
$$
\hat{y} = \text{mode}\{f_b(\mathbf{x})\}_{b=1}^B
$$
Où :

- $\hat{y}$ est la classe prédite finale.
- $B$ est le nombre total d'arbres dans l'ensemble.
- $f_b(\mathbf{x})$ est la prédiction du $b$-ème arbre pour le vecteur d'entrée $\mathbf{x}$.
- $\text{mode}\{\cdot\}$ représente la classe la plus fréquente (vote majoritaire).

Les principales différences avec le RandomForestClassifier sont :

Échantillonnage des données : ExtraTrees utilise l'ensemble complet des données pour chaque arbre, sans échantillonnage bootstrap.
Sélection des seuils : Pour chaque caractéristique considérée, au lieu de chercher le meilleur seuil possible, ExtraTrees génère des seuils aléatoires et choisit le meilleur parmi ceux-ci.

La division à chaque nœud est choisie pour maximiser la réduction de l'impureté, souvent mesurée par l'indice de Gini ou l'entropie. Pour l'entropie :
$$
\text{Entropie} = -\sum_{k=1}^K p_k \log_2(p_k)
$$
Où $K$ est le nombre de classes et $p_k$ est la proportion d'échantillons de la classe $k$ dans le nœud.
La probabilité de classe pour une nouvelle observation $\mathbf{x}$ est calculée comme la moyenne des probabilités prédites par chaque arbre :
$$
P(y = k | \mathbf{x}) = \frac{1}{B} \sum_{b=1}^B P_b(y = k | \mathbf{x})
$$
L'importance des caractéristiques peut être estimée de manière similaire au RandomForestClassifier :
$$
\text{Importance}j = \frac{1}{B} \sum{b=1}^B \sum_{t \in T_b} p(t) \Delta i(s_t, j)
$$
Où $\Delta i(s_t, j)$ est la réduction de l'impureté due à la division $s_t$ sur la caractéristique $j$.
L'ExtraTreesClassifier offre plusieurs avantages :

Meilleure généralisation grâce à la randomisation accrue
Réduction du surapprentissage par rapport au RandomForestClassifier
Temps d'entraînement potentiellement plus court (pas de recherche du meilleur seuil)
Bonne performance sur des problèmes à haute dimension
Robustesse au bruit et aux valeurs aberrantes

Il est particulièrement efficace pour les problèmes de classification complexes où une forte randomisation peut aider à capturer des motifs subtils dans les données.
""",
"SVR":"""
Le modèle de Support Vector Regression (SVR) est une méthode de régression qui utilise les concepts des machines à vecteurs de support (SVM) pour prédire une variable continue. L'objectif principal du SVR est de trouver une fonction qui a au plus une déviation $\epsilon$ des cibles réelles pour toutes les données d'entraînement, tout en étant aussi plate que possible. Mathématiquement, le modèle peut être exprimé comme suit :

$$
f(\\mathbf{x}) = \\langle \\mathbf{w}, \\mathbf{x} \\rangle + b
$$

Où :
- $f(\\mathbf{x})$ est la fonction prédite pour l'entrée $\\mathbf{x}$.
- $\\mathbf{w}$ est le vecteur des poids.
- $\\langle \\mathbf{w}, \\mathbf{x} \\rangle$ représente le produit scalaire entre $\\mathbf{w}$ et $\\mathbf{x}$.
- $b$ est le biais (intercept).

Le SVR cherche à minimiser la fonction de coût suivante :

$$
\\frac{1}{2} ||\\mathbf{w}||^2 + C \\sum_{i=1}^{n} L_{\\epsilon}(y_i, f(\\mathbf{x}_i))
$$

Où :
- $||\\mathbf{w}||^2$ est la norme du vecteur des poids, que l'on cherche à minimiser pour maintenir la fonction aussi plate que possible.
- $C$ est un paramètre de régularisation qui détermine l'équilibre entre la marge et l'erreur d'entraînement.
- $L_{\\epsilon}(y_i, f(\\mathbf{x}_i))$ est la perte $\epsilon$-insensible, définie par :

$$
L_{\\epsilon}(y_i, f(\\mathbf{x}_i)) = \\max(0, |y_i - f(\\mathbf{x}_i)| - \\epsilon)
$$

Où :
- $y_i$ est la valeur réelle pour l'observation $i$.
- $f(\\mathbf{x}_i)$ est la valeur prédite pour l'observation $i$.
- $\\epsilon$ est un hyperparamètre qui définit une zone autour de la prédiction dans laquelle les erreurs ne sont pas pénalisées.

Le SVR utilise également des noyaux (kernels) pour gérer les relations non linéaires entre les variables d'entrée et la sortie. Le noyau le plus couramment utilisé est le noyau gaussien (RBF).

Le modèle SVR est particulièrement utile pour les problèmes de régression où l'on souhaite une bonne généralisation tout en contrôlant les erreurs dans une plage spécifiée.
""",
"KNeighborsClassifier":"""
Le modèle KNeighborsClassifier est une méthode d'apprentissage supervisé utilisée pour les tâches de classification. Ce modèle classe une nouvelle observation en fonction des classes des $k$ observations les plus proches dans l'ensemble de données d'entraînement. L'idée est que des observations similaires se trouvent près les unes des autres dans l'espace des caractéristiques. Mathématiquement, le modèle fonctionne comme suit :

1. **Distance entre les observations :**

   Pour chaque nouvelle observation $\\mathbf{x}$, on calcule la distance entre $\\mathbf{x}$ et chaque observation $\\mathbf{x}_i$ dans l'ensemble d'entraînement. La distance la plus couramment utilisée est la distance euclidienne, définie par :

   $$
   d(\\mathbf{x}, \\mathbf{x}_i) = \\sqrt{\\sum_{j=1}^{p} (x_j - x_{ij})^2}
   $$

   Où :
   - $d(\\mathbf{x}, \\mathbf{x}_i)$ est la distance entre l'observation $\\mathbf{x}$ et l'observation d'entraînement $\\mathbf{x}_i$.
   - $x_j$ est la $j$-ème caractéristique de l'observation $\\mathbf{x}$.
   - $x_{ij}$ est la $j$-ème caractéristique de l'observation d'entraînement $\\mathbf{x}_i$.
   - $p$ est le nombre total de caractéristiques.

2. **Identification des $k$ plus proches voisins :**

   Après avoir calculé les distances, on sélectionne les $k$ observations d'entraînement les plus proches de $\\mathbf{x}$, c'est-à-dire celles avec les plus petites distances $d(\\mathbf{x}, \\mathbf{x}_i)$.

3. **Prédiction de la classe :**

   La classe prédite $\\hat{y}$ pour l'observation $\\mathbf{x}$ est déterminée par un vote majoritaire parmi les classes des $k$ plus proches voisins :

   $$
   \\hat{y} = \\text{mode}(y_{i_1}, y_{i_2}, \\dots, y_{i_k})
   $$

   Où :
   - $y_{i_1}, y_{i_2}, \\dots, y_{i_k}$ sont les classes des $k$ voisins les plus proches.
   - $\\text{mode}$ représente la classe la plus fréquente parmi les $k$ voisins.

Le modèle KNeighborsClassifier est simple et non paramétrique, ce qui le rend facile à comprendre et à appliquer. Cependant, il peut être sensible aux valeurs aberrantes et à la sélection du paramètre $k$, qui doit être choisi en fonction des données spécifiques.
""",
"KNeighborsRegressor":"""
Le modèle KNeighborsRegressor est une méthode d'apprentissage supervisé utilisée pour les tâches de régression. Ce modèle prédit la valeur cible d'une nouvelle observation en fonction des valeurs cibles des $k$ observations les plus proches dans l'ensemble de données d'entraînement. L'idée est que des observations similaires se trouvent près les unes des autres dans l'espace des caractéristiques. Mathématiquement, le modèle fonctionne comme suit :

1. **Distance entre les observations :**

   Pour chaque nouvelle observation $\\mathbf{x}$, on calcule la distance entre $\\mathbf{x}$ et chaque observation $\\mathbf{x}_i$ dans l'ensemble d'entraînement. La distance la plus couramment utilisée est la distance euclidienne, définie par :

   $$
   d(\\mathbf{x}, \\mathbf{x}_i) = \\sqrt{\\sum_{j=1}^{p} (x_j - x_{ij})^2}
   $$

   Où :
   - $d(\\mathbf{x}, \\mathbf{x}_i)$ est la distance entre l'observation $\\mathbf{x}$ et l'observation d'entraînement $\\mathbf{x}_i$.
   - $x_j$ est la $j$-ème caractéristique de l'observation $\\mathbf{x}$.
   - $x_{ij}$ est la $j$-ème caractéristique de l'observation d'entraînement $\\mathbf{x}_i$.
   - $p$ est le nombre total de caractéristiques.

2. **Identification des $k$ plus proches voisins :**

   Après avoir calculé les distances, on sélectionne les $k$ observations d'entraînement les plus proches de $\\mathbf{x}$, c'est-à-dire celles avec les plus petites distances $d(\\mathbf{x}, \\mathbf{x}_i)$.

3. **Prédiction de la valeur cible :**

   La valeur cible prédite $\\hat{y}$ pour l'observation $\\mathbf{x}$ est calculée comme la moyenne des valeurs cibles $y_{i_1}, y_{i_2}, \\dots, y_{i_k}$ des $k$ plus proches voisins :

   $$
   \\hat{y} = \\frac{1}{k} \\sum_{i=1}^{k} y_{i}
   $$

   Où :
   - $y_{i}$ est la valeur cible de l'observation d'entraînement $i$ parmi les $k$ voisins les plus proches.
   - $k$ est le nombre de voisins choisis.

Le modèle KNeighborsRegressor est simple à comprendre et à implémenter. Cependant, comme pour la version classificateur, il peut être sensible au choix de $k$ et aux valeurs aberrantes. Il fonctionne bien pour des données où la relation entre les variables d'entrée et la sortie est relativement simple.
""",
"AdaBoostClassifier":"""
Le modèle AdaBoostClassifier est une méthode d'apprentissage supervisé qui combine plusieurs classificateurs faibles (comme des arbres de décision de faible profondeur) pour former un classificateur robuste. AdaBoost, qui signifie "Adaptive Boosting", ajuste les poids des observations à chaque itération, de sorte que les observations mal classées reçoivent plus d'importance dans les classificateurs suivants. Le modèle peut être exprimé mathématiquement comme suit :

1. **Entraînement des classificateurs faibles :**

   À chaque itération $m$ (de $m = 1$ à $M$), un classificateur faible $h_m(\\mathbf{x})$ est entraîné sur les données pondérées, où $\\mathbf{x}$ représente les variables d'entrée et $y$ la variable cible. Le poids de chaque observation $i$ à l'itération $m$ est noté $w_i^m$.

2. **Erreur du classificateur :**

   L'erreur $\\epsilon_m$ du classificateur $h_m(\\mathbf{x})$ est calculée comme la somme des poids des observations mal classées :

   $$
   \\epsilon_m = \\sum_{i=1}^{n} w_i^m \\cdot I(y_i \\neq h_m(\\mathbf{x}_i))
   $$

   Où :
   - $I(y_i \\neq h_m(\\mathbf{x}_i))$ est une fonction indicatrice valant 1 si l'observation $i$ est mal classée, sinon 0.
   - $n$ est le nombre total d'observations.

3. **Calcul du poids du classificateur :**

   Le poids $\\alpha_m$ du classificateur $h_m(\\mathbf{x})$ est calculé en fonction de son erreur $\\epsilon_m$ :

   $$
   \\alpha_m = \\frac{1}{2} \\ln\\left(\\frac{1 - \\epsilon_m}{\\epsilon_m}\\right)
   $$

   Les classificateurs avec une erreur plus faible reçoivent un poids plus élevé.

4. **Mise à jour des poids des observations :**

   Les poids des observations sont ensuite mis à jour pour l'itération suivante en tenant compte du poids $\\alpha_m$ du classificateur $h_m(\\mathbf{x})$ :

   $$
   w_i^{m+1} = w_i^m \\cdot \\exp\\left(-\\alpha_m \\cdot y_i \\cdot h_m(\\mathbf{x}_i)\\right)
   $$

   Les observations mal classées voient leur poids augmenter, tandis que celles bien classées voient leur poids diminuer.

5. **Prédiction finale :**

   Le classificateur final $H(\\mathbf{x})$ est une combinaison pondérée de tous les classificateurs faibles :

   $$
   H(\\mathbf{x}) = \\text{sign}\\left(\\sum_{m=1}^{M} \\alpha_m h_m(\\mathbf{x})\\right)
   $$

   Où :
   - $\\text{sign}$ est la fonction signe, qui attribue la classe $+1$ ou $-1$ en fonction du signe de la somme pondérée.

Le modèle AdaBoostClassifier est particulièrement efficace pour améliorer les performances des modèles de base (classificateurs faibles) en les combinant de manière adaptative pour corriger les erreurs des modèles précédents.
""",
"AdaBoostRegressor":"""
Le modèle AdaBoostRegressor est une méthode d'apprentissage supervisé qui combine plusieurs régressions faibles pour former un modèle de régression robuste. AdaBoost, qui signifie "Adaptive Boosting", ajuste les poids des observations à chaque itération, de sorte que les observations avec des erreurs importantes reçoivent plus d'importance dans les régressions suivantes. Le modèle peut être exprimé mathématiquement comme suit :

1. **Entraînement des régressions faibles :**

   À chaque itération $m$ (de $m = 1$ à $M$), un régresseur faible $h_m(\\mathbf{x})$ est entraîné sur les données pondérées, où $\\mathbf{x}$ représente les variables d'entrée et $y$ la variable cible. Le poids de chaque observation $i$ à l'itération $m$ est noté $w_i^m$.

2. **Erreur du régresseur :**

   L'erreur $\\epsilon_m$ du régresseur $h_m(\\mathbf{x})$ est calculée en tant que somme des poids des erreurs pondérées :

   $$
   \\epsilon_m = \\frac{\\sum_{i=1}^{n} w_i^m \\cdot |y_i - h_m(\\mathbf{x}_i)|}{\\sum_{i=1}^{n} w_i^m}
   $$

   Où :
   - $|y_i - h_m(\\mathbf{x}_i)|$ représente l'erreur absolue entre la valeur réelle $y_i$ et la valeur prédite $h_m(\\mathbf{x}_i)$.
   - $n$ est le nombre total d'observations.

3. **Calcul du poids du régresseur :**

   Le poids $\\alpha_m$ du régresseur $h_m(\\mathbf{x})$ est calculé en fonction de son erreur $\\epsilon_m$ :

   $$
   \\alpha_m = \\frac{1}{2} \\ln\\left(\\frac{1 - \\epsilon_m}{\\epsilon_m}\\right)
   $$

   Les régressions avec une erreur plus faible reçoivent un poids plus élevé.

4. **Mise à jour des poids des observations :**

   Les poids des observations sont ensuite mis à jour pour l'itération suivante en tenant compte du poids $\\alpha_m$ du régresseur $h_m(\\mathbf{x})$ :

   $$
   w_i^{m+1} = w_i^m \\cdot \\exp\\left(\\alpha_m \\cdot |y_i - h_m(\\mathbf{x}_i)|\\right)
   $$

   Les observations avec de grandes erreurs voient leur poids augmenter, tandis que celles avec de petites erreurs voient leur poids diminuer.

5. **Prédiction finale :**

   Le régresseur final $H(\\mathbf{x})$ est une combinaison pondérée de tous les régressions faibles :

   $$
   H(\\mathbf{x}) = \\sum_{m=1}^{M} \\alpha_m h_m(\\mathbf{x})
   $$

Le modèle AdaBoostRegressor est particulièrement efficace pour améliorer les performances des régressions de base (régressions faibles) en les combinant de manière adaptative pour réduire l'erreur totale.
""",
"MLPClassifier":"""
Le modèle MLPClassifier (Multi-Layer Perceptron Classifier) est un type de réseau de neurones artificiels utilisé pour les tâches de classification. Ce modèle est composé de plusieurs couches de neurones, y compris une couche d'entrée, une ou plusieurs couches cachées, et une couche de sortie. Chaque neurone applique une fonction d'activation pour générer une sortie à partir d'une combinaison linéaire des entrées. Le modèle peut être exprimé mathématiquement comme suit :

1. **Propagation avant (Forward Propagation) :**

   Chaque couche $l$ du réseau de neurones prend les sorties de la couche précédente comme entrées. La sortie $z_j^l$ d'un neurone $j$ de la couche $l$ est calculée comme une combinaison linéaire pondérée des sorties de la couche précédente plus un biais $b_j^l$ :

   $$
   z_j^l = \\sum_{i=1}^{n_{l-1}} w_{ji}^l a_i^{l-1} + b_j^l
   $$

   Où :
   - $w_{ji}^l$ est le poids entre le neurone $i$ de la couche $(l-1)$ et le neurone $j$ de la couche $l$.
   - $a_i^{l-1}$ est la sortie du neurone $i$ de la couche $(l-1)$.
   - $b_j^l$ est le biais ajouté au neurone $j$ de la couche $l$.
   - $n_{l-1}$ est le nombre de neurones dans la couche $(l-1)$.

2. **Fonction d'activation :**

   La sortie du neurone $j$ dans la couche $l$, notée $a_j^l$, est obtenue en appliquant une fonction d'activation $\\sigma$ à $z_j^l$ :

   $$
   a_j^l = \\sigma(z_j^l)
   $$

   Les fonctions d'activation couramment utilisées incluent la fonction sigmoïde, la fonction tanh, et la fonction ReLU (Rectified Linear Unit).

3. **Fonction de coût (Cost Function) :**

   Pour les tâches de classification, la fonction de coût est généralement l'entropie croisée, qui mesure la différence entre les prédictions du modèle et les vraies étiquettes. Elle est définie comme suit :

   $$
   J(\\mathbf{W}, \\mathbf{b}) = -\\frac{1}{n} \\sum_{i=1}^{n} \\sum_{k=1}^{K} y_{ik} \\log(\\hat{y}_{ik})
   $$

   Où :
   - $n$ est le nombre total d'exemples d'entraînement.
   - $K$ est le nombre de classes.
   - $y_{ik}$ est une variable binaire indiquant si l'exemple $i$ appartient à la classe $k$.
   - $\\hat{y}_{ik}$ est la probabilité prédite que l'exemple $i$ appartient à la classe $k$.

4. **Rétropropagation (Backpropagation) :**

   Les poids $\\mathbf{W}$ et les biais $\\mathbf{b}$ sont ajustés en minimisant la fonction de coût à l'aide de l'algorithme de rétropropagation, qui utilise le gradient de la fonction de coût par rapport aux paramètres pour mettre à jour les poids et les biais par descente de gradient.

5. **Prédiction :**

   Lors de la phase de prédiction, le modèle MLPClassifier calcule la sortie finale en passant les données d'entrée à travers toutes les couches et en appliquant la fonction softmax sur la sortie de la dernière couche pour obtenir les probabilités de chaque classe :

   $$
   \\hat{y}_k = \\frac{\\exp(a_k^L)}{\\sum_{j=1}^{K} \\exp(a_j^L)}
   $$

   Où $L$ est la dernière couche et $K$ le nombre de classes.

Le modèle MLPClassifier est puissant pour capturer des relations non linéaires complexes dans les données, et il est largement utilisé pour les tâches de classification en apprentissage profond.
""",
"MLPRegressor":"""
Le modèle MLPRegressor (Multi-Layer Perceptron Regressor) est un type de réseau de neurones artificiels utilisé pour les tâches de régression. Ce modèle est composé de plusieurs couches de neurones, y compris une couche d'entrée, une ou plusieurs couches cachées, et une couche de sortie. Chaque neurone applique une fonction d'activation pour générer une sortie à partir d'une combinaison linéaire des entrées. Le modèle peut être exprimé mathématiquement comme suit :

1. **Propagation avant (Forward Propagation) :**

   Chaque couche $l$ du réseau de neurones prend les sorties de la couche précédente comme entrées. La sortie $z_j^l$ d'un neurone $j$ de la couche $l$ est calculée comme une combinaison linéaire pondérée des sorties de la couche précédente plus un biais $b_j^l$ :

   $$
   z_j^l = \\sum_{i=1}^{n_{l-1}} w_{ji}^l a_i^{l-1} + b_j^l
   $$

   Où :
   - $w_{ji}^l$ est le poids entre le neurone $i$ de la couche $(l-1)$ et le neurone $j$ de la couche $l$.
   - $a_i^{l-1}$ est la sortie du neurone $i$ de la couche $(l-1)$.
   - $b_j^l$ est le biais ajouté au neurone $j$ de la couche $l$.
   - $n_{l-1}$ est le nombre de neurones dans la couche $(l-1)$.

2. **Fonction d'activation :**

   La sortie du neurone $j$ dans la couche $l$, notée $a_j^l$, est obtenue en appliquant une fonction d'activation $\\sigma$ à $z_j^l$ :

   $$
   a_j^l = \\sigma(z_j^l)
   $$

   Les fonctions d'activation couramment utilisées incluent la fonction sigmoïde, la fonction tanh, et la fonction ReLU (Rectified Linear Unit).

3. **Fonction de coût (Cost Function) :**

   Pour les tâches de régression, la fonction de coût est généralement l'erreur quadratique moyenne (MSE), qui mesure la différence entre les prédictions du modèle et les vraies valeurs cibles. Elle est définie comme suit :

   $$
   J(\\mathbf{W}, \\mathbf{b}) = \\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2
   $$

   Où :
   - $n$ est le nombre total d'exemples d'entraînement.
   - $y_i$ est la valeur réelle pour l'exemple $i$.
   - $\\hat{y}_i$ est la valeur prédite par le modèle pour l'exemple $i$.

4. **Rétropropagation (Backpropagation) :**

   Les poids $\\mathbf{W}$ et les biais $\\mathbf{b}$ sont ajustés en minimisant la fonction de coût à l'aide de l'algorithme de rétropropagation, qui utilise le gradient de la fonction de coût par rapport aux paramètres pour mettre à jour les poids et les biais par descente de gradient.

5. **Prédiction :**

   Lors de la phase de prédiction, le modèle MLPRegressor calcule la sortie finale en passant les données d'entrée à travers toutes les couches, et cette sortie est directement utilisée comme la prédiction pour la variable cible continue.

Le modèle MLPRegressor est particulièrement efficace pour capturer des relations non linéaires complexes dans les données, et il est largement utilisé pour les tâches de régression en apprentissage profond.
""",
"GaussianProcessRegressor":"""
Le modèle GaussianProcessRegressor (Régression par processus gaussien) est une méthode bayésienne non paramétrique pour les tâches de régression. Un processus gaussien (GP) est une collection de variables aléatoires, dont chaque ensemble fini suit une distribution gaussienne. Le modèle est entièrement défini par sa fonction de moyenne $m(\\mathbf{x})$ et sa fonction de covariance $k(\\mathbf{x}, \\mathbf{x}')$, où $\\mathbf{x}$ et $\\mathbf{x}'$ sont des points d'entrée.

1. **Distribution a priori :**

   On suppose que les observations $\\mathbf{y}$ sont générées à partir d'un processus gaussien avec une moyenne nulle et une covariance définie par une fonction de noyau $k(\\mathbf{x}, \\mathbf{x}')$ :

   $$
   \\mathbf{y} \\sim \\mathcal{N}(\\mathbf{0}, K + \\sigma_n^2 I)
   $$

   Où :
   - $K$ est la matrice de covariance avec $K_{ij} = k(\\mathbf{x}_i, \\mathbf{x}_j)$.
   - $\\sigma_n^2$ est la variance du bruit.

2. **Prédiction :**

   Pour un nouvel ensemble de points d'entrée $\\mathbf{X}_*$, la prédiction $\\mathbf{y}_*$ est également un processus gaussien avec :

   $$
   \\mathbf{y}_* \\mid \\mathbf{X}, \\mathbf{y}, \\mathbf{X}_* \\sim \\mathcal{N}(\\mathbf{\\mu}_*, \\mathbf{\\Sigma}_*)
   $$

   Où :
   - $\\mathbf{\\mu}_* = K_*^T (K + \\sigma_n^2 I)^{-1} \\mathbf{y}$
   - $\\mathbf{\\Sigma}_* = K_{**} - K_*^T (K + \\sigma_n^2 I)^{-1} K_*$

   Avec :
   - $K_*$ la covariance entre les points d'entraînement $\\mathbf{X}$ et les nouveaux points $\\mathbf{X}_*$.
   - $K_{**}$ la covariance entre les nouveaux points $\\mathbf{X}_*$.

Le modèle GaussianProcessRegressor est particulièrement puissant pour capturer les relations complexes et fournir des estimations de l'incertitude sur les prédictions.
""",
"GaussianProcessClassifier":"""
Le modèle GaussianProcessClassifier (Classification par processus gaussien) est une méthode bayésienne non paramétrique pour les tâches de classification. Il modélise les probabilités a posteriori des classes en utilisant un processus gaussien. Contrairement à la régression, ici, la sortie est une probabilité, et on applique une fonction de lien logistique pour transformer les sorties du processus gaussien.

1. **Distribution a priori :**

   On suppose que la fonction latente $f(\\mathbf{x})$ suit un processus gaussien avec une moyenne nulle et une covariance $k(\\mathbf{x}, \\mathbf{x}')$ :

   $$
   f(\\mathbf{x}) \\sim \\mathcal{N}(\\mathbf{0}, K)
   $$

   Où :
   - $K$ est la matrice de covariance avec $K_{ij} = k(\\mathbf{x}_i, \\mathbf{x}_j)$.

2. **Fonction de lien logistique :**

   Les probabilités a posteriori pour les classes sont obtenues en appliquant la fonction logistique à la fonction latente $f(\\mathbf{x})$ :

   $$
   \\pi(\\mathbf{x}) = \\sigma(f(\\mathbf{x})) = \\frac{1}{1 + \\exp(-f(\\mathbf{x}))}
   $$

3. **Prédiction :**

   Pour un nouvel ensemble de points d'entrée $\\mathbf{X}_*$, la prédiction est donnée par l'intégration de la fonction logistique sur la distribution gaussienne a posteriori des fonctions latentes, ce qui nécessite généralement une approximation, comme l'approximation de Laplace ou l'échantillonnage de Monte Carlo.

Le modèle GaussianProcessClassifier est efficace pour capturer des frontières de décision complexes et fournit des estimations de l'incertitude sur les probabilités de classe.
""",
"SGDClassifier":"""
Le modèle SGDClassifier (classificateur par descente de gradient stochastique) est un classificateur linéaire qui optimise une fonction de perte convexes en utilisant l'algorithme de descente de gradient stochastique. Il est particulièrement efficace pour les grands ensembles de données.

1. **Modèle linéaire :**

   Le SGDClassifier modélise la relation entre les variables d'entrée $\\mathbf{x}$ et la variable de sortie $y$ sous la forme d'un modèle linéaire :

   $$
   y = \\mathbf{w}^T \\mathbf{x} + b
   $$

   Où :
   - $\\mathbf{w}$ est le vecteur des poids.
   - $b$ est le biais ou l'ordonnée à l'origine.

2. **Fonction de perte :**

   Le modèle peut optimiser différentes fonctions de perte, comme la perte logistique (pour la régression logistique), la perte hinge (pour la machine à vecteurs de support), ou la perte per-critic. Par exemple, pour la perte logistique utilisée pour la classification binaire :

   $$
   L(\\mathbf{w}, b) = \\frac{1}{n} \\sum_{i=1}^{n} \\log(1 + \\exp(-y_i (\\mathbf{w}^T \\mathbf{x}_i + b)))
   $$

3. **Descente de gradient stochastique :**

   Les paramètres $\\mathbf{w}$ et $b$ sont mis à jour itérativement en minimisant la fonction de perte par descente de gradient stochastique :

   $$
   \\mathbf{w} \\leftarrow \\mathbf{w} - \\eta \\nabla_w L(\\mathbf{w}, b)
   $$

   Où :
   - $\\eta$ est le taux d'apprentissage.
   - $\\nabla_w L(\\mathbf{w}, b)$ est le gradient de la fonction de perte par rapport aux poids.

Le modèle SGDClassifier est particulièrement adapté pour les grandes quantités de données et permet de traiter différents types de problèmes de classification.
""",
"NuSVC": """
Le modèle NuSVC est une variante des machines à vecteurs de support (SVM) qui introduit un paramètre $\\nu$ pour contrôler à la fois la proportion maximale d'exemples marginaux et la proportion minimale de vecteurs de support.

1. **Frontière de décision :**

   Le modèle NuSVC cherche à séparer les classes en trouvant une hyperplan qui maximise la marge entre les exemples de classes opposées, tout en contrôlant le nombre de vecteurs de support et la marge par le paramètre $\\nu$.

   $$
   \\mathbf{w}^T \\mathbf{x} + b = 0
   $$

   Où :
   - $\\mathbf{w}$ est le vecteur des poids.
   - $b$ est le biais.

2. **Optimisation :**

   Le problème d'optimisation de NuSVC est formulé de manière à maximiser la marge tout en respectant la contrainte imposée par $\\nu$ sur les vecteurs de support et les erreurs de classification.

   $$
   \\min_{\\mathbf{w}, b, \\xi} \\frac{1}{2} \\| \\mathbf{w} \\|^2
   $$

   Sous les contraintes :
   - $\\nu n \\geq \\sum_{i=1}^{n} \\alpha_i \\geq 0$
   - $y_i(\\mathbf{w}^T \\mathbf{x}_i + b) \\geq 1 - \\xi_i$
   - $\\xi_i \\geq 0$

3. **Paramètre $\\nu$ :**

   Le paramètre $\\nu \\in (0, 1]$ contrôle :
   - La fraction des exemples marginaux.
   - La fraction minimale de vecteurs de support.

Le modèle NuSVC est utile lorsque vous souhaitez contrôler explicitement les proportions de vecteurs de support et d'exemples marginaux, offrant une flexibilité supplémentaire par rapport au SVM standard.
""",
"NuSVR":"""
Le modèle NuSVR est une variante des machines à vecteurs de support pour les tâches de régression, similaire au NuSVC, mais adapté à la prédiction de valeurs continues. Il utilise un paramètre $\\nu$ pour contrôler le nombre de vecteurs de support et la largeur de la zone insensible $\\epsilon$ autour de la prédiction.

1. **Fonction de décision :**

   Le modèle NuSVR cherche à prédire une variable continue $y$ en fonction des variables d'entrée $\\mathbf{x}$, en trouvant un hyperplan qui minimise les erreurs tout en utilisant le paramètre $\\nu$ pour contrôler les vecteurs de support.

   $$
   y = \\mathbf{w}^T \\mathbf{x} + b
   $$

2. **Optimisation :**

   Le problème d'optimisation de NuSVR est formulé pour minimiser l'erreur de prédiction tout en respectant les contraintes imposées par $\\nu$ et $\\epsilon$.

   $$
   \\min_{\\mathbf{w}, b, \\xi, \\xi^*} \\frac{1}{2} \\| \\mathbf{w} \\|^2 + C \\sum_{i=1}^{n} (\\xi_i + \\xi_i^*)
   $$

   Sous les contraintes :
   - $|y_i - (\\mathbf{w}^T \\mathbf{x}_i + b)| \\leq \\epsilon + \\xi_i + \\xi_i^*$
   - $\\nu n \\geq \\sum_{i=1}^{n} \\alpha_i \\geq 0$
   - $\\xi_i, \\xi_i^* \\geq 0$

3. **Paramètre $\\nu$ :**

   Le paramètre $\\nu \\in (0, 1]$ contrôle :
   - La fraction des vecteurs de support.
   - La largeur de la zone insensible $\\epsilon$.

Le modèle NuSVR est utile pour les tâches de régression où vous souhaitez contrôler explicitement le nombre de vecteurs de support et la tolérance aux erreurs autour des prédictions.
""",
"RadiusNeighborsClassifier":"""
Le modèle RadiusNeighborsClassifier est un classificateur basé sur les k plus proches voisins, où chaque point est classé en fonction des voisins situés dans un rayon spécifié. Contrairement au KNeighborsClassifier, ce modèle ne fixe pas un nombre de voisins $k$, mais utilise plutôt un rayon $r$ pour déterminer les voisins pertinents.

1. **Classification par majorité :**

   Pour une observation donnée $\\mathbf{x}_i$, le modèle identifie tous les voisins $\\mathbf{x}_j$ dans le rayon $r$ et assigne à $\\mathbf{x}_i$ la classe la plus fréquente parmi ces voisins :

   $$
   y_i = \\text{argmax}_{c} \\sum_{j \\in N_i} \\mathbb{1}(y_j = c)
   $$

   Où :
   - $N_i$ est l'ensemble des voisins de $\\mathbf{x}_i$ tels que $\\|\\mathbf{x}_i - \\mathbf{x}_j\\| \\leq r$.
   - $y_j$ est la classe de l'observation $\\mathbf{x}_j$.
   - $c$ est une classe possible.

2. **Poids des voisins :**

   Le modèle peut également pondérer les votes des voisins en fonction de la distance, donnant plus de poids aux voisins plus proches :

   $$
   y_i = \\text{argmax}_{c} \\sum_{j \\in N_i} w_{ij} \\mathbb{1}(y_j = c)
   $$

   Où :
   - $w_{ij}$ est un poids qui dépend de la distance entre $\\mathbf{x}_i$ et $\\mathbf{x}_j$.

Le modèle RadiusNeighborsClassifier est particulièrement utile lorsque la densité des points de données varie fortement dans différentes régions de l'espace des caractéristiques, car il ajuste automatiquement le nombre de voisins en fonction de la densité locale.
""",
"RadiusNeighborsRegressor":"""
Le modèle RadiusNeighborsRegressor est un régresseur basé sur les k plus proches voisins, où chaque point est prédit en fonction des voisins situés dans un rayon spécifié. Comme pour le RadiusNeighborsClassifier, ce modèle ne fixe pas un nombre de voisins $k$, mais utilise plutôt un rayon $r$ pour déterminer les voisins pertinents.

1. **Prédiction par moyenne :**

   Pour une observation donnée $\\mathbf{x}_i$, le modèle calcule la prédiction $\\hat{y}_i$ comme la moyenne des valeurs cibles $y_j$ des voisins $\\mathbf{x}_j$ situés dans le rayon $r$ :

   $$
   \\hat{y}_i = \\frac{1}{|N_i|} \\sum_{j \\in N_i} y_j
   $$

   Où :
   - $N_i$ est l'ensemble des voisins de $\\mathbf{x}_i$ tels que $\\|\\mathbf{x}_i - \\mathbf{x}_j\\| \\leq r$.
   - $y_j$ est la valeur cible de l'observation $\\mathbf{x}_j$.

2. **Poids des voisins :**

   Le modèle peut également pondérer les contributions des voisins en fonction de la distance, donnant plus de poids aux voisins plus proches :

   $$
   \\hat{y}_i = \\frac{\\sum_{j \\in N_i} w_{ij} y_j}{\\sum_{j \\in N_i} w_{ij}}
   $$

   Où :
   - $w_{ij}$ est un poids qui dépend de la distance entre $\\mathbf{x}_i$ et $\\mathbf{x}_j$.

Le modèle RadiusNeighborsRegressor est particulièrement adapté pour des situations où la densité des données varie dans l'espace des caractéristiques, permettant des prédictions plus robustes dans des zones à densité variable.
""",
"GLM":"""
Le modèle GLM (Modèle Linéaire Généralisé) est une généralisation des modèles linéaires qui permet de modéliser une relation linéaire entre les variables d'entrée $\\mathbf{X}$ et une variable de sortie $y$ tout en prenant en compte différentes distributions de la variable de sortie. Un GLM se compose de trois éléments principaux :

1. **Fonction de lien :**

   La fonction de lien $g(\\mu)$ relie la moyenne $\\mu = \\mathbb{E}(y)$ de la variable de sortie à la combinaison linéaire des variables explicatives $\\mathbf{X}$ :

   $$
   g(\\mu) = \\mathbf{X} \\boldsymbol{\\beta}
   $$

   Où :
   - $g(\\mu)$ est la fonction de lien.
   - $\\mathbf{X}$ est la matrice des variables explicatives.
   - $\\boldsymbol{\\beta}$ est le vecteur des coefficients.

2. **Distribution de la famille exponentielle :**

   Le modèle GLM suppose que la variable de sortie $y$ suit une distribution de la famille exponentielle (par exemple, normale, binomiale, poisson) avec une moyenne $\\mu$ et une fonction de variance $V(\\mu)$.

   $$
   f(y; \\theta, \\phi) = \\exp\\left(\\frac{y \\theta - b(\\theta)}{a(\\phi)} + c(y, \\phi)\\right)
   $$

   Où :
   - $\\theta$ est le paramètre canonique (lié à $\\mu$).
   - $\\phi$ est le paramètre de dispersion.

3. **Fonction de variance :**

   La variance de la variable de sortie $y$ est une fonction de la moyenne $\\mu$ :

   $$
   \\text{Var}(y) = \\phi V(\\mu)
   $$

   Où :
   - $V(\\mu)$ est la fonction de variance.

4. **Exemples courants de GLM :**
   - **Régression Linéaire** : $y \\sim \\mathcal{N}(\\mu, \\sigma^2)$ avec $g(\\mu) = \\mu$ (fonction identité).
   - **Régression Logistique** : $y \\sim \\text{Bernoulli}(\\mu)$ avec $g(\\mu) = \\log\\left(\\frac{\\mu}{1 - \\mu}\\right)$ (logit).
   - **Régression de Poisson** : $y \\sim \\text{Poisson}(\\mu)$ avec $g(\\mu) = \\log(\\mu)$ (logarithme).

Le modèle GLM est extrêmement flexible et peut être adapté à différents types de données en choisissant la fonction de lien et la distribution appropriées.
""",
"CatBoostRegressor": """
Le modèle **CatBoostRegressor** est un modèle d'ensemble basé sur les arbres de décision, conçu pour les tâches de régression. Il est particulièrement efficace pour les ensembles de données contenant des variables catégorielles, car il intègre un encodage optimal de ces variables directement dans le processus d'entraînement.

1. **Modèle d'ensemble :**

   Le CatBoostRegressor construit un modèle en combinant plusieurs arbres de décision faibles, où chaque arbre est formé pour corriger les erreurs des arbres précédents. Le modèle final est une somme pondérée de ces arbres :

   $$
   \\hat{y} = \\sum_{m=1}^{M} \\lambda_m T_m(\\mathbf{x})
   $$

   Où :
   - $\\hat{y}$ est la valeur prédite.
   - $M$ est le nombre total d'arbres.
   - $\\lambda_m$ est le poids de l'arbre $T_m(\\mathbf{x})$.

2. **Traitement des variables catégorielles :**

   Le CatBoostRegressor traite les variables catégorielles sans nécessiter de prétraitement spécifique, comme l'encodage one-hot. Il utilise une technique appelée **encodage de moyenne cible** avec un schéma de permutation pour éviter le surapprentissage.

3. **Fonction de perte :**

   Le modèle minimise une fonction de perte basée sur l'erreur entre les valeurs observées $y_i$ et les valeurs prédites $\\hat{y}_i$. Pour la régression, une fonction de perte courante est l'erreur quadratique moyenne (MSE) :

   $$
   L(\\hat{y}, y) = \\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2
   $$

Le CatBoostRegressor est particulièrement adapté pour les données avec de nombreuses variables catégorielles, et il offre des performances robustes tout en réduisant le besoin de prétraitement des données.
""",
"CatBoostClassifier":"""
Le modèle **CatBoostClassifier** est un modèle d'ensemble basé sur les arbres de décision, conçu pour les tâches de classification. Il est particulièrement performant pour les ensembles de données contenant des variables catégorielles, grâce à son encodage optimal intégré.

1. **Modèle d'ensemble :**

   Le CatBoostClassifier combine plusieurs arbres de décision faibles pour former un modèle puissant. Chaque arbre est formé de manière séquentielle pour corriger les erreurs des arbres précédents. La classe prédite est déterminée par un vote pondéré des arbres :

   $$
   \\hat{y} = \\text{argmax}_k \\left(\\sum_{m=1}^{M} \\lambda_m T_m^k(\\mathbf{x})\\right)
   $$

   Où :
   - $\\hat{y}$ est la classe prédite.
   - $M$ est le nombre total d'arbres.
   - $\\lambda_m$ est le poids de l'arbre $T_m^k(\\mathbf{x})$ pour la classe $k$.

2. **Traitement des variables catégorielles :**

   Comme pour le CatBoostRegressor, le CatBoostClassifier intègre un encodage optimal des variables catégorielles, utilisant l'encodage de moyenne cible avec un schéma de permutation pour éviter le surapprentissage.

3. **Fonction de perte :**

   Le modèle minimise une fonction de perte adaptée à la classification. Pour la classification binaire, une fonction de perte courante est la **log loss** ou perte logarithmique :

   $$
   L(\\hat{y}, y) = -\\frac{1}{n} \\sum_{i=1}^{n} \\left[ y_i \\log(\\hat{y}_i) + (1 - y_i) \\log(1 - \\hat{y}_i) \\right]
   $$

Le CatBoostClassifier est un choix performant pour les problèmes de classification, particulièrement dans les cas où les données incluent de nombreuses variables catégorielles.
""",
"LGBMRegressor":"""
Le modèle **LGBMRegressor** est un modèle d'ensemble basé sur le **Gradient Boosting**, optimisé pour la performance et l'efficacité. Il construit une série d'arbres de décision, où chaque nouvel arbre corrige les erreurs commises par les arbres précédents.

1. **Modèle d'ensemble :**

   Le LGBMRegressor combine plusieurs arbres de décision pour faire des prédictions. La prédiction finale est obtenue en sommant les contributions de chaque arbre :

   $$
   \\hat{y} = \\sum_{m=1}^{M} \\lambda_m T_m(\\mathbf{x})
   $$

   Où :
   - $\\hat{y}$ est la valeur prédite.
   - $M$ est le nombre total d'arbres.
   - $\\lambda_m$ est le poids de l'arbre $T_m(\\mathbf{x})$.
   - $T_m(\\mathbf{x})$ est la prédiction de l'arbre $m$ pour l'entrée $\\mathbf{x}$.

2. **Fonction de perte :**

   Le modèle minimise une fonction de perte basée sur l'erreur entre les valeurs observées $y_i$ et les valeurs prédites $\\hat{y}_i$. Une fonction de perte typique pour la régression est l'erreur quadratique moyenne (MSE) :

   $$
   L(\\hat{y}, y) = \\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2
   $$

3. **Optimisation par histogramme :**

   LightGBM utilise une technique d'optimisation par histogramme qui divise les valeurs des caractéristiques en bins (ou seaux) pour accélérer le processus de construction des arbres tout en réduisant la consommation de mémoire.

Le LGBMRegressor est particulièrement adapté pour les grandes bases de données et les tâches nécessitant une grande rapidité d'exécution tout en maintenant une haute performance prédictive.
""",
"LGBMClassifier":"""
Le modèle **LGBMClassifier** est un classificateur basé sur le **Gradient Boosting**, conçu pour offrir une haute performance tout en étant extrêmement rapide et efficace en termes de mémoire. Comme le LGBMRegressor, il construit une série d'arbres de décision, où chaque arbre est formé pour corriger les erreurs des arbres précédents.

1. **Modèle d'ensemble :**

   Le LGBMClassifier combine plusieurs arbres de décision faibles pour faire une prédiction finale. Chaque arbre contribue à la décision finale, qui est une somme pondérée des prédictions des arbres :

   $$
   \\hat{y} = \\text{argmax}_k \\left(\\sum_{m=1}^{M} \\lambda_m T_m^k(\\mathbf{x})\\right)
   $$

   Où :
   - $\\hat{y}$ est la classe prédite.
   - $M$ est le nombre total d'arbres.
   - $\\lambda_m$ est le poids de l'arbre $T_m^k(\\mathbf{x})$ pour la classe $k$.
   - $T_m^k(\\mathbf{x})$ est la prédiction de l'arbre $m$ pour la classe $k$.

2. **Fonction de perte :**

   Le modèle minimise une fonction de perte adaptée à la classification, comme la **log loss** pour la classification binaire :

   $$
   L(\\hat{y}, y) = -\\frac{1}{n} \\sum_{i=1}^{n} \\left[ y_i \\log(\\hat{y}_i) + (1 - y_i) \\log(1 - \\hat{y}_i) \\right]
   $$

3. **Optimisation par histogramme :**

   LightGBM divise les valeurs des caractéristiques en bins (ou seaux) pour construire les arbres plus rapidement et avec une efficacité mémoire accrue, ce qui le rend particulièrement performant pour les grands ensembles de données.

Le LGBMClassifier est particulièrement efficace pour les tâches de classification avec de grandes quantités de données, où la rapidité et la performance sont cruciales.
""",
"ExplainableBoostingRegressor":"""
Le modèle **ExplainableBoostingRegressor** (EBM) est un modèle de régression basé sur les **Generalized Additive Models (GAMs)**, conçu pour offrir des prédictions précises tout en restant interprétable. Il utilise une combinaison d'arbres de décision boostés pour modéliser chaque caractéristique de manière additive, ce qui permet de comprendre facilement l'impact de chaque caractéristique sur la prédiction.

1. **Modèle Additif Généralisé :**

   Le modèle EBM pour la régression est exprimé comme une somme des effets individuels de chaque caractéristique :

   $$
   \\hat{y} = \\beta_0 + \\sum_{j=1}^{p} f_j(x_j)
   $$

   Où :
   - $\\hat{y}$ est la valeur prédite.
   - $\\beta_0$ est l'intercept (constante).
   - $f_j(x_j)$ est la fonction de forme pour la caractéristique $x_j$, qui est modélisée par un ensemble d'arbres de décision.

2. **Fonctions de forme :**

   Chaque fonction $f_j(x_j)$ est formée indépendamment pour chaque caractéristique $x_j$, ce qui permet de capturer la relation spécifique entre $x_j$ et la variable cible $y$. Cela rend le modèle interprétable, car l'impact de chaque caractéristique est isolé.

3. **Fonction de perte :**

   Le modèle minimise une fonction de perte adaptée à la régression, telle que l'erreur quadratique moyenne (MSE) :

   $$
   L(\\hat{y}, y) = \\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2
   $$

4. **Interprétabilité :**

   Le principal avantage de l'EBM est son interprétabilité. Chaque fonction de forme $f_j(x_j)$ peut être visualisée pour montrer comment chaque caractéristique affecte la prédiction, permettant aux utilisateurs de comprendre et de faire confiance aux décisions du modèle.

Le modèle ExplainableBoostingRegressor est particulièrement utile pour les applications où la transparence et l'interprétabilité sont aussi importantes que la précision.
""",
"ExplainableBoostingClassifier":"""
Le modèle **ExplainableBoostingClassifier** (EBM) est un classificateur basé sur les **Generalized Additive Models (GAMs)**, conçu pour fournir des prédictions précises et interprétables. Comme l'EBM pour la régression, il modélise chaque caractéristique de manière additive en utilisant des arbres de décision boostés.

1. **Modèle Additif Généralisé :**

   Le modèle EBM pour la classification est exprimé comme une somme des effets individuels de chaque caractéristique, avec une fonction de lien logistique pour produire des probabilités :

   $$
   \\log\\left(\\frac{P(y=1)}{1-P(y=1)}\\right) = \\beta_0 + \\sum_{j=1}^{p} f_j(x_j)
   $$

   Où :
   - $P(y=1)$ est la probabilité que $y$ prenne la valeur 1.
   - $\\beta_0$ est l'intercept (constante).
   - $f_j(x_j)$ est la fonction de forme pour la caractéristique $x_j$, modélisée par un ensemble d'arbres de décision.

2. **Fonctions de forme :**

   Chaque fonction $f_j(x_j)$ est ajustée pour capturer la relation spécifique entre la caractéristique $x_j$ et la probabilité de la classe cible, ce qui permet une interprétation facile de l'effet de chaque caractéristique.

3. **Fonction de perte :**

   Le modèle minimise une fonction de perte adaptée à la classification binaire, telle que la **log loss** :

   $$
   L(\\hat{y}, y) = -\\frac{1}{n} \\sum_{i=1}^{n} \\left[ y_i \\log(\\hat{y}_i) + (1 - y_i) \\log(1 - \\hat{y}_i) \\right]
   $$

4. **Interprétabilité :**

   L'EBM pour la classification est hautement interprétable, permettant de visualiser l'impact de chaque caractéristique sur les probabilités de la classe cible. Cela rend le modèle transparent et compréhensible pour les utilisateurs finaux.

Le modèle ExplainableBoostingClassifier est idéal pour les cas où il est crucial de comprendre et d'expliquer les décisions du modèle, en plus de fournir des prédictions fiables.
""",
"XGBRegressor":"""
Le modèle **XGBRegressor** est un modèle de régression basé sur l'**Extreme Gradient Boosting (XGBoost)**, une méthode puissante d'**ensemble** qui combine les prédictions de plusieurs arbres de décision pour améliorer la précision.

1. **Modèle d'ensemble :**

   Le XGBRegressor construit un ensemble d'arbres de décision de manière séquentielle, où chaque nouvel arbre est formé pour corriger les erreurs des arbres précédents. La prédiction finale est la somme pondérée des prédictions de tous les arbres :

   $$
   \\hat{y} = \\sum_{m=1}^{M} \\lambda_m T_m(\\mathbf{x})
   $$

   Où :
   - $\\hat{y}$ est la valeur prédite.
   - $M$ est le nombre total d'arbres.
   - $\\lambda_m$ est le poids associé à l'arbre $T_m(\\mathbf{x})$.
   - $T_m(\\mathbf{x})$ est la prédiction de l'arbre $m$ pour l'entrée $\\mathbf{x}$.

2. **Fonction de perte :**

   Le XGBRegressor minimise une fonction de perte qui mesure l'erreur entre les valeurs observées $y_i$ et les valeurs prédites $\\hat{y}_i$. Une fonction de perte courante pour la régression est l'erreur quadratique moyenne (MSE) :

   $$
   L(\\hat{y}, y) = \\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2
   $$

3. **Régularisation :**

   Le XGBRegressor intègre des termes de régularisation pour éviter le surapprentissage. La régularisation $L1$ et $L2$ sur les coefficients des arbres permet de contrôler la complexité du modèle.

4. **Optimisation par Gradient Boosting :**

   XGBoost optimise le modèle en utilisant une méthode de **boosting par gradient**, qui ajuste chaque nouvel arbre pour réduire l'erreur de gradient de la fonction de perte.

Le XGBRegressor est particulièrement apprécié pour sa flexibilité, sa capacité à gérer des données complexes, et sa performance élevée sur une large variété de tâches de régression.
""",
"XGBClassifier":"""
Le modèle **XGBClassifier** est un classificateur basé sur l'**Extreme Gradient Boosting (XGBoost)**, qui est une méthode d'ensemble puissante combinant plusieurs arbres de décision pour effectuer des prédictions robustes.

1. **Modèle d'ensemble :**

   Le XGBClassifier construit un ensemble d'arbres de décision de manière séquentielle, où chaque arbre est formé pour corriger les erreurs des arbres précédents. La prédiction finale est déterminée par un vote pondéré des arbres :

   $$
   \\hat{y} = \\text{argmax}_k \\left(\\sum_{m=1}^{M} \\lambda_m T_m^k(\\mathbf{x})\\right)
   $$

   Où :
   - $\\hat{y}$ est la classe prédite.
   - $M$ est le nombre total d'arbres.
   - $\\lambda_m$ est le poids de l'arbre $T_m^k(\\mathbf{x})$ pour la classe $k$.
   - $T_m^k(\\mathbf{x})$ est la prédiction de l'arbre $m$ pour la classe $k$.

2. **Fonction de perte :**

   Le modèle minimise une fonction de perte adaptée à la classification, comme la **log loss** :

   $$
   L(\\hat{y}, y) = -\\frac{1}{n} \\sum_{i=1}^{n} \\left[ y_i \\log(\\hat{y}_i) + (1 - y_i) \\log(1 - \\hat{y}_i) \\right]
   $$

3. **Régularisation :**

   Le XGBClassifier utilise des techniques de régularisation $L1$ et $L2$ pour prévenir le surapprentissage et améliorer la généralisation du modèle.

4. **Optimisation par Gradient Boosting :**

   XGBoost optimise le modèle en utilisant une méthode de **boosting par gradient**, qui ajuste chaque nouvel arbre pour réduire l'erreur de gradient de la fonction de perte.

Le XGBClassifier est très performant pour les tâches de classification, surtout lorsqu'il s'agit de données complexes ou de grandes dimensions, grâce à sa capacité à gérer des interactions complexes entre les caractéristiques.
""",
}