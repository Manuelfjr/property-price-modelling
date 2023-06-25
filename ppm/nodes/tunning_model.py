from sklearn.model_selection import GridSearchCV, KFold
from tqdm import tqdm
from numpy import ndarray
import warnings

def perform_grid_search(
        X: ndarray,
        y: ndarray, 
        model: list, 
        param_grid: dict, 
        scorer: object, 
        cv: int = 5
        ) -> tuple:
    grid_search = GridSearchCV(model, param_grid, scoring=scorer, cv=cv)
    
    with tqdm(total=cv) as pbar:
        def update_pbar():
            pbar.update()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Create cross-validation folds
            kf = KFold(n_splits=cv)
            
            # Loop through each fold
            for train_index, _ in kf.split(X, y):
                X_train, y_train = X[train_index], y[train_index]
                grid_search.fit(X_train, y_train)
                update_pbar()

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    return best_model, best_params