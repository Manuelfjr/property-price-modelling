from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
import warnings

def perform_grid_search(X, y, model, param_grid, scorer, cv=5):
    grid_search = GridSearchCV(model, param_grid, scoring=scorer, cv=cv)

    with tqdm(total=len(param_grid)) as pbar:
        def update_pbar(*args):
            pbar.update()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid_search.fit(X, y)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    return best_model, best_params