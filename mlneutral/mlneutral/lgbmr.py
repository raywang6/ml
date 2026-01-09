import numpy as np
import lightgbm as lgb
import copy
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error


def prepare_lgbm():
    """
    Prepare LightGBM model class and default parameters.

    Returns:
        Tuple of (model_class, default_params)
    """
    standard_params = {
        'n_estimators': 500,
        'boosting_type': 'gbdt',
        #'num_leaves': 256,  # Increased for better expressiveness
        'random_state': 61,
        #'max_depth': 12,
        #'reg_lambda': 0.,
        #'reg_alpha': 0.,  # Added L1 regularization
        #'early_stopping_rounds': 20,
        #'subsample': 0.8,  # Use subsample instead of bagging_fraction
        #'subsample_freq': 5,  # Renamed from bagging_freq
        #'colsample_bytree': 0.8,  # Feature sampling
        #'min_child_samples': 20,  # Prevent overfitting
    }
    return LGBMRegressor, standard_params



def sign_sensitive_objective(y_pred, dataset):
    """
    Custom objective: squared error plus an extra penalty term
    for sign mismatches.

    Fixed version that handles edge cases and ensures valid gradients.
    """
    y_true = dataset.get_label()

    # Basic gradient & hessian for squared error
    residual = y_pred - y_true
    grad = residual.copy()

    # Hessian should be positive and not too small for numerical stability
    # Using 2.0 as base (second derivative of squared loss)
    hess = np.full_like(grad, 2.0)

    # Only apply sign penalty when predictions are non-trivial
    # Skip when predictions are near zero (early training)
    pred_magnitude = np.abs(y_pred)
    significant_pred = pred_magnitude > 1e-6

    # Identify where sign of pred != sign of true (only for significant predictions)
    wrong_sign = significant_pred & (np.sign(y_pred) != np.sign(y_true))

    # Apply penalty (upweight the gradient for wrong-sign predictions)
    penalty = 2.0  # Reduced from 3 for stability
    grad[wrong_sign] *= (1 + penalty)
    hess[wrong_sign] *= (1 + penalty)

    # Ensure hessians are never too small (prevents -inf gain)
    hess = np.maximum(hess, 1e-3)

    return grad, hess

def sign_sensitive_eval(y_pred, dataset):
    """
    Custom eval metric if you also want to monitor "sign error rate".
    """
    y_true = dataset.get_label()
    bad = (np.sign(y_pred) != np.sign(y_true)).sum()
    rate = bad / len(y_true)
    return 'sign_error_rate', rate, False  # False = lower is better



def rmse_objective(y_pred, dataset):
    """
    Custom objective: squared error plus an extra penalty term
    for sign mismatches.
    """
    y_true = dataset.get_label()

    # Basic gradient & hessian for squared error
    grad = y_pred - y_true
    hess = np.ones_like(grad)

    return grad, hess


def rmse_eval(y_pred, dataset):
    """
    Custom eval metric if you also want to monitor "sign error rate".
    """
    y_true = dataset.get_label()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return 'sign_error_rate', rmse, False  # False = lower is better



class LGBMRegressor(object):
    # Valid LightGBM parameters (filter out unknown ones)
    VALID_PARAMS = {
        'boosting_type', 'num_leaves', 'max_depth', 'learning_rate',
        'n_estimators', 'subsample_for_bin', 'objective', 'class_weight',
        'min_split_gain', 'min_child_weight', 'min_child_samples',
        'subsample', 'subsample_freq', 'colsample_bytree', 'reg_alpha',
        'reg_lambda', 'random_state', 'n_jobs', 'importance_type',
        'bagging_fraction', 'bagging_freq', 'feature_fraction',
        'verbose', 'metric', 'num_iterations', 'num_boost_round',
        'force_col_wise', 'force_row_wise', 'seed',
        'early_stopping_rounds',  # Handled specially but allowed
        'min_data_in_leaf', 'lambda_l1', 'lambda_l2',
    }

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=-1,
        num_leaves=31,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=0.0,
        early_stopping_rounds=10,
        random_state=61,
        add_sign_penalty=True,
        **kwargs
    ):
        """
        LightGBM regressor wrapper with optional custom objective and eval metric.
        """
        # Filter out unknown parameters to avoid warnings
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in self.VALID_PARAMS}
        ignored_params = set(kwargs.keys()) - set(filtered_kwargs.keys())
        if ignored_params:
            print(f"[LGBMRegressor] Ignoring unknown parameters: {ignored_params}")

        # Handle subsample vs bagging_fraction conflict
        # If bagging_fraction is provided, use it; otherwise use subsample
        if 'bagging_fraction' in filtered_kwargs:
            subsample = filtered_kwargs.pop('bagging_fraction')

        # Set num_leaves appropriately if max_depth is set
        if max_depth > 0 and num_leaves == 31:
            # Recommended: num_leaves <= 2^max_depth
            num_leaves = min(2 ** max_depth, 1024)

        # Store init params
        self._params = {
            'objective': 'regression',
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'num_leaves': num_leaves,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'random_state': random_state,
            'verbose': -1,  # Reduce verbosity
            **filtered_kwargs
        }
        
        # --- POTENTIAL ERROR/IMPROVEMENT 1: `early_stopping_rounds` is a local var in init, not instance var ---
        self.early_stopping_rounds = early_stopping_rounds # You might want to store this if you intend to save it as a direct attribute

        self.early_stopping_callback = lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True)
        
        self.add_sign_penalty = add_sign_penalty
        
        # --- POTENTIAL ERROR/IMPROVEMENT 2: Set feval in init if using custom objective ---
        self.feval = None # Initialize feval to None
        #if add_sign_penalty:
        #    self._params['objective'] = sign_sensitive_objective
        #    self.feval = sign_sensitive_eval # Assign the custom eval function here
        #    # When using a custom objective, it's often good practice to set 'metric' to 'None'
        #    # in params to avoid LightGBM trying to compute a default metric if your custom feval handles it.
        #    self._params['metric'] = 'None' # Or choose an appropriate built-in metric if desired alongside your feval

        self._booster = None
        self.model_namecard = {}

    def fit(self, X, y, validation_data=None, sample_weight=None):
        """
        Fit the model using either sklearn wrapper or lightgbm.train when custom objective is provided.
        """
        # If custom objective or eval metric provided, use lgb.train
        if self.add_sign_penalty:
            train_set = lgb.Dataset(X, label=y, weight=sample_weight, free_raw_data=False)
            
            # --- POTENTIAL ERROR/IMPROVEMENT 3: Handling validation_data for lgb.train ---
            valid_sets = []
            valid_names = []
            if validation_data is not None:
                X_val, y_val = validation_data
                valid_sets.append(lgb.Dataset(X_val, label=y_val, free_raw_data=False))
                valid_names.append('valid') # The name for the validation set
            print(f"[debug!]: {self._params}")
            init_params = self._params.copy()
            init_params['learning_rate'] = init_params['learning_rate']# * 5
            clean_params = {k: v for k, v in init_params.items() 
                if k not in ['num_boost_round', 'n_estimators', 'early_stopping_rounds']}

            warmup_booster = lgb.train(
                clean_params,
                train_set,
                num_boost_round=1,
                valid_sets=valid_sets,
                valid_names=valid_names,
                # Remove early_stopping_callback here so it cannot stop
                callbacks=[lgb.log_evaluation(period=1)] 
            )

            # --- Step 2: Main Training (Resume with Early Stopping) ---
            self._booster = lgb.train(
                self._params,
                train_set,
                # Subtract the 5 rounds we already did
                num_boost_round=self._params['n_estimators'], 
                valid_sets=valid_sets,
                valid_names=valid_names,
                # KEY: This loads the trees from Step 1 so we don't start from scratch
                init_model=warmup_booster, 
                # Now we add the early stopper back in
                callbacks=[self.early_stopping_callback, lgb.log_evaluation(period=10)]
            )
            # After training, it's good practice to set best_iteration for prediction
            # This is automatically handled if early stopping occurs.
            #if self.early_stopping_callback.best_iteration is not None:
            #    self._booster.best_iteration = self.early_stopping_callback.best_iteration

        else:
            # fall back to sklearn API
            # --- POTENTIAL ERROR/IMPROVEMENT 4: `n_estimators` for LGBMRegressor init ---
            # Ensure _params doesn't contain 'objective' key directly if falling back to sklearn API
            # since the sklearn wrapper expects the objective as a string.
            print("[error]: using sklearn api!!")
            sklearn_params = self._params.copy()
            if 'objective' in sklearn_params and callable(sklearn_params['objective']):
                # If custom objective was set in _params, remove it for sklearn API
                del sklearn_params['objective']
                # You might need to set a default objective for sklearn API if no custom one is used
                sklearn_params['objective'] = 'regression' # Default for LGBMRegressor

            self._lgbm = lgb.LGBMRegressor(**sklearn_params) # Pass the cleaned params

            fit_kwargs = {'X': X, 'y': y, 'sample_weight': sample_weight}
            if validation_data is not None:
                X_val, y_val = validation_data
                fit_kwargs.update({
                    'eval_set': [(X_val, y_val)],
                    'eval_metric': 'rmse', # Ensure this matches your _params['metric'] if not None
                    'early_stopping_rounds': self.early_stopping_rounds, #[self.early_stopping_callback], # Pass early stopping here too for sklearn API
                    'verbose': True # This might be redundant if using verbose=True in callback
                })
            
            # --- POTENTIAL ERROR 5: early_stopping_rounds in sklearn API ---
            # For sklearn API, `early_stopping_rounds` is a direct parameter to `fit`,
            # not a callback. Or you need to use a callback for the sklearn API as well.
            # However, the direct `early_stopping_rounds` parameter in `fit` is still common.
            # Let's adjust this to use the old style if no custom objective, for simplicity
            # (though callbacks are more consistent).
            if not self.add_sign_penalty and validation_data is not None:
                # Remove callbacks from fit_kwargs if using early_stopping_rounds directly
                fit_kwargs.pop('callbacks', None)

            self._lgbm.fit(**fit_kwargs)
            self._booster = self._lgbm.booster_

        if self.add_sign_penalty:
            self.model_namecard['epoch'] = self._booster.best_iteration
            self.model_namecard['acc_validate'] = self._booster.best_score
            print(f"finish training: {self._booster.best_iteration} early stop, with best {self._booster.best_score}")

        return self

    def predict(self, X):
        """
        Predict continuous targets for X.
        """
        # --- POTENTIAL ERROR 6: `best_iteration` might not be set if early stopping didn't happen ---
        # Or if training completed all rounds
        if self._booster is None:
            raise RuntimeError("Model has not been fitted yet.")
        
        # Check if best_iteration is available, otherwise predict with all iterations
        num_iteration_to_use = self._booster.best_iteration if hasattr(self._booster, 'best_iteration') and self._booster.best_iteration is not None else None
        
        return self._booster.predict(X, num_iteration=num_iteration_to_use)

    def evaluate(self, X, y):
        """
        Compute RMSE on (X, y).
        """
        preds = self.predict(X)
        #if self.feval is None:
        rmse = np.sqrt(mean_squared_error(y, preds))
        #else:
        #    dataset = lgb.Dataset(X, label=y)
        #    _, rmse, _ = self.feval(preds, dataset)
        return rmse, rmse

    def save(self, path):
        """
        Save booster and init params.
        """
        if self._booster is None:
            raise RuntimeError("Model has not been fitted yet.")

        # booster
        self._booster.save_model(path + '.txt')
        joblib.dump(self.model_namecard, path + '_namecard')
        
        # --- POTENTIAL ERROR 7: `self.early_stopping_rounds` might not exist ---
        # You didn't store `early_stopping_rounds` as an instance variable `self.early_stopping_rounds`
        # in `__init__`. You only used it to initialize the callback.
        # It's better to save the `stopping_rounds` directly from the callback.
        # Also, `feval` might be `None`.
        joblib.dump({
            'params': self._params,
            'stopping_rounds_for_callback': self.early_stopping_rounds, # Save the actual value
            'add_sign_penalty': self.add_sign_penalty, # This needs to be saved to recreate the logic
            # 'feval': self.feval # Don't save feval directly, it's a function and can cause issues with pickling/joblib
                                # Instead, recreate its state based on `add_sign_penalty`
        }, path + '_lgbmparams.pkl')

    @classmethod
    def load(cls, path):
        """
        Load wrapper including custom objective settings.
        """
        meta = joblib.load(path + '_lgbmparams.pkl')
        
        # --- POTENTIAL ERROR 8: Reconstructing the object and its state ---
        # Reconstruct the class parameters correctly
        # The 'feval' and 'objective' should be recreated based on 'add_sign_penalty'
        
        # Create a dictionary for __init__ parameters
        init_kwargs = meta['params'].copy()
        init_kwargs['early_stopping_rounds'] = meta['stopping_rounds_for_callback']
        init_kwargs['add_sign_penalty'] = meta['add_sign_penalty']
        
        # Remove objective if it was a custom function during saving, as it's set by `add_sign_penalty`
        # during __init__
        if 'objective' in init_kwargs and callable(init_kwargs['objective']):
            del init_kwargs['objective']

        obj = cls(**init_kwargs) # Initialize the object

        obj._booster = lgb.Booster(model_file=path + '.txt')
        return obj
