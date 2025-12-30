
import polars as pl
from .types import Dict, List, ModelType, ParamType, PathType, Callable
from hyperopt import tpe, hp, fmin, STATUS_OK, space_eval, pyll
import joblib
import gc
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
import os

#from .lightgbm import (
#    LGBMClassifier, 
#    prepare_lgbm
#    )

#%% training:
# split the training dataset into train + valid * 6
# for each train+valid set
# run the loss, compute the objective function for hyperopt
# select the hyper param
# store the model for last two batch for each hyper params, restore

def compute_batchsize(train):
    return (2 ** int(np.log2(max(len(train)//40000, 1)))) * 512


def translate_hpspace(config):
    hpspace = {}
    param_config = config['training']['hyperparameters']
    
    for name, spec in param_config.items():
        if spec['type'] == 'uniform':
            hpspace[name] = hp.uniform(name, float(spec['min']), float(spec['max']))
        elif spec['type'] == 'loguniform':
            hpspace[name] = hp.loguniform(name, np.log(float(spec['min'])), np.log(float(spec['max'])))
        elif spec['type'] == 'quniform':
            hpspace[name] = hp.quniform(name, float(spec['min']), float(spec['max']), int(spec['step']))
        elif spec['type'] == 'choice':
            hpspace[name] = hp.choice(name, spec['options'])
        elif spec['type'] == 'randint':
            # draw integer in [min, max]
            low = int(spec['min'])
            high = int(spec['max'])
            # hp.randint(label, upper) draws 0..upper-1, so shift by low
            hpspace[name] = hp.randint(name, high - low + 1) + low
        else:
            raise ValueError(f"Unsupported param type: {spec['type']}")
    
    return hpspace

def train_classifier(
        training_set: pl.DataFrame,
        target: str,
        features: List[str],
        modelClass: ModelType,
        params: ParamType,
        save_path: PathType,
        config: Dict,
        name: str = '',
        use_sw: bool = True,
        no_early_stop_before: int = 10
    ):
    buff = config['training']['overlap_buff']
    training_set = training_set.with_columns(
        pl.col(i).fill_nan(None).fill_null(0.0).alias(i) for i in features
    )
    training_set = training_set.fill_nan(None).drop_nulls(target)
    
    # Get unique sorted dates for proper time-series splitting
    unique_dates = training_set.select('datetime').unique().sort('datetime')['datetime'].to_list()
    n_dates = len(unique_dates)
    
    if n_dates < 20:  # Need minimum dates for train/valid splits
        raise ValueError(f"Insufficient unique dates: {n_dates}")
    
    # Calculate date indices for splits
    min_valid_dates = max(5, n_dates // 8)  # At least 5 dates per validation fold
    
    if n_dates >= config['training']['min_datasize_thres']:  # Rough scaling
        n_train_dates = n_dates - 4 * min_valid_dates  # Reserve space for 6 validation folds
    else:
        n_train_dates = n_dates // 2
        min_valid_dates = n_dates // 8
    
    hpspace = translate_hpspace(config)
    
    def objective(hpcomb):
        params.update(hpcomb)
        scores = []
        print(f"[trial]: {params}")
        
        for idvalid in range(1, 4):
            # Calculate date cutoffs
            train_end_idx = n_train_dates + min_valid_dates * idvalid - buff
            valid_start_idx = n_train_dates + min_valid_dates * idvalid
            valid_end_idx = n_train_dates + min_valid_dates * (idvalid + 1)
            
            train_end_date = unique_dates[train_end_idx]
            valid_start_date = unique_dates[valid_start_idx]
            valid_end_date = unique_dates[min(valid_end_idx, n_dates - 1)]
            
            # Split by datetime, not index
            train_data = training_set.filter(pl.col('datetime') < train_end_date)
            valid_data = training_set.filter(
                (pl.col('datetime') >= valid_start_date) & 
                (pl.col('datetime') < valid_end_date)
            )
            
            train_X = train_data.select(features).to_numpy()
            train_y = train_data.select(target).to_numpy()
            valid_X = valid_data.select(features).to_numpy()
            valid_y = valid_data.select(target).to_numpy()
            
            model = modelClass(**params)
            
            if use_sw:
                sw = train_data.select(f"sw_{target}").to_numpy()
            else:
                sw = None
            
            model.fit(train_X, train_y, validation_data=(valid_X, valid_y), sample_weight=sw)
            score, acc = model.evaluate(valid_X, valid_y)
            
            if idvalid >= 2 and model.model_namecard['epoch'] < no_early_stop_before:
                acc += 10
            scores.append(acc)
            del model
            gc.collect()
        
        output_loss = 0.2*scores[0] + 0.3*scores[1] + 0.5*scores[2]
        print(f"[trial] {np.mean(scores)}")
        return {'loss': output_loss, 'status': STATUS_OK, 'score_list': scores}

    max_evals = config['training']['max_hp_evals']

    if max_evals > 1:
        bestidx = fmin(
            fn=objective,
            space=hpspace,
            algo=tpe.suggest,
            max_evals=max_evals,
        )
        best = space_eval(hpspace, bestidx)
    else:
        # Single eval: sample one point and run objective directly
        best = pyll.stochastic.sample(hpspace)
        objective(best) 

    print(f"[param]: {save_path}: {best}")
    params.update(best)
    
    # Train final models with date-based splits
    for model_id, idvalid in enumerate([2, 3], start=1):
        train_end_idx = n_train_dates + min_valid_dates * idvalid - buff
        valid_start_idx = n_train_dates + min_valid_dates * idvalid
        valid_end_idx = n_train_dates + min_valid_dates * (idvalid + 1)
        
        train_end_date = unique_dates[train_end_idx]
        valid_start_date = unique_dates[valid_start_idx]
        valid_end_date = unique_dates[min(valid_end_idx, n_dates - 1)]
        
        train_data = training_set.filter(pl.col('datetime') < train_end_date)
        valid_data = training_set.filter(
            (pl.col('datetime') >= valid_start_date) & 
            (pl.col('datetime') < valid_end_date)
        )
        
        train_X = train_data.select(features).to_numpy()
        train_y = train_data.select(target).to_numpy()
        valid_X = valid_data.select(features).to_numpy()
        valid_y = valid_data.select(target).to_numpy()
        
        model = modelClass(**params)
        sw = train_data.select(f"sw_{target}").to_numpy() if use_sw else None
        model.fit(train_X, train_y, validation_data=(valid_X, valid_y), sample_weight=sw)
        model.save(os.path.join(save_path, f'{name}_{target}__{model_id}'))

def train_classifier_simple(
        training_set: pl.DataFrame,
        target: str,
        features: List[str],
        modelClass: ModelType,
        params: ParamType,
        save_path: PathType,
        config: Dict,
        name: str = '',
        batch_size: int = 128,
        sampler: Callable = None,
        use_sw = True,
    ):
    buff = config['training']['overlap_buff']
    training_set = training_set.with_columns(
        pl.col(i).fill_nan(None).fill_null(0.0).alias(i) for i in features
    )
    training_set = training_set.fill_nan(None).drop_nulls(target)
    X_ = training_set.select(features).to_numpy()[:-buff]
    y_ = training_set.select(target).to_numpy()[:-buff]
    N = len(X_)
    if use_sw:
        sw_ = training_set.select(f"sw_{target}").to_numpy()[:-buff]
    else:
        sw_ = None
    # split dataset
    if N >= config['training']['min_datasize_thres']:
        ntrain = int(N - config['training']['min_datasize_thres']/2)
        nvalid = int(config['training']['min_datasize_thres'] / 12)
    else:
        ntrain = int(N / 2)
        nvalid = int(N / 12)
    # preprocessing y
    y_transformer = KBinsDiscretizer(n_bins=config['training']['class_size'], encode='ordinal', strategy='quantile')
    y_transformer.fit(y_[:ntrain])
    joblib.dump(y_transformer, os.path.join(save_path, '_'.join(['y_transformer',name,target])))
    #hpspace = {hpname: hp.choice(hpname, hplist) for hpname, hplist in config['training']['hyperparameters'].items()}
    #def objective(hpcomb):
    # train predict1
    idvalid = 4
    train_X = X_[:ntrain+nvalid*idvalid - buff]
    train_y = y_[:ntrain+nvalid*idvalid - buff]
    valid_X = X_[ntrain+nvalid*idvalid:ntrain+nvalid*(idvalid+1)]
    valid_y = y_[ntrain+nvalid*idvalid:ntrain+nvalid*(idvalid+1)]
    #params['batch_size'] = compute_batchsize(train_X)
    model1 = modelClass(**params) 
    if use_sw:
        sw = sw_[:ntrain+nvalid*idvalid - buff] # should i store the sw?
    else:
        sw = None
    train_y = y_transformer.transform(train_y)
    valid_y = y_transformer.transform(valid_y)
    # debug
    #joblib.dump(train_X, os.path.join(save_path, '_'.join(['debug_train_X',name])))
    #joblib.dump(train_y, os.path.join(save_path, '_'.join(['debug_train_y',name])))
    #joblib.dump(valid_X, os.path.join(save_path, '_'.join(['debug_valid_X',name])))
    #joblib.dump(valid_y, os.path.join(save_path, '_'.join(['debug_valid_y',name])))
    print(f"[debug]: train set 1 total {train_y.shape[0]}, pos {(train_y==2).mean()}, neg {(train_y==0).mean()}")
    model1.fit(train_X, train_y, validation_data = (valid_X, valid_y), sample_weight = sw, sampler = sampler)    
    model1.save(os.path.join(save_path, '_'.join([name,target,'_1'])))
    # train predict2
    idvalid = 5
    train_X = X_[:ntrain+nvalid*idvalid - buff]
    train_y = y_[:ntrain+nvalid*idvalid - buff]
    valid_X = X_[ntrain+nvalid*idvalid:ntrain+nvalid*(idvalid+1)]
    valid_y = y_[ntrain+nvalid*idvalid:ntrain+nvalid*(idvalid+1)]
    #params['batch_size'] = batch_size#compute_batchsize(train_X)
    model2 = modelClass(**params) 
    if use_sw:
        sw = sw_[:ntrain+nvalid*idvalid - buff] # should i store the sw?
    else:
        sw = None
    train_y = y_transformer.transform(train_y)
    valid_y = y_transformer.transform(valid_y)
    print(f"[debug]: train set 2 total {train_y.shape[0]}, pos {(train_y==2).mean()}, neg {(train_y==0).mean()}")
    model2.fit(train_X, train_y, validation_data = (valid_X, valid_y), sample_weight = sw, sampler = sampler)
    model2.save(os.path.join(save_path, '_'.join([name,target,'_2'])))
    

def predict_classifier(
        model_class, 
        test_set: pl.DataFrame,
        target: str,
        features: List[str],
        save_path: PathType,
        name: str = ''
    ) -> pl.DataFrame:
    test_set = test_set.with_columns(
        pl.col(i).fill_nan(None).fill_null(0.0).alias(i) for i in features
    )
    X_ = test_set.select(features).to_numpy()
    # load preprocess
    X_scaler = joblib.load(os.path.join(save_path, '_'.join(['X_scaler',name,target])))
    trans_X = X_scaler.transform(X_)
    #y_transformer = joblib.load(os.path.join(save_path, '_'.join(['y_transformer',name,target])))
    model1 = model_class.load(os.path.join(save_path, '_'.join([name,target,'_1'])))
    model2 = model_class.load(os.path.join(save_path, '_'.join([name,target,'_2'])))
    pred1 = model1.predict(trans_X)
    pred2 = model2.predict(trans_X)
    pred = (pred1 + pred2)/2
    test_set = test_set.with_columns(
            pl.Series(pred[:,0]).alias(f'pred_{target}_neg'),
            pl.Series(pred[:,1]).alias(f'pred_{target}_neu'),
            pl.Series(pred[:,2]).alias(f'pred_{target}_pos'),
        )
    return test_set



#%%%%
# archive ----------------------------------------
def train_regressor(
        training_set: pl.DataFrame,
        targets: List[str],
        features: List[str],
        model: ModelType,
        params: ParamType,
        save_path: PathType,
        name: str = ''
    ):
    training_set = training_set.with_columns(
        pl.col(i).fill_nan(None).fill_null(0.0).alias(i) for i in features
    )
    training_set = training_set.drop_nulls(targets)
    X_ = training_set.select(features).to_numpy()
    for target in targets:
        y_ = training_set.select(target).to_numpy()
        model0 = model(**params)
        model0.fit(X_,y_)
        model0.save(os.path.join(save_path, '_'.join([name,target])))
    
def predict_regressor(
        model_class, 
        test_set: pl.DataFrame,
        targets: List[str],
        features: List[str],
        save_path: PathType,
        name: str = ''
    ) -> pl.DataFrame:
    test_set = test_set.with_columns(
        pl.col(i).fill_nan(None).fill_null(0.0).alias(i) for i in features
    )
    X_ = test_set.select(features).to_numpy()
    for target in targets:
        model0 = model_class.load(os.path.join(save_path, '_'.join([name,target])))
        pred = model0.predict(X_)
        test_set = test_set.with_columns(
            pl.lit(pred).alias(f'pred_{target}'),
        )
    return test_set

