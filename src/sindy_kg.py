import sys, os

import torch
from copy import deepcopy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.src_rlsindy.theorist import rl_sindy_theorist
from src.src_rlsindy.utils.convert_dataset import convert_dataset
from src.src_rlsindy.resources.rnn_utils import DatasetRNN
from src.src_rlsindy.resources.model_evaluation import log_likelihood, bayesian_information_criterion, akaike_information_criterion


def translate_sindy_to_kg(features, file_to_save):
    pass

def sindy_pipeline(path_to_dataset: str, device: torch.device = None, epochs: int = 128, ratio_train_test: float = 0.8, verbose: bool = False, participant_embedding=True, file_to_save: str = None):
    
    # check that ratio_train_test is in valid range [0, 1]
    if ratio_train_test > 1 or ratio_train_test < 0:
        raise ValueError(f'Argument ratio_train_test ({ratio_train_test}) is outside of range [0, 1]')
    
    if device is None or not isinstance(device, torch.device):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = convert_dataset(path_to_dataset)[0]
    index_train = int(dataset.xs.shape[1] * ratio_train_test)
    xs_train, ys_train = dataset.xs[:, :index_train], dataset.ys[:, :index_train]
    xs_test, ys_test = dataset.xs[:, index_train:], dataset.ys[:, index_train:]
    dataset_train = DatasetRNN(xs_train, ys_train)
    dataset_test = DatasetRNN(xs_test, ys_test)
    
    rl_sindy = rl_sindy_theorist(
        n_participants=dataset_train.xs.shape[0] if participant_embedding else 0,
        device=device,
        verbose=verbose,
        epochs=epochs,
        )

    rl_sindy.fit(dataset_train.xs.numpy(), dataset_train.ys.numpy())
    
    prediction_rnn, prediction_sindy = rl_sindy.predict(dataset_test.xs.numpy())
    
    n_parameters_rnn = sum(p.numel() for p in rl_sindy.model_rnn.parameters() if p.requires_grad)
    n_parameters_sindy = [rl_sindy.sindy_agents[id]._count_sindy_parameters(without_self=True) for id in rl_sindy.sindy_agents]
    
    ll_rnn = log_likelihood(dataset_test.ys.numpy(), prediction_rnn)
    ll_sindy = log_likelihood(dataset_test.ys.numpy(), prediction_sindy)
    
    bic_rnn = bayesian_information_criterion(data=dataset_test.ys.numpy(), probs=prediction_rnn, ll=ll_rnn, n_parameters=n_parameters_rnn)
    aic_rnn = akaike_information_criterion(data=dataset_test.ys.numpy(), probs=prediction_rnn, ll=ll_rnn, n_parameters=n_parameters_rnn)

    bic_sindy = 0
    aic_sindy = 0
    for id in rl_sindy.sindy_agents:
        bic_sindy += bayesian_information_criterion(data=dataset_test.ys[id].unsqueeze(0).numpy(), probs=prediction_sindy[id][None, :, :], ll=ll_sindy, n_parameters=n_parameters_sindy[id])
        aic_sindy += akaike_information_criterion(data=dataset_test.ys[id].unsqueeze(0).numpy(), probs=prediction_sindy[id][None, :, :], ll=ll_sindy, n_parameters=n_parameters_sindy[id])
    
    features = {}
    for id in rl_sindy.sindy_agents:
        rl_sindy.sindy_agents[id].new_sess()
        
        sindy_models = rl_sindy.sindy_agents[id]._models
        
        # save a dictionary of trained features per model
        features_id = {
            'beta_v': (tuple(['beta_v']), tuple([rl_sindy.sindy_agents[id]._beta_reward]), [str(rl_sindy.sindy_agents[id]._beta_reward) + ' 1']),
            'beta_c': (tuple(['beta_c']), tuple([rl_sindy.sindy_agents[id]._beta_choice]), [str(rl_sindy.sindy_agents[id]._beta_choice) + ' 1']),
            }
        
        for m in sindy_models:
            features_m = sindy_models[m].get_feature_names()
            coeffs_m = [c for c in sindy_models[m].coefficients()[0]]
            # sort out every dummy control parameter (i.e. any candidate term which contains 'u')
            index_u = []
            for i, f in enumerate(features_m):
                if 'u' in f:
                    index_u.append(i)
            features_m = [item for idx, item in enumerate(features_m) if idx not in index_u]
            coeffs_m = [item for idx, item in enumerate(coeffs_m) if idx not in index_u]
            features_id[m] = (tuple(features_m), tuple(coeffs_m), sindy_models[m].equations())
        
        features[id] = deepcopy(features_id)
        
        translate_sindy_to_kg(features, file_to_save)
    
    return (-ll_rnn, -ll_sindy), (bic_rnn, bic_sindy), (aic_rnn, aic_sindy)


if __name__=='__main__':
    path_to_data = 'src/src_rlsindy/data/sugawara2021_143_processed_short.csv'
    
    metrics = sindy_pipeline(path_to_data, epochs=16, verbose=True, participant_embedding=True)
    
    print(f'NLL RNN: {metrics[0][0]}')
    print(f'NLL SINDy: {metrics[0][1]}')
    print(f'BIC RNN: {metrics[1][0]}')
    print(f'BIC SINDy: {metrics[1][1]}')
    print(f'AIC RNN: {metrics[2][0]}')
    print(f'AIC SINDy: {metrics[2][1]}')