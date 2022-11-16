from dataset import UciDataset
from metrics import *
from model import *
from utils import *
import constants as keys
import numpy as np
import json
import torch


# Function used for testing a model with and without dropout
def test(dataset, dr_load, lr, bs, mc_size=1000, dr_test=None, temporary=False, light_mode=False):
    # Get cpu or gpu device for training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('----- Testing with Dropout Rate {} -----'.format(dr_test))

    # Choosing the dataset and methodology
    if dr_test is None:
        dr_test = dr_load

    # Obtaining the path used for loading the model
    path_to_load = obtain_training_folder(dataset, dr_load, lr, bs) if not temporary else keys.TEMP_EXP_FOLDER
    path_to_load = os.path.join(path_to_load, 'cross_validation')

    # Obtaining the path used for saving the experimental results
    path_to_save = os.path.join(path_to_load, 'test_dr{}'.format(str(dr_test).split('.')[1]))
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)

    # Choosing the configuration used for the selected dataset according to the Hernandez-Lobato experiment
    hidden_neurons = 100 if dataset in keys.LARGE_DATASETS else 50
    iterations = 5 if dataset in keys.LARGE_DATASETS else 20

    # Declaring the lists for saving nll, aucc, and mse
    nll_test_unscaled_list, nll_test_scaled_list, nll_test_perfect_list = [], [], []
    nll_training_unscaled_list, nll_training_scaled_list, nll_training_perfect_list = [], [], []
    nll_validation_perfect_list, nll_validation_scaled_list, nll_validation_unscaled_list = [], [], []
    aucc_test_unscaled_list, aucc_training_unscaled_list, aucc_validation_unscaled_list = [], [], []
    aucc_test_scaled_list, aucc_training_scaled_list, aucc_validation_scaled_list = [], [], []
    aucc_test_relaxed_list, aucc_training_relaxed_list, aucc_validation_relaxed_list = [], [], []
    mse_training_list, mse_validation_list, mse_test_list = [], [], []

    # Iterating over the multiple splits
    for k in range(iterations):
        # Obtaining the dataset and loading the trained model
        my_dataset = UciDataset(dataset, mode='training', light_mode=light_mode, split=k)
        my_model = SimpleMLP(input_units=my_dataset.X_train.shape[1], hidden_units=hidden_neurons,
                             dropout_rate=dr_test).to(device).double()
        weights_file = 'weights_{}.pt'.format(k)
        my_model.load_state_dict(torch.load(os.path.join(path_to_load, weights_file)))

        # Setting training mode for enabling monte carlo stochasticity
        my_model.train()

        # Getting predictions, uncertainty and quadratic error
        gt = np.squeeze(my_dataset.y_data)
        pred, unc = my_model.get_uncertainty(my_dataset.X_data, device, mc_size, normalize=keys.NORMALIZE_DICT[dataset])
        gt = gt * keys.NORMALIZE_DICT[dataset][1] + keys.NORMALIZE_DICT[dataset][0]
        square_errors = (pred - gt) ** 2

        # Getting the mean square error
        mse_training_list.append(np.mean(square_errors[my_dataset.ids_train]))
        mse_validation_list.append(np.mean(square_errors[my_dataset.ids_validation]))
        mse_test_list.append(np.mean(square_errors[my_dataset.ids_test]))

        # Computing the optimal scale value using the validation set and getting the scaled uncertainty measure
        opt = optimal_scaler(pred[my_dataset.ids_validation],
                             gt[my_dataset.ids_validation],
                             unc[my_dataset.ids_validation])
        unc_scaled = unc * opt

        # Computing the relaxed scale value using the validation set and getting the relaxed uncertainty measure
        relaxed_opt = optimal_calibration(pred[my_dataset.ids_validation],
                                          gt[my_dataset.ids_validation],
                                          unc_scaled[my_dataset.ids_validation])
        unc_relaxed = unc_scaled * relaxed_opt

        # Computing the area under the calibration curve for training, validation and test sets
        aucc_test_unscaled_list.append(area_under_calibration_curve(pred[my_dataset.ids_test],
                                                                    unc[my_dataset.ids_test],
                                                                    gt[my_dataset.ids_test]))

        aucc_training_unscaled_list.append(area_under_calibration_curve(pred[my_dataset.ids_train],
                                                                        unc[my_dataset.ids_train],
                                                                        gt[my_dataset.ids_train]))

        aucc_validation_unscaled_list.append(area_under_calibration_curve(pred[my_dataset.ids_validation],
                                                                          unc[my_dataset.ids_validation],
                                                                          gt[my_dataset.ids_validation]))

        aucc_test_scaled_list.append(area_under_calibration_curve(pred[my_dataset.ids_test],
                                                                  (unc_scaled[my_dataset.ids_iid])[my_dataset.ids_test],
                                                                  gt[my_dataset.ids_test]))

        aucc_training_scaled_list.append(area_under_calibration_curve(pred[my_dataset.ids_train],
                                                                      (unc_scaled[my_dataset.ids_iid])[my_dataset.ids_train],
                                                                      gt[my_dataset.ids_train]))

        aucc_validation_scaled_list.append(area_under_calibration_curve(pred[my_dataset.ids_validation],
                                                                        (unc_scaled[my_dataset.ids_iid])[my_dataset.ids_validation],
                                                                        gt[my_dataset.ids_validation]))

        # Computing the AUCC on the relaxed measure
        aucc_test_relaxed_list.append(area_under_calibration_curve(pred[my_dataset.ids_test],
                                                                  (unc_relaxed[my_dataset.ids_iid])[my_dataset.ids_test],
                                                                  gt[my_dataset.ids_test]))

        aucc_training_relaxed_list.append(area_under_calibration_curve(pred[my_dataset.ids_train],
                                                                      (unc_relaxed[my_dataset.ids_iid])[my_dataset.ids_train],
                                                                      gt[my_dataset.ids_train]))

        aucc_validation_relaxed_list.append(area_under_calibration_curve(pred[my_dataset.ids_validation],
                                                                        (unc_relaxed[my_dataset.ids_iid])[my_dataset.ids_validation],
                                                                        gt[my_dataset.ids_validation]))

        # --- Computing the Negative Log Likelihood for test set ---
        # NLL on the test set using unscaled measure
        nll_test_unscaled = neg_log_likelihood(pred[my_dataset.ids_test],
                                               gt[my_dataset.ids_test],
                                               unc[my_dataset.ids_test])
        nll_test_unscaled_list.append(nll_test_unscaled)

        # NLL on the test set using scaled measure
        nll_test_scaled = neg_log_likelihood(pred[my_dataset.ids_test],
                                             gt[my_dataset.ids_test],
                                             (unc_scaled[my_dataset.ids_iid])[my_dataset.ids_test])
        nll_test_scaled_list.append(nll_test_scaled)

        # NLL on the test set using ideal measure
        nll_test_perfect = neg_log_likelihood(pred[my_dataset.ids_test],
                                              gt[my_dataset.ids_test],
                                              square_errors[my_dataset.ids_test])
        nll_test_perfect_list.append(nll_test_perfect)

        # --- Computing the Negative Log Likelihood for training set ---
        # NLL on the training set using unscaled measure
        nll_training_unscaled = neg_log_likelihood(pred[my_dataset.ids_train],
                                                   gt[my_dataset.ids_train],
                                                   unc[my_dataset.ids_train])
        nll_training_unscaled_list.append(nll_training_unscaled)

        # NLL on the training set using scaled measure
        nll_training_scaled = neg_log_likelihood(pred[my_dataset.ids_train],
                                                 gt[my_dataset.ids_train],
                                                 (unc_scaled[my_dataset.ids_iid])[my_dataset.ids_train])
        nll_training_scaled_list.append(nll_training_scaled)

        # NLL on the training set using ideal measure
        nll_training_perfect = neg_log_likelihood(pred[my_dataset.ids_train],
                                                  gt[my_dataset.ids_train],
                                                  square_errors[my_dataset.ids_train])
        nll_training_perfect_list.append(nll_training_perfect)

        # --- Computing the Negative Log Likelihood for validation set ---
        # NLL on the validation set using unscaled measure
        nll_validation_unscaled = neg_log_likelihood(pred[my_dataset.ids_validation],
                                                     gt[my_dataset.ids_validation],
                                                     unc[my_dataset.ids_validation])
        nll_validation_unscaled_list.append(nll_validation_unscaled)

        # NLL on the validation set using scaled measure
        nll_validation_scaled = neg_log_likelihood(pred[my_dataset.ids_validation],
                                                   gt[my_dataset.ids_validation],
                                                   (unc_scaled[my_dataset.ids_iid])[my_dataset.ids_validation])
        nll_validation_scaled_list.append(nll_validation_scaled)

        # NLL on the validation set using ideal measure
        nll_validation_perfect = neg_log_likelihood(pred[my_dataset.ids_validation],
                                                    gt[my_dataset.ids_validation],
                                                    square_errors[my_dataset.ids_validation])
        nll_validation_perfect_list.append(nll_validation_perfect)


    # Obtaining mean NLL on test
    nll_test_unscaled, std_nll_test_unscaled = np.mean(np.array(nll_test_unscaled_list)), np.std(np.array(nll_test_unscaled_list))
    nll_test_scaled, std_nll_test_scaled = np.mean(np.array(nll_test_scaled_list)), np.std(np.array(nll_test_scaled_list))
    nll_test_perfect, std_nll_test_perfect = np.mean(np.array(nll_test_perfect_list)), np.std(np.array(nll_test_perfect_list))

    # Obtaining mean NLL on train
    nll_training_unscaled = np.mean(np.array(nll_training_unscaled_list))
    nll_training_scaled = np.mean(np.array(nll_training_scaled_list))
    nll_training_perfect = np.mean(np.array(nll_training_perfect_list))

    # Obtaining mean NLL on validation
    nll_validation_perfect = np.mean(np.array(nll_validation_perfect_list))
    nll_validation_scaled = np.mean(np.array(nll_validation_scaled_list))
    nll_validation_unscaled = np.mean(np.array(nll_validation_unscaled_list))

    # Obtaining mean and standard deviation of test AUCC
    aucc_test_unscaled = np.mean(np.array(aucc_test_unscaled_list))
    std_aucc_test_unscaled = np.std(np.array(aucc_test_unscaled_list))
    aucc_test_scaled = np.mean(np.array(aucc_test_scaled_list))
    std_aucc_test_scaled = np.std(np.array(aucc_test_scaled_list))
    aucc_test_relaxed = np.mean(np.array(aucc_test_relaxed_list))
    std_aucc_test_relaxed = np.std(np.array(aucc_test_relaxed_list))

    # Obtaining the mean of training and validation AUCC
    aucc_training_unscaled = np.mean(np.array(aucc_training_unscaled_list))
    aucc_validation_unscaled = np.mean(np.array(aucc_validation_unscaled_list))
    aucc_training_scaled = np.mean(np.array(aucc_training_scaled_list))
    aucc_validation_scaled = np.mean(np.array(aucc_validation_scaled_list))
    aucc_training_relaxed = np.mean(np.array(aucc_training_relaxed_list))
    aucc_validation_relaxed = np.mean(np.array(aucc_validation_relaxed_list))

    # Obtaining training, validation and test MSEs
    mse_training = np.mean(np.array(mse_training_list))
    mse_validation = np.mean(np.array(mse_validation_list))
    mse_test, std_mse_test = np.mean(np.array(mse_test_list)), np.std(np.array(mse_test_list))

    # Obtaining training, validation and test RMSEs
    mse_test_array = np.array(mse_test_list)
    rmses_test_array = np.sqrt(mse_test_array)
    rmse_test, std_rmse_test = np.mean(rmses_test_array), np.std(rmses_test_array)

    config_dict = {
        'dropout_rate_load': dr_load,
        'dropout_rate_test': dr_test,
        'learning_rate': lr,
        'batch_size': bs
    }

    mses_dict = {
        'MSE_training': mse_training,
        'MSE_validation': mse_validation,
        'MSE_test': mse_test,
        'RMSE_test': rmse_test
    }

    aucc_dict = {
        'AUCC_training_unscaled': aucc_training_unscaled,
        'AUCC_training_scaled': aucc_training_scaled,
        'AUCC_training_relaxed': aucc_training_relaxed,
        'AUCC_validation_unscaled': aucc_validation_unscaled,
        'AUCC_validation_scaled': aucc_validation_scaled,
        'AUCC_validation_relaxed': aucc_validation_relaxed,
        'AUCC_test_unscaled': aucc_test_unscaled,
        'AUCC_test_scaled': aucc_test_scaled,
        'AUCC_test_relaxed': aucc_test_relaxed
    }

    nll_dict = {
        'nll_training_unscaled': nll_training_unscaled,
        'nll_training_scaled': nll_training_scaled,
        'nll_training_perfect': nll_training_perfect,
        'nll_validation_unscaled': nll_validation_unscaled,
        'nll_validation_scaled': nll_validation_scaled,
        'nll_validation_perfect': nll_validation_perfect,
        'nll_test_unscaled': nll_test_unscaled,
        'nll_test_scaled': nll_test_scaled,
        'nll_test_perfect': nll_test_perfect
    }

    stds = {
        'std_nll_test_unscaled': std_nll_test_unscaled,
        'std_nll_test_scaled': std_nll_test_scaled,
        'std_nll_test_perfect': std_nll_test_perfect,
        'std_aucc_test_unscaled': std_aucc_test_unscaled,
        'std_aucc_test_scaled': std_aucc_test_scaled,
        'std_mse_test': std_mse_test,
        'std_rmse_test': std_rmse_test
    }

    results_dict = {
        'config': config_dict,
        'mses': mses_dict,
        'aucc': aucc_dict,
        'nll': nll_dict,
        'stds': stds
    }

    # Saving the files
    # f = open(os.path.join(path_to_save, 'results.pkl'), "wb")
    # pickle.dump(results_dict, f)
    # f.close()
    f = open(os.path.join(path_to_save, 'results.json'), "w")
    json.dump(results_dict, f)
    f.close()
