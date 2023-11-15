import json, sys
# sys.path.append('models/PBP_net/')
# import PBP_net
from TAGI.BayesNNactfunction import Bayesian_Network_torch as Bayesian_Network
# from meanfield import Bayesian_Network_torch as Bayesian_Network
# from tagireproduce import Bayesian_Network_torch as Bayesian_Network
# from models.BayesNN import Bayesian_Network_torch as Bayesian_Network
from helper_functions.aux_function import *
from helper_functions.plots import plot_synthetic_plot
from models.Dropout_yaringal import net
from numpy.random import seed
seed(1)
import tensorflow as tf
tf.random.set_seed(2)
tf.get_logger().setLevel('ERROR')


def run_regression(ds, size, epochs, trials, hidden):
    t = time.strftime("%Y%m%d-%H%M%S")
    dir_name = f"./experiments/{ds}/exp_{t}"
    createFolder(dir_name)

    print("Dataset: ", ds)
    config_and_hyperparams_dict = {"size":size, "hidden neurons":hidden, "epochs":epochs, "trials":trials}

    X_train, y_train, X_train_scaler, y_train_scaler = create_data(dataset=ds, feature_range=[-1,1], data_size=size)
    X_test, y_test, X_test_scaler, y_test_scaler = create_data(dataset=ds, feature_range=[-1,1], data_size=int(size*0.1))
    input_dim = X_train.shape[1]
    output_dim = 1

    hidden_activation = ["sigm" for i in range(len(hidden))]
    output_activation = ["linear"]
    activations = hidden_activation + output_activation
    layers = [input_dim] + hidden + [output_dim]

    config_and_hyperparams_dict["activations"] = activations

    rmse_kbnn = np.zeros(trials)
    nll_kbnn = np.zeros(trials)
    train_times_kbnn = np.zeros(trials)
    # rmse_pbp = np.zeros(trials)
    # nll_pbp = np.zeros(trials)
    # train_times_pbp = np.zeros(trials)
    # rmse_mc = np.zeros(trials)
    # train_times_mc = np.zeros(trials)

    for t in range(trials):
        print(f"Trial {t+1}")

        # print("Train MC Dropout")
        # dropout_bnn = net(n_input=1, n_output=1, n_hidden=hidden, n_obs=size, lr_rate=0.01, normalize=True, tau=1., dropout=0.1)
        # start = time.time()
        # dropout_bnn.train(X_train=X_train, y_train=y_train, n_epochs=1, batch_size=16)
        # end = time.time()
        # train_times_mc[t] = end - start
        # _, MC_rmse, MC_ll, _, _ = dropout_bnn.predict(X_test=X_test, y_test=y_test)
        # rmse_mc[t] = MC_rmse

        # print("Train PBP")
        # start = time.time()
        # pbp = PBP_net.PBP_net(X_train, y_train, hidden, normalize=True, n_epochs=epochs)
        # end = time.time()
        # train_times_pbp[t] = end - start
        # m, v, v_noise = pbp.predict(X_test)
        # rmse_pbp[t], nll_pbp[t] = rmse_regression(y_test, m, v+v_noise)

        bnn = Bayesian_Network(layers, activations, load_from_keras=False, input_scaler=X_train_scaler, output_scaler=y_train_scaler, verbose=False)

        X_train, y_train = torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()
        X_test, y_test = torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()

        print("Train KBNN")
        start = time.time()
        for e in range(epochs):
            bnn.train(X_train, y_train)
        end = time.time()
        train_times_kbnn[t] = end - start
        y_pred, y_cov = bnn.predict(X_test)
        rmse_kbnn[t], nll_kbnn[t] = rmse_regression(y_test, y_pred, y_cov)

    # rmse_mc_mean, rmse_mc_std = np.mean(rmse_mc), np.std(rmse_mc)
    # time_mc_mean, time_mc_std = np.mean(train_times_mc), np.std(train_times_mc)
    #
    # nll_pbp_mean, nll_pbp_std = np.mean(nll_pbp), np.std(nll_pbp)
    # rmse_pbp_mean, rmse_pbp_std = np.mean(rmse_pbp), np.std(rmse_pbp)
    # time_pbp_mean, time_pbp_std = np.mean(train_times_pbp), np.std(train_times_pbp)

    nll_kbnn_mean, nll_kbnn_std = np.mean(nll_kbnn), np.std(nll_kbnn)
    rmse_kbnn_mean, rmse_kbnn_std = np.mean(rmse_kbnn), np.std(rmse_kbnn)
    time_kbnn_mean, time_kbnn_std = np.mean(train_times_kbnn), np.std(train_times_kbnn)

    pm = u"\u00B1"

    # print(f"MC Dropout RMSE: {'%.3f'%rmse_mc_mean} {pm} {'%.3f'%rmse_mc_std}")
    # print(f"MC Dropout total train time / s: {'%.3f'%time_mc_mean} {pm} {'%.3f'%time_mc_std}")
    # print(f"PBP RMSE: {'%.3f' % rmse_pbp_mean} {pm} {'%.3f' % rmse_pbp_std}")
    # print(f"PBP NLL: {'%.3f' % nll_pbp_mean} {pm} {'%.3f' % nll_pbp_std}")
    # print(f"PBP total train time / s: {'%.3f'%time_pbp_mean} {pm} {'%.3f'%time_pbp_std}")
    print(f"KBNN RMSE: {'%.3f'%rmse_kbnn_mean} {pm} {'%.3f'%rmse_kbnn_std}")
    print(f"KBNN NLL: {'%.3f'%nll_kbnn_mean} {pm} {'%.3f'%nll_kbnn_std}")
    print(f"KBNN total train time / s: {'%.3f'%time_kbnn_mean} {pm} {'%.3f'%time_kbnn_std}")

    with open(os.path.join(dir_name, "config_and_hyperparams.json"), "w") as f:
        json.dump(config_and_hyperparams_dict, f, indent=4, sort_keys=True)

    fun = lambda u: u ** 3
    # plot_synthetic_plot(bnn,bnn_dropout,pbp  X_train, y_train, fun, dir_name, x_scaler=X_train_scaler, y_scaler=y_train_scaler)
    plot_synthetic_plot(bnn, X_train, y_train, fun, dir_name, x_scaler=X_train_scaler, y_scaler=y_train_scaler)
    # eval = np.stack([rmse_mc, rmse_kbnn, train_times_kbnn]).T
    eval = np.stack([ rmse_kbnn, train_times_kbnn]).T
    np.savetxt(os.path.join(dir_name, "results.csv"), eval, delimiter=',', header="rmse_mc,rmse_kbnn,train_time")


if __name__ == "__main__":

    ds = "dataset_synthetic_regression"
    size = 800
    epochs = 20
    hidden_neurons = [50]

    print("\nUse only one trail for testing codes. \nIn paper we use 10 trails to get statistical results.\n")
    trials = 1

    run_regression(ds, size, epochs, trials, hidden_neurons)








