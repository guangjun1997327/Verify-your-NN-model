import json, sys
from model.yourNN import NN_torch as NN
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

        nn = Network(layers, activations, load_from_keras=False, input_scaler=X_train_scaler, output_scaler=y_train_scaler, verbose=False)

        X_train, y_train = torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()
        X_test, y_test = torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()

        print("Train NN")
        start = time.time()
        for e in range(epochs):
            bnn.train(X_train, y_train)
        end = time.time()
        train_times[t] = end - start
        y_pred, y_cov = bnn.predict(X_test)
        rmse[t], nll[t] = rmse_regression(y_test, y_pred, y_cov)

  
    pm = u"\u00B1"


    print(f" RMSE: {'%.3f'%rmse_mean} {pm} {'%.3f'%rmse_std}")
    print(f" NLL: {'%.3f'%nll_mean} {pm} {'%.3f'%nll_std}")
    print(f" total train time / s: {'%.3f'%timen_mean} {pm} {'%.3f'%time_std}")

    with open(os.path.join(dir_name, "config_and_hyperparams.json"), "w") as f:
        json.dump(config_and_hyperparams_dict, f, indent=4, sort_keys=True)

    fun = lambda u: u ** 3
    # plot_synthetic_plot(bnn,bnn_dropout,pbp  X_train, y_train, fun, dir_name, x_scaler=X_train_scaler, y_scaler=y_train_scaler)
    plot_synthetic_plot(bnn, X_train, y_train, fun, dir_name, x_scaler=X_train_scaler, y_scaler=y_train_scaler)
 
    eval = np.stack([ rmse, train_times]).T
    np.savetxt(os.path.join(dir_name, "results.csv"), eval, delimiter=',', header="rmse,train_time")


if __name__ == "__main__":

    ds = "dataset_synthetic_regression"
    size = 800
    epochs = 20
    hidden_neurons = [50]

    print("\nUse only one trail for testing codes. \nIn paper we use 10 trails to get statistical results.\n")
    trials = 1

    run_regression(ds, size, epochs, trials, hidden_neurons)








