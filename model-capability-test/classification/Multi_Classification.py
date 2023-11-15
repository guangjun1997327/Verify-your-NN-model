import time, os, json
# from models.yourNN import yourNN_torch as yourNN

from sklearn.model_selection import train_test_split
from numpy.random import seed
seed(1)
torch.manual_seed(0)


def run_classification(datasets, size, pause_list, hidden_neurons, trials):
    for ds in datasets:
        # create dir
        t = time.strftime("%Y%m%d-%H%M%S")
        dir_name = f"./experiments/{ds}/exp_{t}"
        createFolder(dir_name)

        print("Dataset: ", ds)

        # store hyper params in dict
        config_and_hyperparams_dict = {"size":size, "hidden neurons":hidden_neurons, "trials":trials}

        # load data
        feature_range = [-1, 1]
        X, Y, _, _ = create_data(dataset=ds, data_size=size, feature_range=feature_range)
        input_dim = X.shape[1]
        output_dim = 10
        X = torch.from_numpy(X).float()


        # define layer and activation functions
        hidden_activation = ["sigmoid" for i in range(len(hidden_neurons))]
        output_activation = ["softmax"]
        activations = hidden_activation + output_activation
        layers = [input_dim] + hidden_neurons + [output_dim]

        config_and_hyperparams_dict["activations"] = activations

        acc_arr, nll_arr, time_arr = [np.zeros((trials, len(pause_list))) for _ in range(3)]

        for t in range(trials):
            print(f"Trial {t+1}")
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, shuffle=True, random_state=1)
            # initialize model
            model = Network(layers, activations, verbose=False)
            # model1 = Network2(layers, activations, verbose=False)
            # start training, store results and plot after each element in pause_list
            i_last = 0
            for n,i in enumerate(pause_list):
                train_time, acc, nll = train_and_evaluate(model, X_train[i_last:i], y_train[i_last:i], X_test, y_test, "multi_classification")
                if t == 0:
                    fig_path = os.path.join(dir_name, f"{i}_points.pdf")
                    # plot_binary_classification(model, X_train[:i], y_train[:i], i, fig_path, feature_range)

                acc_arr[t, n] = acc
                nll_arr[t, n] = nll
                time_arr[t, n] = np.sum(time_arr[t,max(0, n-1)]) + train_time
                i_last = i

        # store results
        eval = mean_std(acc_arr, nll_arr, time_arr)
        eval.append(np.array(pause_list))
        np.savetxt(os.path.join(dir_name, "results.csv"), np.stack(eval).T, delimiter=',', header="acc_mean,acc_std,nll_mean,nll_std,time_mean,time_std,n_points")

        # store hyper params
        with open(os.path.join(dir_name, "config_and_hyperparams.json"), "w") as f:
            json.dump(config_and_hyperparams_dict, f, indent=4, sort_keys=True)

        # print results from the last epoch over all trials
        pm = u"\u00B1" # +/-
        print(f"ACC: {'%.3f'%eval[0][-1]} {pm} {'%.3f'%eval[1][-1]}")
        print(f"NLL: {'%.3f'%eval[2][-1]} {pm} {'%.3f'%eval[3][-1]}")
        print(f"Total train time / s: {'%.3f'%eval[4][-1]} {pm} {'%.3f'%eval[5][-1]}")


if __name__ == "__main__":

    # hyper params
    hidden_neurons = [80, 80]
    datasets = [
        #"dataset_moon"
        #, "dataset_circles"
     "dataset_MNIST"
        ]
    size = 1500

    print("\nUse only one trail for testing codes. \nIn paper we use 10 trails to get statistical results.\n")

    trials = 5
    pause_list = [5, 50, 500, 1000, 1350]

    run_classification(datasets, size, pause_list, hidden_neurons, trials)
# if __name__ == "__main__":
# #
#     # hyper params
#     number_of_neurons = 40
#     while 1:
#         hidden_neurons = [number_of_neurons, number_of_neurons]
#
#         datasets = [

#          "dataset_MNIST"
#             ]
#         size = 1500
#
#         print("\nUse only one trail for testing codes. \nIn paper we use 10 trails to get statistical results.\n")
#
#         trials = 5
#         pause_list = [5, 50, 500, 1000, 1350]
#
#         run_classification(datasets, size, pause_list, hidden_neurons, trials)
#
#         number_of_neurons = number_of_neurons + 10
#         print(number_of_neurons)
