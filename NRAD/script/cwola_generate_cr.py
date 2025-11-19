import numpy as np
import torch
import os
import sys
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

helpers_path = os.path.join('/ether/aegis/Research_HEP/NRAD/model_scripts')
sys.path.insert(0, os.path.abspath(helpers_path))
from Classifier import Classifier

seed = 2
n_context = 2
data_path = f"/home/aegis/ether/Research_HEP/NRAD/SemiVisJets/data/data_seed{seed}"
test_path = f"/home/aegis/ether/Research_HEP/NRAD/SemiVisJets/data/data_test"
samples_path = "/home/aegis/ether/Research_HEP/NRAD/SemiVisJets/samples"
eval_dir = "/home/aegis/ether/Research_HEP/NRAD/SemiVisJets/eval_cr"
mc_path = "/home/aegis/ether/Research_HEP/NRAD/SemiVisJets/data"

CUDA = torch.cuda.is_available()
device = torch.device("cuda" if CUDA else "cpu")
print("Device:", device)

config_path = "/home/aegis/ether/Research_HEP/NRAD/oldver/NRAD/non-resonant-AD/configs"
with open(f"{config_path}/bc_discrim.yml", 'r') as stream:
    params = yaml.safe_load(stream)


def run_eval(set_1, set_2, code, save_dir, classifier_params, device, w_1 = None, w_2 = None, classifier_runs = 20):

    if w_1 is None or w_1.size == 0:
        w_1 = np.ones(set_1.shape[0])
    if w_2 is None or w_2.size == 0:
        w_2 = np.ones(set_2.shape[0])

    # define test size — roughly 20% or limited to 10,000 samples
    test_size_ratio = min(10000 / set_1.shape[0], 0.2)

    # split each dataset independently
    trainset_1, testset_1, wtrain_1, wtest_1 = train_test_split(
        set_1, w_1, test_size=test_size_ratio, random_state=42
    )
    trainset_2, testset_2, wtrain_2, wtest_2 = train_test_split(
        set_2, w_2, test_size=test_size_ratio, random_state=42
    )

    # ---------- Build train/test sets ----------
    # Combine the two datasets
    input_x_train = np.concatenate([trainset_1, trainset_2], axis=0)
    input_y_train = np.concatenate([
        np.zeros(trainset_1.shape[0]),
        np.ones(trainset_2.shape[0])
    ]).reshape(-1, 1)
    input_w_train = np.concatenate([wtrain_1, wtrain_2], axis=0).reshape(-1, 1)

    input_x_test = np.concatenate([testset_1, testset_2], axis=0)
    input_y_test = np.concatenate([
        np.zeros(testset_1.shape[0]),
        np.ones(testset_2.shape[0])
    ]).reshape(-1, 1)
    # input_w_test = np.concatenate([wtest_1, wtest_2], axis=0).reshape(-1, 1)

    
    # ---------- Logging ----------
    print(f"\nWorking on {code}...")
    print("      X_train, y_train, w_train:", input_x_train.shape, input_y_train.shape, input_w_train.shape)
    print("      X_test, y_test:", input_x_test.shape, input_y_test.shape)
    
    aucs_list = []

    for i in range(int(classifier_runs)):
        
        print(f"Classifier run {i+1} of {classifier_runs}.")
        local_id = f"{code}_run{i}"
                
        # train classifier
        NN = Classifier(n_inputs=5, layers=classifier_params["layers"], learning_rate=classifier_params["learning_rate"], device=device, scale_data=False)
        print("Using device:", NN.device)
        NN.train(input_x_train, input_y_train, weights=input_w_train,  save_model=True, model_name = f"model_{local_id}" , n_epochs=classifier_params["n_epochs"], seed = i, outdir=save_dir)

        scores = NN.evaluation(input_x_test)
        auc = roc_auc_score(input_y_test, scores, sample_weight=np.concatenate([wtest_1, wtest_2]))
        if auc < 0.5:
            auc = 1.0 - auc  # symmetry adjustment
        aucs_list.append(auc)
        print(f"   AUC: {auc}")
    
    # ---------- Save results ----------
    os.makedirs(f"{save_dir}/auc_scores", exist_ok=True)
    np.savez(f"{save_dir}/auc_scores/auc_{code}.npz", auc_scores=np.array(aucs_list))

    print("\nMedian AUC, 16th percentile, 84th percentile:")
    print(np.median(aucs_list), [np.percentile(aucs_list, 16), np.percentile(aucs_list, 84)])
    print("Done.\n")

print("CWoLA Evaluation on Generate Samples")
for i in range(1, 6):
    generate_events = np.load(f"{samples_path}/generate_MC{seed:02d}_Data{i:02d}_CR_samples.npz", allow_pickle=True)
    context_weights = np.load(f"{samples_path}/context_weight_MC{seed:02d}_Data{i:02d}_CR_samples.npz", allow_pickle=True)
    data_events = np.load(f"/home/aegis/ether/Research_HEP/NRAD/SemiVisJets/data/data_test/data_events_chunk{6:02d}.npz", allow_pickle=True)
    set_1 = generate_events["generate_cr"]
    set_2 = data_events["data_events_cr"][:, n_context:]
    run_eval(set_1, set_2, code=f"generate_MC{seed:02d}_Data{i:02d}_cr", save_dir=eval_dir, classifier_params=params, device=device)