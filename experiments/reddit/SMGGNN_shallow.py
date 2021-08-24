import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
import torch
from dataReader.dataReader import DGLDatasetReader

from model.network import SMGNetwork
from conv.SMG_GC import SMG_mulithead
from impl.nodeClassificationImpl import modelImplementation_nodeClassificator
from utils.utils_method import printParOnFile, normalize

if __name__ == '__main__':

    test_type = 'SMG_mulithead'

    # sis setting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_list = range(5)
    n_epochs = 25
    test_epoch = 1
    early_stopping_patience = 25

    # test hyper par
    dropout_list = [0, 0.5]
    lr_list = [1, 0.5, 0.05, 0.005]
    weight_decay_list = [None]
    k_list = [2, 4, 6]
    tied = True
    criterion = torch.nn.CrossEntropyLoss()

    # Dataset
    dataset_name = 'reddit'
    self_loops = True

    graph, features, labels, n_classes, train_mask, test_mask, valid_mask = DGLDatasetReader(dataset_name, self_loops,
                                                                                             device)

    for k in k_list:
        for dropout in dropout_list:
            for weight_decay in weight_decay_list:
                for lr in lr_list:
                    for run in run_list:
                        test_name = "run_" + str(run) +'_'+ test_type
                        #Env
                        test_name = test_name +\
                                    "_data-" + dataset_name +\
                                    "_lr-" + str(lr) +\
                                    "_dropout-" + str(dropout) +\
                                    "_weight-decay-" + str(weight_decay) +\
                                    "_k-" + str(k) +\
                                    "_tide-" + str(tied)

                        test_type_folder = os.path.join("./test_log/", test_type)
                        if not os.path.exists(test_type_folder):
                            os.makedirs(test_type_folder)
                        training_log_dir = os.path.join(test_type_folder, test_name)
                        print(test_name)
                        if not os.path.exists(training_log_dir):
                            os.makedirs(training_log_dir)

                            printParOnFile(test_name=test_name, log_dir=training_log_dir, par_list={"dataset_name": dataset_name,
                                                                                                    "learning_rate": lr,
                                                                                                    "dropout": dropout,
                                                                                                    "weight_decay": weight_decay,
                                                                                                    "k": k,
                                                                                                    "tide": tied,
                                                                                                    "test_epoch": test_epoch,
                                                                                                    "self_loops": self_loops})





                            model = SMGNetwork(g=graph,
                                              in_feats=features.shape[1],
                                              n_classes=n_classes,
                                              dropout=dropout,
                                              k=k,
                                              convLayer=SMG_mulithead,
                                              device=device,
                                              norm=normalize,
                                              bias=True).to(device)

                            model_impl = modelImplementation_nodeClassificator(model=model,
                                                                               criterion=criterion,
                                                                               device=device)
                            model_impl.set_optimizer_reddit(lr=lr)

                            model_impl.train_test_model_reddit(input_features=features,
                                                        labels=labels,
                                                        train_mask=train_mask,
                                                        test_mask=test_mask,
                                                        valid_mask=valid_mask,
                                                        n_epochs=n_epochs,
                                                        test_epoch=test_epoch,
                                                        test_name=test_name,
                                                        log_path=training_log_dir,
                                                        patience=early_stopping_patience)

                            if str(device) == 'cuda':
                                del model
                                del model_impl
                                torch.cuda.empty_cache()
                        else:
                            print("test has been already execute")
                            torch.cuda.empty_cache()
