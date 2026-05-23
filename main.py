import argparse
import os
import scipy.sparse as sp

from pandas.plotting import table
from prettytable import PrettyTable

import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd

from progkd import ProGKD
from progkd_graph import ProGKDGraph
from utils import *
from tqdm import tqdm

cuda = torch.cuda.is_available()


def parse_args():
    parser = argparse.ArgumentParser(description='Train ProGKD with cross-validation and save model to file')
    parser.add_argument('-e', '--epochs', help='maximum number of epochs (default: 1000)',
                        dest='epochs',
                        default=3000,
                        type=int
                        )
    parser.add_argument('-p', '--patience', help='patience (default: 20)',
                        dest='patience',
                        default=3000,
                        type=int
                        )
    parser.add_argument('-dp', '--dropout', help='the dropout rate (default: 0.25)',
                        dest='dp',
                        default=0.25,
                        type=float
                        )
    parser.add_argument('-lr', '--learningrate', help='the learning rate (default: 0.001)',
                        dest='lr',
                        default=0.00002,
                        type=float
                        )
    parser.add_argument('-wd', '--weightdecay', help='the weight decay (default: 0.0005)',
                        dest='wd',
                        default=0.0005,
                        type=float
                        )
    parser.add_argument('-hs1', '--hiddensize1', help='the hidden size of first layer',
                        dest='hs1',
                        default=250,
                        type=int
                        )
    parser.add_argument('-hs2', '--hiddensize2', help='the hidden size of second  layer',
                        dest='hs2',
                        default=100,
                        type=int
                        )
    parser.add_argument('-seed', '--seed', help='the random seed (default: 42)',
                        dest='seed',
                        default=42,
                        type=int
                        )
    args = parser.parse_args()
    return args


def main(args):
    n_student = 1
    alpha =1
    modelname = f'progkd'
    tr = 0.6
    lr_rate = f"{args['lr']}"

    table1 = PrettyTable()
    table1.field_names = ['model', 'best', 'acc', 'auroc', 'aupr', 'f1-score', 'epoch']
    table1.add_row(['GLIMS', '    ....    ', 0.858, 0.904, 0.768, 0.667, 0])
    table1.add_row([modelname, '', 0, 0, 0, 0, 0])
    table1.add_row([modelname, '', 0, 0, 0, 0, 0])
    table1.add_row([modelname, '', 0, 0, 0, 0, 0])
    table1.add_row([modelname, '', 0, 0, 0, 0, 0])
    table2 = table1.get_string()
    table3 = table2.split("\n")

    d_input = 1024
    d_feedforward = 4096
    nhead = 4
    K = 1



    seed_torch(args['seed'])
    file_save_path = os.path.join(f'./Output/{modelname}')
    make_dir(file_save_path)

    # load data
    progkd_input = ProGKDGraph()
    omicsfeature, emb_feature, final_gene_node = progkd_input.get_node_omic_embedding_feature()
    ppi_network = progkd_input.generate_graph()


    emb_feature = emb_feature.squeeze(1).cpu()
    print("==========================================================")
    print('Network INFO')
    name_of_network = ['PPI']
    graphlist = []
    for i, network in enumerate([ppi_network]):
        featured_graph = progkd_input.load_featured_graph(network, emb_feature)
        featured_graph1 = progkd_input.load_featured_graph(network, emb_feature)
        # featured_graph = progkd_input.load_featured_graph(network, omicsfeature)
        # featured_graph1 = progkd_input.load_featured_graph(network, omicsfeature)
        print(f'The {name_of_network[i]} graph: {featured_graph}')
        graphlist.append(featured_graph)
        graphlist.append(featured_graph1)

    emb_feature = emb_feature.cuda()

    n_fdim = graphlist[0].x.shape[1]
    graphlist_adj = [graph.cuda() for graph in graphlist]
    k_sets, idx_list, label_list = progkd_input.ten_fold_five_crs_validation_psudo_label(file_save_path, K=K)
    print("==========================================================")

    def loss_fn_kd(outputs, labels, teacher_outputs, T, alpha):
        loss = F.kl_div(F.log_softmax(outputs / T, dim=1), F.softmax(teacher_outputs / T, dim=1), reduction='batchmean') * (alpha) + F.binary_cross_entropy_with_logits(outputs, labels, pos_weight=torch.Tensor([2.7]).cuda()) * (1 - alpha)
        return loss
    
    def num_params(model):
        nums = sum(p.numel() for p in model.parameters()) / 1e6
        return nums

    def ron(x, tr):
        rounded_array = np.where(x >= tr, np.ceil(x), np.floor(x))
        return rounded_array

    def train_teacher(model, optimizer, mask, label):
        model.train()
        optimizer.zero_grad()
        output = model(graphlist_adj[1], mask=mask)
        loss = F.binary_cross_entropy_with_logits(output[mask], label, pos_weight=torch.Tensor([2.7]).cuda())

        acc = metrics.accuracy_score(label.cpu(), ron(torch.sigmoid(output[mask]).cpu().detach().numpy(), tr))
        loss.backward()
        optimizer.step()

        return output, loss.item(), acc


    def train_student(model, model_teacher, optimizer, mask, label, alpha, mask_real):
        model.train()
        optimizer.zero_grad()
        output = model(graphlist_adj[0], mask=mask)
        with torch.no_grad():
            teacer_output = model_teacher(graphlist_adj[1], mask=mask)

        loss = loss_fn_kd(output[mask], label, teacer_output[mask], T=2, alpha=alpha)

        acc = metrics.accuracy_score(label.cpu(), ron(torch.sigmoid(output[mask]).cpu().detach().numpy(), tr))
        loss.backward()
        optimizer.step()

        return output, loss.item(), acc

    @torch.no_grad()
    def test(model, mask, label,student):
        model.eval()
        if student:
            output = model(graphlist_adj[0], mask=mask)
        else:
            output = model(graphlist_adj[1], mask=mask)
        loss = F.binary_cross_entropy_with_logits(output[mask], label, pos_weight=torch.Tensor([2.7]).cuda())

        acc = metrics.accuracy_score(label.cpu(), ron(torch.sigmoid(output[mask]).cpu().detach().numpy(), tr))
        pred = torch.sigmoid(output[mask]).cpu().detach().numpy()
        auroc = metrics.roc_auc_score(label.to('cpu'), pred)
        pr, rec, _ = metrics.precision_recall_curve(label.to('cpu'), pred)
        aupr = metrics.auc(rec, pr)
        F1Score = metrics.f1_score(label.to('cpu'), ron(torch.sigmoid(output[mask]).cpu().detach().numpy(), tr))

        return output, pred, loss.item(), acc, auroc, aupr, F1Score


    ####### K-Fold ########
    torch.manual_seed(42)
    AUROC = np.zeros(shape=(K, 5))
    AUPR = np.zeros(shape=(K, 5))
    ACC = np.zeros(shape=(K, 5))
    F1Score = np.zeros(shape=(K, 5))

    best_AUROC = 0
    best_AUPR = 0
    best_ACC = 0
    best_F1Score = 0

    pred_all = []
    label_all = []

    for j in range(len(k_sets)):
        print(j)
        for cv_run in range(5):
            print(f'fold:{cv_run}')
            print('train teacher')
            train_mask, val_mask, train_label, val_label, mask_truth_idx, mask_truth_label = [p.cuda() for p in k_sets[j][cv_run] if type(p) == torch.Tensor]

            teacher_model = ProGKD(nfeat=graphlist[1].x.shape[1], hidden_size1=args['hs1'], hidden_size2=args['hs2'], dropout=args['dp'], d_input=d_input, nhead=nhead, d_feedforward=d_feedforward, student=False)
            teacher_model.cuda()
            optimizer = optim.Adam(teacher_model.parameters(), lr=0.00002, weight_decay=args['wd'])

            early_stopping = EarlyStopping(patience=200, verbose=True)

            with tqdm(range(1, args['epochs'] + 1), unit='epoch') as tepoch:
                for epoch in tepoch:
                    tepoch.set_description(f'train epoch for teacher : {epoch}')

                    # _, _, _ = train_teacher(teacher_model, optimizer, train_mask, train_label)
                    _, _, _ = train_teacher(teacher_model, optimizer, mask_truth_idx, mask_truth_label)
                    _, _, loss_val1, ACC1, AUROC1, AUPR1, F1Score1 = test(teacher_model, val_mask, val_label, student=False)

                    tepoch.set_postfix(loss_val=loss_val1, acc=ACC1, auroc=AUROC1, aupr=AUPR1, f1score=F1Score1)

                    early_stopping(loss_val1, teacher_model)
                    if early_stopping.early_stop:
                        print(f"Early stopping at the epoch {epoch}")
                        break

            print('train student')
            for s in range(n_student):

                student_model = ProGKD(nfeat=n_fdim, hidden_size1=args['hs1'], hidden_size2=args['hs2'], dropout=args['dp'], d_input=d_input, nhead=nhead, d_feedforward=d_feedforward, student=True)
                student_model.cuda()
                optimizer = optim.Adam(student_model.parameters(), lr=args['lr'], weight_decay=args['wd'])

                early_stopping = EarlyStopping(patience=args['patience'], verbose=True)

                with tqdm(range(1, args['epochs'] + 1), unit='epoch') as tepoch:
                    for epoch in tepoch:
                        tepoch.set_description(f'train epoch for student{s+1} : {epoch}')
                        if epoch%500==0:
                            if alpha>0:
                                alpha = alpha - 0.1

                        # _, _, _ = train_student(student_model, teacher_model, optimizer, train_mask, train_label, alpha)
                        _, _, _ = train_student(student_model, teacher_model, optimizer, train_mask, train_label, alpha, mask_truth_idx)
                        _, _, loss_val1, ACC1, AUROC1, AUPR1, F1Score1 = test(student_model, val_mask, val_label, student=True)

                        tepoch.set_postfix(loss_val=loss_val1, acc=ACC1, auroc=AUROC1, aupr=AUPR1, f1score=F1Score1)


                        if (s+1)==n_student:
                            if ACC1 > best_ACC:
                                best_ACC = ACC1
                                table3[4] = f"| {modelname} |   best_acc   | {ACC1:.3f} | {AUROC1:.3f} | {AUPR1:.3f} | {F1Score1:.3f} | {epoch} |"

                            if AUROC1 > best_AUROC:
                                best_AUROC = AUROC1
                                table3[5] = f"| {modelname} |  best_auroc  | {ACC1:.3f} | {AUROC1:.3f} | {AUPR1:.3f} | {F1Score1:.3f} | {epoch} |"

                                torch.save(student_model,f'./Output/{modelname}/best_model_weights_for_best_AUROC_{modelname}_{lr_rate}.pth')

                            if AUPR1 > best_AUPR:
                                best_AUPR = AUPR1
                                table3[6] = f"| {modelname} |  best_aupr   | {ACC1:.3f} | {AUROC1:.3f} | {AUPR1:.3f} | {F1Score1:.3f} | {epoch} |"

                            if F1Score1 > best_F1Score:
                                best_F1Score = F1Score1
                                table3[7] = f"| {modelname} | best_f1score | {ACC1:.3f} | {AUROC1:.3f} | {AUPR1:.3f} | {F1Score1:.3f} | {epoch} |"



                        early_stopping(loss_val1, teacher_model)
                        if early_stopping.early_stop:
                            print(f"Early stopping at the epoch {epoch}")
                            break



            _, pred, _, ACC[j][cv_run], AUROC[j][cv_run], AUPR[j][cv_run], F1Score[j][cv_run] = test(student_model, val_mask, val_label, student=True)

            pred_all.append(pred)
            label_all.append(val_label.to('cpu'))

    table4 = '\n'.join(table3)
    with open(f'./Output/{modelname}/best_table_{modelname}_{lr_rate}.txt', "w") as f:
        f.write(table4)



    print(f'model: {modelname}, lr: {lr_rate}')
    print('Mean AUROC', AUROC.mean())
    print('Var AUROC', AUROC.var())
    print('Mean AUPR', AUPR.mean())
    print('Var AUPR', AUPR.var())
    print('Mean ACC', ACC.mean())
    print('Var ACC', ACC.var())
    print('Mean F1Score', F1Score.mean())
    print('Var F1Score', F1Score.var())

    with open(f'./Output/{modelname}/result_{modelname}_{lr_rate}.txt', 'w') as f:
        f.write(f'model: {modelname}, lr: {lr_rate}\n')
        f.write(f'Mean AUROC: {AUROC.mean():.3f}\n')
        f.write(f'Var AUROC: {AUROC.var():.3f}\n')
        f.write(f'Mean AUPR: {AUPR.mean():.3f}\n')
        f.write(f'Var AUPR: {AUPR.var():.3f}\n')
        f.write(f'Mean ACC: {ACC.mean():.3f}\n')
        f.write(f'Var ACC: {ACC.var():.3f}\n')
        f.write(f'mean F1Score: {F1Score.mean():.3f}\n')
        f.write(f'Var F1Score: {F1Score.var():.3f}\n')



    torch.save(pred_all, os.path.join(file_save_path, 'pred_all.pkl'))
    torch.save(label_all, os.path.join(file_save_path, 'label_all.pkl'))



    # Use all label to train a final model
    all_mask = torch.LongTensor(idx_list)
    all_label = torch.FloatTensor(label_list).reshape(-1, 1)
   
    with torch.no_grad():
        output = student_model(graphlist_adj[0], mask=all_mask)
        output = output[all_mask]
        output0 = torch.sigmoid(output[all_label==0]).cpu().detach().numpy()
        output1 = torch.sigmoid(output[all_label == 1]).cpu().detach().numpy()
        plt.hist(output0, bins=600, range=(0, 1), alpha=0.5, color='blue')
        plt.hist(output1, bins=600, range=(0, 1), alpha=0.5, color='red')
        plt.title('Histogram of Data')
        plt.show()



if __name__ == '__main__':

    args = parse_args()
    args_dic = vars(args)
    print('args_dict', args_dic)

    main(args_dic)
    print('The Training is finished!')