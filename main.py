import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import KFold
import scipy.io as sio
from utils import feature_normalized, create_graph_from_embedding, sample_mask, preprocess_adj, preprocess_features
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from heapq import nlargest
from collections import Counter


def process_adj_SF(A):
    adj = create_graph_from_embedding(A, name='knn', n=30)

    adj, edge = preprocess_adj(adj)
    adj = adj.todense()
    adj = torch.from_numpy(adj).to(torch.float32)
    return adj


def permute_matrix(feature, label, index):
    seed_100 = np.arange(100)
    print("seed: ", seed_100[index])
    rand_list = np.random.RandomState(seed_100[index]).permutation(feature.shape[0])
    permute_feature = []
    permute_label = []
    for i in range(len(rand_list)):
        permute_feature.append(feature[rand_list[i]])
        permute_label.append(label[rand_list[i]])

    return np.array(permute_feature), np.array(permute_label)


def get_data_AD_SF(index, dataset):
    if dataset == "AD-NC":
        feature_file = '/data/AD_NC/feature.mat'
        label_file = '/data/AD_NC/label.mat'

    elif dataset == "AD-MCI":
        feature_file = '/data/AD_MCI/feature.mat'
        label_file = '/data/AD_MCI/label.mat'

    elif dataset == "NC-MCI":
        feature_file = '/data/NC_MCI/feature.mat'
        label_file = '/data/NC_MCI/label.mat'

    elif dataset == "MCIn-MCIc":
        feature_file = '/data/MCIn_MCIc/feature.mat'
        label_file = '/data/MCIn_MCIc/label.mat'

    data_feature = sio.loadmat(feature_file)
    data_label = sio.loadmat(label_file)

    feature = data_feature['feature']
    label = data_label['label']
    label = np.argmax(label, axis=1)

    features, labels = permute_matrix(feature, label, index)
    features = feature_normalized(features)

    adj = process_adj_SF(features)
    s = process_adj_SF(features.T)
    features = preprocess_features(features)

    return adj, features, s, labels


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, input_feature, adjacency):

        support = torch.mm(input_feature, self.weight)
        output = torch.mm(adjacency, support)
        if self.use_bias:
            output += self.bias
        return output


class SYNet_AD(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SYNet_AD, self).__init__()
        nd_1 = 128
        nd_2 = 64
        nd_3 = 32
        n_4 = 2

        self.gcn1_1 = GraphConvolution(input_dim, nd_1)  # AD-NC
        self.gcn1_2 = GraphConvolution(nd_1, nd_2)

        self.gcn2_1 = GraphConvolution(93, nd_3)
        self.gcn2_2 = GraphConvolution(nd_3, output_dim)

        self.linear1 = torch.nn.Linear(93, n_4)  # AD-NC
        self.linear2 = torch.nn.Linear(n_4, output_dim)
        self.linear3 = torch.nn.Linear(output_dim, output_dim)


        self.W = nn.Parameter(torch.ones(input_dim, 93))  # nxd

    def forward(self, adjacency1, feature, adjacency2):
        # adjacency1:S  adjacency2:A feature:X
        lam1 = 1e-5
        lam2 = 1e-3
        y_hat = F.relu(self.gcn1_1(feature.t(), adjacency1))
        y_hat = F.dropout(y_hat, 0.2, training=self.training)
        y_hat = self.gcn1_2(y_hat, adjacency1)
        s_hat = process_adj_SF((y_hat.cpu().detach().numpy())).to(device)

        x_hat = F.relu(self.gcn2_1(feature, adjacency2))
        x_hat = F.dropout(x_hat, 0.2, training=self.training)
        x_hat = self.gcn2_2(x_hat, adjacency2)
        a_hat = process_adj_SF(x_hat.cpu().detach().numpy()).to(device)

        s_hat += lam1 * torch.eye(s_hat.size(0)).to(y_hat.device)
        a_hat += lam1 * torch.eye(a_hat.size(0)).to(y_hat.device)

        # (A'+I+A)XS'(S'+I+S)
        s_hat += lam2 * adjacency1.to(y_hat.device)  # dxd
        a_hat += lam2 * adjacency2.to(y_hat.device)  # nxn

        X_a = torch.mm(feature, s_hat)
        X_s = torch.mm(a_hat, feature)
        new_feature = torch.mm(a_hat, X_a)
        h = self.linear1(new_feature)
        h = F.relu(self.linear2(h))
        h = self.linear3(h)

        logits = h

        return logits, X_a, X_s


def train_SF():
    for epoch in range(epochs):
        model.train()
        logits, X_a, X_s = model(s, tensor_x, adj)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return X_a, X_s


def sen_spe(a, b):
    TP = 0
    FN = 0
    TN = 0
    FP = 0
    for i in range(len(a)):  # a:label
        if a[i] == 1 and b[i] == 1:
            TP = TP + 1
        elif a[i] == 1 and b[i] == 0:
            FN = FN + 1
        elif a[i] == 0 and b[i] == 1:
            FP = FP + 1
        elif a[i] == 0 and b[i] == 0:
            TN = TN + 1
        else:
            pass

    TPR = TP / (TP + FN + 1e-6)  # True positive rate, Sensitivity
    TNR = TN / (TN + FP + 1e-6)  # True Negative Rate, Specificity
    return TPR, TNR


def test_SF(mask):
    model.eval()

    logits, X_a, X_s = model(s, tensor_x, adj)
    mask_logits = logits[mask]

    predicted = mask_logits.max(dim=1)[1].long()
    label = labels[mask]

    predicted = predicted.cpu().detach().numpy()
    label = label.cpu().detach().numpy()

    accuracy = accuracy_score(label, predicted)
    sensitivity, specificity = sen_spe(label, predicted)
    continue_pred = F.softmax(mask_logits, dim=1)
    continue_pred = continue_pred[:, 1]
    auc = roc_auc_score(label, continue_pred.cpu().detach().numpy())
    return accuracy, sensitivity, specificity, auc


def topn_dict(d, n):
    return nlargest(n, d, key=lambda k: d[k])


dataset = ["AD-NC", "AD-MCI", "NC-MCI", "MCIn-MCIc"]

for dataset_index in range(len(dataset)):
    if __name__ == "__main__":
        learning_rate = 0.005
        weight_decay = 5e-4
        epochs = 500

        device = "cuda:1" if torch.cuda.is_available() else "cpu"
        ACC_1 = []
        log = '{:.4f}  {:.4f} {:.4f} {:.4f}'
        List_acc = []
        List_sen = []
        List_spe = []
        List_auc = []
        FEA = []

        for j in range(100):
            adj, features, s, labels = get_data_AD_SF(j, dataset[dataset_index])

            labels = torch.from_numpy(labels).long().to(device)
            adj = adj.to(device)
            s = s.to(device)

            input = features.shape[0]
            output = 2

            kf = KFold(n_splits=5)
            ACC = []
            SEN = []
            SPE = []
            AUC = []
            FEATURE = []
            best_acc = 0
            torch.manual_seed(0)
            index = 0
            for idx_train, idx_test in kf.split(features):
                pth = dataset[dataset_index] + "net_params_" + str(index) + ".pth"
                train_mask = sample_mask(idx_train, labels.shape[0])
                val_mask = sample_mask(idx_test, labels.shape[0])
                test_mask = sample_mask(idx_test, labels.shape[0])

                tensor_x = torch.from_numpy(features).to(torch.float32).to(device)

                model = SYNet_AD(input, output).to(device)
                optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

                #
                # model.load_state_dict(torch.load(pth))
                #
                X_a, X_s = train_SF()
                #
                # torch.save(model.state_dict(), pth)
                feature_sum = torch.norm(X_s, p=2, dim=0).cpu().detach().numpy()
                fea = nlargest(20, range(len(feature_sum)), feature_sum.take)
                fea.sort()
                print(fea)
                FEATURE = FEATURE + fea

                test_accs, test_sen, test_spe, test_auc = test_SF(test_mask)

                print(test_accs)

                ACC.append(test_accs)
                SEN.append(test_sen)
                SPE.append(test_spe)
                AUC.append(test_auc)

                index += 1
                count = topn_dict(Counter(FEATURE), 20)
                # print(count.sort())
                FEA = FEA + count

            mean_acc = np.mean(np.array(ACC))
            List_acc.append(mean_acc)
            print(j)
            print(dataset[dataset_index], "： Mean times", j, "ACC+/-std", mean_acc, "+/-", np.std(np.array(ACC)))

            mean_sen = np.mean(np.array(SEN))
            List_sen.append(mean_sen)
            print("Mean times", j, "SEN+/-std", mean_sen, "+/-", np.std(np.array(SEN)))

            mean_spe = np.mean(np.array(SPE))
            List_spe.append(mean_spe)
            print("Mean times", j, "SPE+/-std", mean_spe, "+/-", np.std(np.array(SPE)))

            mean_auc = np.mean(np.array(AUC))
            List_auc.append(mean_auc)
            print("Mean times", j, "AUC+/-std", mean_auc, "+/-", np.std(np.array(AUC)))

        print("The final result after 100 times: ")
        print("Mean times", j, "ACC+/-std", np.mean(np.array(List_acc)), "+/-", np.std(np.array(List_acc)))
        print("Mean times", j, "SEN+/-std", np.mean(np.array(List_sen)), "+/-", np.std(np.array(List_sen)))
        print("Mean times", j, "SPE+/-std", np.mean(np.array(List_spe)), "+/-", np.std(np.array(List_spe)))
        print("Mean times", j, "AUC+/-std", np.mean(np.array(List_auc)), "+/-", np.std(np.array(List_auc)))

        print(Counter(FEA))