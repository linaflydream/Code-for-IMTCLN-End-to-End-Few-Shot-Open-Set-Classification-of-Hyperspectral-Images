import numpy as np
import scipy.io as sio
from SLSLDE import SLSLDE
import torch
import torch.nn as nn

import IMT_model
from sklearn import metrics
from sklearn.metrics import  confusion_matrix
import random
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# time.sleep(2*60*60)
oaoa,aaaa,f1f1,kappakappa,classs_aa = np.array([], dtype=np.int64),np.array([], dtype=np.int64),np.array([],
      dtype=np.int64),np.array([], dtype=np.int64),np.array([], dtype=np.int64)
aa_acc=np.array([], dtype=np.int64)

def MD_distance(support_feature, support_labels, query_features):
    NUM_SAMPLES = 1
    class_representations, class_precision_matrices = build_class_reps_and_covariance_estimates(support_feature,
                                                                                                support_labels)

    class_means = torch.stack(list(class_representations.values())).squeeze(1)
    class_precision_matrices = torch.stack(list(class_precision_matrices.values()))

    # grabbing the number of classes and query examples for easier use later in the function
    number_of_classes = class_means.size(0)
    number_of_targets = query_features.size(0)

    repeated_target = query_features.repeat(1, number_of_classes).view(-1, class_means.size(1))
    repeated_class_means = class_means.repeat(number_of_targets, 1)
    repeated_difference = (repeated_class_means - repeated_target)
    repeated_difference = repeated_difference.view(number_of_targets, number_of_classes,
                                                   repeated_difference.size(1)).permute(1, 0, 2)
    first_half = torch.matmul(repeated_difference, class_precision_matrices)
    sample_logits = torch.mul(first_half, repeated_difference).sum(dim=2).transpose(1, 0) * -1

    # return split_first_dim_linear(sample_logits, [NUM_SAMPLES, query_features.shape[0]])
    return sample_logits


def build_class_reps_and_covariance_estimates(context_features, context_labels):
    class_representations = {}
    class_precision_matrices = {}
    task_covariance_estimate = estimate_cov(context_features)
    for c in torch.unique(context_labels):
        # filter out feature vectors which have class c
        class_mask = torch.eq(context_labels, c)
        class_mask_indices = torch.nonzero(class_mask)
        class_features = torch.index_select(context_features, 0, torch.reshape(class_mask_indices, (-1,)).cuda())
        # mean pooling examples to form class means
        class_rep = mean_pooling(class_features)
        # updating the class representations dictionary with the mean pooled representation
        class_representations[c.item()] = class_rep
        """
        Calculating the mixing ratio lambda_k_tau for regularizing the class level estimate with the task level estimate."
        Then using this ratio, to mix the two estimate; further regularizing with the identity matrix to assure invertability, and then
        inverting the resulting matrix, to obtain the regularized precision matrix. This tensor is then saved in the corresponding
        dictionary for use later in infering of the query data points.
        """
        lambda_k_tau = (class_features.size(0) / (class_features.size(0) + 1))
        class_precision_matrices[c.item()] = torch.inverse(
            (lambda_k_tau * estimate_cov(class_features)) + ((1 - lambda_k_tau) * task_covariance_estimate) \
            + torch.eye(class_features.size(1), class_features.size(1)).to(device))
    return class_representations, class_precision_matrices


def estimate_cov(examples, rowvar=False, inplace=False):
    if examples.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if examples.dim() < 2:
        examples = examples.view(1, -1)
    if not rowvar and examples.size(0) != 1:
        examples = examples.t()
    factor = 1.0 / (examples.size(1) - 1)
    if inplace:
        examples -= torch.mean(examples, dim=1, keepdim=True)
    else:
        examples = examples - torch.mean(examples, dim=1, keepdim=True)
    examples_t = examples.t()
    return factor * examples.matmul(examples_t).squeeze()\

def mean_pooling(x):
    return torch.mean(x, dim=0, keepdim=True)



for jjj in range(10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = sio.loadmat('/home/sxp/sxp-2/data/Salinas.mat')
    X = np.array(X['Salinas'])
    label = sio.loadmat('/home/sxp/sxp-2/data/salinas_label.mat')
    label = np.array(label['salinas_label']).reshape([111104, 1])
    label_short = label[np.where(label != 0)[0], :]
    label_short2 = label[np.where(np.logical_and(label != 0, label != 16))[0], :]

    num_class = 15
    m1 = 512
    m2 = 217
    d = 30

    Ni = 5
    kw = 20
    km = 20
    S = 7
  
    loss1=[]
    LDE = SLSLDE()

    X = X.reshape([111104, -1])
    X_short = X[np.where(label != 0)[0], :]  # 10249 200
    #
    index_train = LDE.find_Ni_labels(Ni, label_short, num_class)  # 444   indices
    label_train = label_short[index_train, :]
    index_train2 = LDE.find_Ni_labels(Ni, label_short2, num_class)  # 444   indices
    label_train2 = label_short2[index_train2, :]  # 444,1   label
    Location = LDE.location(m1, m2)
    # Ww, Wb, nei_w, ww, nei_b = LDE.Ww_Wb(index_train,label_train,kw,km,X,b,Location,m1,m2,S,label)
    ww = torch.from_numpy(np.load('ww3.npy')).to(device)
    nei_w = torch.from_numpy(np.load('nei_w3.npy')).to(device)
    nei_b = torch.from_numpy(np.load('nei_b3.npy')).to(device)
    SS = 5

    X_SS_short_L = torch.from_numpy(LDE.generate_data(X, Location, label, m1, m2, SS)).to(device)  # 10249 5 5 202
    X_SS_short_L2 = LDE.generate_data2(X, Location, label, m1, m2, SS)  # 10249 5 5 202
    X_SS_label_L2 = torch.from_numpy(X_SS_short_L2[index_train2]).to(device)  # 444 5 5 202
    label_16_2 = LDE.generate_label(label_short2, num_class)  # 10249 16
    label_train_16_2 = torch.from_numpy(label_16_2[index_train2, :]).to(device)
    label_tensor_indices = torch.argmax(label_train_16_2 , dim=1).to(device)
    cost_D1, loss_cross_entropy1, f_loss1, f_loss21, loss_mse41 ,un_acc=[], [], [], [], [], []
    acc3 = []
    D_Loss = list()
    C_Loss = list()
    Pd_Acc = list()

    n = 54129
    tr = list(range(n))
    n_i = label_train_16_2.shape[0]
    B_i = list(range(n_i))
    band = 206
    epoch =300
    bt = 300
    iterations = int(n / bt)
    learn_rate = 8e-5
    DEN = IMT_model.sxp(num_class, 5, band).to(device)
    criterion = nn.CrossEntropyLoss()

    params_list = [{'params': DEN.parameters()},
                   {'params': criterion.parameters()}]
    torch.nn.utils.clip_grad_norm_(DEN.parameters(), 0.1)
    optimizer = torch.optim.Adam(params_list, lr=learn_rate)
    DEN.apply(DEN.kaiming_init)

    criterion.to(device)
    BL = len(B_i)
 
    def count_parameters(model):
        total_params = sum(p.numel() for p in model.parameters())
        return total_params
    total_params = count_parameters(DEN)
    start_time=time.time()
    print(total_params)
    selfsoftmax = torch.nn.Softmax(dim=1).to(device)
    selfmse = np.array([], dtype=np.int64)
    for i in range(epoch + 1):
        np.random.shuffle(tr)
        for j in range(iterations):
            with torch.set_grad_enabled(True):
                optimizer.zero_grad() 
                B_i = torch.randperm(BL).to(device)
                B = torch.tensor(tr[j * bt:min(j * bt + bt, n)]).to(device)
                label_tensor_indices2=label_tensor_indices[B_i]
                B_nei = torch.cat((torch.cat((B, nei_w[B].view(-1)), dim=0), nei_b[B].view(-1)), dim=0)
                X_train_tensor = torch.cat((X_SS_short_L[B_nei], X_SS_label_L2[B_i]), dim=0)
                X_train_tensor = X_train_tensor.permute(0, 3, 1, 2)
                z, h, f = DEN(X_train_tensor, flag=True)
                original_labels = label_tensor_indices2
                features = f[-BL:, :]
                specified_indices = []
                Ww=ww[B]
                B = len(B)
                BB = B * kw + B
                for class_label in range(num_class):
                    indices_in_class = torch.nonzero(original_labels == class_label).squeeze()
                    specified_index = random.choice(indices_in_class)
                    specified_indices.append(specified_index)
                specified_indices = torch.stack(specified_indices)
                # 获取指定标签对应的特征向量
                specified_features = features[specified_indices]
                specified_labels = original_labels[specified_indices]
                # 获取剩余标签对应的特征向量
                remaining_indices = torch.tensor([i for i in range(len(original_labels)) if i not in specified_indices]).to(device)
                remaining_features = features[remaining_indices]
                remaining_labels = original_labels[remaining_indices]
                logits = MD_distance(specified_features, specified_labels, remaining_features)
                f_loss = criterion(logits, remaining_labels.long().cuda())
                logits2 = MD_distance(specified_features, specified_labels, specified_features)
                f_loss2 = criterion(logits2, specified_labels.long().cuda())
                pre_nc = z[-BL:, :]

                loss_cross_entropy = criterion(pre_nc, label_tensor_indices2)

                logits3 = MD_distance(features, original_labels, f[:B, :])
                probs = selfsoftmax(logits3)

                max_probs2, max_indices2 = torch.max(logits3, dim=1)

                mask10 = torch.rand(B) < 0.5
                z1 = h[:B, :].reshape(B, 1, num_class + 1)
                z2 = h[B:BB, :].reshape(B, kw, num_class + 1)
                z3 = h[BB:-BL, :].reshape(B, km, num_class + 1)
                Dw = torch.sum(torch.square(z1[mask10] - z2[mask10]), dim=2)
                Db = torch.sum(torch.square(z1[mask10] - z3[mask10]), dim=2)
                D_Ww = torch.mean(torch.sum(Ww[mask10] * Dw, dim=1))
                D_Wb = torch.mean(torch.sum(Db, dim=1))
                cost_D = D_Ww + torch.exp(-D_Wb)

                loss_mse4 = torch.tensor(0.0)
                if i == 100:
                    top_values, top_indices = torch.topk(max_probs2, largest=False, k=1, dim=0)
                    selfmse = np.append(selfmse,
                                         (- torch.log(h[:B, :][top_indices][:, num_class])).cpu().detach().numpy())
                if i >= 100:
                    top_values, top_indices = torch.topk(max_probs2, largest=False, k=1, dim=0)
                    loss_mse4 = -  torch.log(h[:B, :][top_indices][:, num_class])
                    if loss_mse4 >= 1.5* np.mean(selfmse):
                        loss_mse4 = 0.01*loss_mse4
                    else:
                        selfmse = np.append(selfmse, loss_mse4.cpu().detach().numpy())


                loss_all = cost_D + loss_cross_entropy + f_loss + f_loss2 + loss_mse4
                loss_all.backward()
                optimizer.step()
        if i % 5 == 0 :
            DEN.eval()
            correct3, correct2, total = 0, 0, 0
            with torch.no_grad():  # 不需要计算梯度
                X_short_TEST = X_SS_short_L.permute(0, 3, 1, 2)
                z = DEN(X_short_TEST, flag=False)
                label_test_short = torch.from_numpy(label_short - 1).float().to(device)
                total += label_test_short.size(0)
                pre2 = torch.argmax(z, dim=1).view(-1, 1).float()
                correct2 += (pre2 == label_test_short).sum().item()
                acc2 = correct2 / 54129 * 100.0
                C = metrics.confusion_matrix(label_test_short.cpu().detach().numpy().squeeze(),
                                                pre2.cpu().detach().numpy().squeeze())
                # 最后一个类别的真正例 (TP)
                TP = C[-1, -1]
                # 最后一个类别的假正例 (FP)，即最后一列的和减去TP
                FP = np.sum(C[-1, :]) - TP
                # 计算准确率
                if TP + FP == 0:
                    accuracy = 0
                else:
                    accuracy = TP / (TP + FP)
                acc3 = np.append(acc3, acc2)
                un_acc = np.append(un_acc, accuracy)
            
                print(i, acc2, accuracy)
            DEN.train()
    