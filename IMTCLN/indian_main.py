import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from SLSLDE import SLSLDE
from sklearn import metrics
import random
import argparse
import IMT_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def MD_distance(support_feature, support_labels, query_features):
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
    return factor * examples.matmul(examples_t).squeeze()


def mean_pooling(x):
    return torch.mean(x, dim=0, keepdim=True)

def main():
    parser = argparse.ArgumentParser(description='Hyperspectral Image Classification')
    
    # 数据路径参数
    parser.add_argument('--data_path', type=str, default='data/indian.mat', 
                       help='Path to the hyperspectral data file')
    parser.add_argument('--label_path', type=str, default='data/label.mat',
                       help='Path to the label data file')
    
    # 模型训练参数
    parser.add_argument('--num_class', type=int, default=15, 
                       help='Number of classes in the dataset')
    parser.add_argument('--m1', type=int, default=145, 
                       help='First dimension parameter for location')
    parser.add_argument('--m2', type=int, default=145, 
                       help='Second dimension parameter for location')
    parser.add_argument('--d', type=int, default=30, 
                       help='Feature dimension parameter')
    
    # 采样参数
    parser.add_argument('--Ni', type=int, default=5, 
                       help='Number of samples per class for training')
    parser.add_argument('--kw', type=int, default=10, 
                       help='Number of within-class neighbors')
    parser.add_argument('--km', type=int, default=10, 
                       help='Number of between-class neighbors')
    parser.add_argument('--S', type=int, default=5, 
                       help='Spatial window size for preprocessing')
    parser.add_argument('--SS', type=int, default=5, 
                       help='Spatial window size for data generation')
    
    # 训练超参数
    parser.add_argument('--epochs', type=int, default=500, 
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=200, 
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=8e-5, 
                       help='Learning rate for optimizer')
    parser.add_argument('--mask_prob', type=float, default=0.5, 
                       help='Probability for random masking')
    
    # 图对比学习样本以及权重路径
    parser.add_argument('--ww_path', type=str, default='ww.npy', 
                       help='Path to ww weights file')
    parser.add_argument('--nei_w_path', type=str, default='nei_w.npy', 
                       help='Path to within-class neighbors file')
    parser.add_argument('--nei_b_path', type=str, default='nei_b.npy', 
                       help='Path to between-class neighbors file')
    
    # 实验设置
    parser.add_argument('--num_experiments', type=int, default=5, 
                       help='Number of times to repeat the experiment')
    parser.add_argument('--eval_interval', type=int, default=5, 
                       help='Evaluation interval during training')
    
    args = parser.parse_args()
    
 
    
    # 主训练循环
    for jjj in range(args.num_experiments):
       
        
        # 加载数据
        X = sio.loadmat(args.data_path)
        X = np.array(X['indian'])
        label = sio.loadmat(args.label_path)
        label = np.array(label['label']).reshape([21025, 1])
        
        # 数据预处理
        label_short = label[np.where(label != 0)[0], :]
        label_short2 = label[np.where(np.logical_and(label != 0, label != 16))[0], :]
        selfmse = np.array([], dtype=np.int64)
        
        # 初始化算法
        LDE = SLSLDE()
        X = LDE.WMF(X, args.S)
        
        # 准备训练数据
        index_train2 = LDE.find_Ni_labels(args.Ni, label_short2, args.num_class)
        Location = LDE.location(args.m1, args.m2)
        
        # 加载预计算同质图与边权重
        ww = torch.from_numpy(np.load(args.ww_path)).to(device)
        nei_w = torch.from_numpy(np.load(args.nei_w_path)).to(device)
        nei_b = torch.from_numpy(np.load(args.nei_b_path)).to(device)
        
        # 生成空间光谱数据
        X_SS_short_L = torch.from_numpy(LDE.generate_data(X, Location, label, args.m1, args.m2, args.SS)).to(device)
        X_SS_short_L2 = LDE.generate_data2(X, Location, label, args.m1, args.m2, args.SS)
        X_SS_label_L2 = torch.from_numpy(X_SS_short_L2[index_train2]).to(device)
        label_16_2 = LDE.generate_label(label_short2, args.num_class)
        label_train_16_2 = torch.from_numpy(label_16_2[index_train2, :]).to(device)
        label_tensor_indices = torch.argmax(label_train_16_2, dim=1).to(device)
        
        # 训练设置
        n = 10249
        tr = list(range(n))
        n_i = label_train_16_2.shape[0]
        B_i = list(range(n_i))
        band = 202
        bt = args.batch_size
        iterations = int(n / bt)
        
        # 初始化模型
        DEN = IMT_model.IMT(args.num_class, args.SS, band).to(device)
        criterion = nn.CrossEntropyLoss()
        params_list = [{'params': DEN.parameters()}]
        optimizer = torch.optim.Adam(params_list, lr=args.learning_rate)
        DEN.apply(DEN.kaiming_init)
        criterion.to(device)
        BL = len(B_i)  
        DEN.train()

        
        # 训练循环
        for i in range(args.epochs + 1):
            np.random.shuffle(tr)
            for j in range(iterations):
                with torch.set_grad_enabled(True):
                    optimizer.zero_grad()
                    B_i = torch.randperm(BL).to(device)
                    B = torch.tensor(tr[j * bt:min(j * bt + bt, n)]).to(device)
                    label_tensor_indices2 = label_tensor_indices[B_i]
                    
                    # 构建输入数据以及对应的图结构样本输入
                    B_nei = torch.cat((torch.cat((B, nei_w[B].view(-1)), dim=0), nei_b[B].view(-1)), dim=0)
                    X_train_tensor = torch.cat((X_SS_short_L[B_nei], X_SS_label_L2[B_i]), dim=0)
                    X_train_tensor = X_train_tensor.permute(0, 3, 1, 2).to(device)
                    
                    # 前向传播
                    z, h, f = DEN(X_train_tensor, flag=True)
                    original_labels = label_tensor_indices2
                    features = f[-BL:, :]
                    specified_indices = []
                    Ww = ww[B]
                    B_len = len(B)
                    BB = B_len * args.kw + B_len
                    
                    # 选择每个类一个样本
                    for class_label in range(args.num_class):
                        indices_in_class = torch.nonzero(original_labels == class_label).squeeze()
                        if len(indices_in_class) > 0:
                            specified_index = random.choice(indices_in_class)
                            specified_indices.append(specified_index)
                    specified_indices = torch.stack(specified_indices)
                    
                    # 特征分组
                    specified_features = features[specified_indices]
                    specified_labels = original_labels[specified_indices]
                    remaining_indices = torch.tensor([i for i in range(len(original_labels)) if i not in specified_indices])
                    remaining_features = features[remaining_indices]
                    remaining_labels = original_labels[remaining_indices]
                    
                    # 计算马氏距离损失
                    logits = MD_distance(specified_features, specified_labels, remaining_features)
                    f_loss = criterion(logits, remaining_labels.long().cuda())
                    logits2 = MD_distance(specified_features, specified_labels, specified_features)
                    f_loss2 = criterion(logits2, specified_labels.long().cuda())
                    pre_nc = z[-BL:, :]
                    loss_cross_entropy = criterion(pre_nc, label_tensor_indices2)
                    
                    logits3 = MD_distance(features, original_labels, f[:B_len, :])
                 
                    max_probs2, _ = torch.max(logits3, dim=1)
                    
                    # 计算图约束对比损失
                    mask10 = torch.rand(B_len) < args.mask_prob
                    z1 = h[:B_len, :].reshape(B_len, 1, args.num_class + 1)
                    z2 = h[B_len:BB, :].reshape(B_len, args.kw, args.num_class + 1)
                    z3 = h[BB:-BL, :].reshape(B_len, args.km, args.num_class + 1)
                    Dw = torch.sum(torch.square(z1[mask10] - z2[mask10]), dim=2)
                    Db = torch.sum(torch.square(z1[mask10] - z3[mask10]), dim=2)
                    D_Ww = torch.mean(torch.sum(Ww[mask10] * Dw, dim=1))
                    D_Wb = torch.mean(torch.sum(Db, dim=1))
                    cost_D = D_Ww + torch.exp(-D_Wb)
                    
                    # 自监督损失
                    loss_mse4 = torch.tensor(0.0)
                    if i == 300:
                        _, top_indices = torch.topk(max_probs2, largest=False, k=1, dim=0)#多种未知类需调整k的值
                        selfmse = np.append(selfmse, (-torch.log(h[:B_len, :][top_indices][:, args.num_class])).cpu().detach().numpy())
                    if i >= 300:
                        _, top_indices = torch.topk(max_probs2, largest=False, k=1, dim=0)
                        loss_mse4 = -torch.log(h[:B_len, :][top_indices][:, args.num_class])
                        if loss_mse4 >= 1.5 * np.mean(selfmse):
                            loss_mse4 = 0.01 * loss_mse4
                        else:
                            selfmse = np.append(selfmse, loss_mse4.cpu().detach().numpy())
                    
                    # 总损失和反向传播
                    loss_all = cost_D + loss_cross_entropy + f_loss + f_loss2 + loss_mse4
                    loss_all.backward()
                    optimizer.step()
            
            # 评估
            if i % args.eval_interval == 0:
                DEN.eval()
                correct2, total = 0, 0
                with torch.no_grad():
                    X_short_TEST = X_SS_short_L.permute(0, 3, 1, 2)
                    z = DEN(X_short_TEST, flag=False)
                    label_test_short = torch.from_numpy(label_short - 1).float().to(device)
                    total += label_test_short.size(0)
                    pre2 = torch.argmax(z, dim=1).view(-1, 1).float()
                    correct2 += (pre2 == label_test_short).sum().item()
                    acc2 = correct2 / 10249 * 100.0
                    
                    C = metrics.confusion_matrix(label_test_short.cpu().detach().numpy().squeeze(),
                                                pre2.cpu().detach().numpy().squeeze())
                    TP = C[-1, -1]
                    FP = np.sum(C[-1, :]) - TP
                    accuracy = TP / (TP + FP) if TP + FP > 0 else 0
                    
                    print(f'Experiment {jjj+1}, Epoch {i}, Accuracy: {acc2:.2f}%, Class Accuracy: {accuracy:.4f}')
                
                DEN.train()

if __name__ == '__main__':
    main()