import argparse
import math
# import time
import numpy as np
import torch
import torch.nn as nn
# import importlib
import read_data
import random
from model import UTDN
import utils
# import read_meodata
# import read_ncdata

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
def evaluate(Data,model,evaluate2,evaluate1,batch_size,target):
    model.eval()
    # X = Data.valid[0]
    # Y= Data.valid[1]
    X = Data.test[0]
    Y = Data.test[1]


    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    rmses = 0
    maes = 0
    temp = 0.3
    evaluate3 = nn.MSELoss(size_average=False,reduction='sum')
    evaluate4 = nn.L1Loss(size_average=False, reduction='sum')
    for X, Y in Data.get_batches(X, Y, batch_size, False):
        # X = X[:, target, :, :].unsqueeze(1)
        X_local, X_global, Y = create_input(X, Y, target)
        Y = Y[:,:,target]
        output,_ = model(X_local, X_global, temp)
        # if predict is None:
        #     predict = output
        #     test = Y
        # else:
        #     try:
        #         predict = torch.cat((predict, output))
        #         test = torch.cat((test, Y))
        #     except:
        #         continue

        scale = Data.scale.expand(output.size(0), Data.prelen,Data.predicted)[:,:,target]
        total_loss += evaluate2(output * scale, Y * scale).data
        total_loss_l1 += evaluate1(output * scale, Y * scale).data
        # n_samples += (output.size(0) * Data.predicted)
        n_samples += (output.size(0)*output.size(1))


        rmses += evaluate3(output * scale, Y * scale).data
        maes += evaluate4(output * scale, Y * scale).data
        # if output.data is None:
        #     print("!")

    rmse = math.sqrt(rmses/n_samples)
    mae = maes / n_samples

    # print("test rse {:5.4f} | test rae {:5.4f} ".format(rse,rae))
    return mae,rmse

def create_input(X,Y):
    #X:B,T, 35,fea+station
    #x_local: B,T,F
    #x_global:B,T,35
    X = X.permute(0, 2, 1, 3)  # B,35,T,F
    batch = X.size(0)
    station_idx = torch.LongTensor([random.uniform(0, 35) for i in range(batch)]).to(device)
    # target_station
    index_x = torch.arange(0, X.size(0), 1)
    X_ind = X[index_x,station_idx] #B,T,F
    X_local = X_ind[:,:,:].transpose(1,2)
    X_local = X_local.reshape(batch,-1,args.window//args.seglen,args.seglen)
    X_global = X[:,:,-args.seglen:,:] #B,35,T,F

    Y = Y.permute(0,2,1,3)
    label = Y[index_x,station_idx,:,:]
    return X_local,X_global,label

def train(Data,args):
    model = UTDN.Model(args, Data, device)
    model = model.to(device)

    print(args)

    if args.L1Loss:
        criterion = nn.L1Loss(size_average=False).to(device)
    else:
        criterion = nn.MSELoss(size_average=False).to(device)

    evaluateL2 = nn.MSELoss(size_average=False)
    evaluateL1 = nn.L1Loss(size_average=False)

    score = []
    # optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    print('begin training')
    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0
        n_samples = 0
        batch_id = 0
        temp = 10
        for X, Y in Data.get_batches(Data.train[0], Data.train[1], args.batch_size, True):
            model.zero_grad()
            # Y = Y[:, 0, :3]
            X_local, X_global, Y = create_input(X, Y, args.target)
            Y = Y[:,:,args.target]
            batch_id += 1
            if batch_id % 10 == 1:
                temp = np.maximum(temp * np.exp(-1e-4 * batch_id), 0.3)
            # print( torch.sum(torch.isnan(X)))
            output, avg_score = model(X_local, X_global, temp)
            #rolling
            # output = []
            # for i in range(Data.prelen):
            #     output_one, avg_score = model(X_local, X_global, temp,type = 'rolling')
            #     local_shape = X_local.shape
            #     X_local = X_local.view(local_shape[0],local_shape[1],local_shape [2]*local_shape [3])
            #     X_local = torch.cat([X_local[:,:,1:],output_one],dim=-1)
            #     X_local = X_local.view(local_shape[0],local_shape[1],local_shape [2],local_shape [3])
            #     output.append(output_one)
            # output = torch.cat(output,dim=-1)
            # score.append(avg_score)
            scale = Data.scale.expand(output.size(0), Data.prelen,Data.predicted)[:,:,args.target]
            loss = criterion(output * scale, Y * scale)
            loss.backward()
            grad_norm = optim.step()
            total_loss += loss.data
            n_samples += (output.size(0) * Data.predicted)
        # score = torch.cat(score,dim=0)
        # print('AVG SCORE',torch.mean(score,dim=0))
        # if epoch%10 == 0:
            mae,rmse = evaluate(Data, model, evaluateL2, evaluateL1, args.batch_size,args.target)
            print('======METRIC-Epoch:{}====='.format(epoch))
            print("train loss",total_loss)
            print("rmse = {:.6f}|mae {:.6f}".format(rmse,mae))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch weather series forecasting')
    parser.add_argument('--data', type=str, default="aoti2017.csv",
                        help='location of the data file')
    parser.add_argument('--hidden_state_features', type=int, default=11,
                        help='number of features in LSTMs hidden states')
    parser.add_argument('--num_layers_lstm', type=int, default=1,
                        help='num of lstm layers')
    parser.add_argument('--predict_fea', type=int, default=6,
                        help='num of predicted feature')
    # , required=True
    parser.add_argument('--model', type=str, default='CNN',
                        help='')

    parser.add_argument('--window', type=int, default=72,
                        help='window size')
    parser.add_argument('--target', type=int, default=0,
                        help='target')
    parser.add_argument('--epochs', type=int, default=6000,
                        help='upper epoch limit')  # 30
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--seed', type=int, default=54321,
                        help='random seed')
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default='model/model.pt',
                        help='path to save the final model')
    parser.add_argument('--cuda', type=str, default=False)
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=1e-05)
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--horizon', type=int, default=1)
    parser.add_argument('--L1Loss', type=bool, default=True)
    parser.add_argument('--normalize', type=int, default=2)
    parser.add_argument('--output_fun', type=str, default='sigmoid')
    parser.add_argument('--seglen', type=int, default=12,
                        help='window size')
    args = parser.parse_args()

    random.seed(args.seed)

    # traindata,testdata = read_data.read_m4data()
    # Data = utils_m4.Data_utility(traindata,testdata, 'CNN', 0.6, 0.2, args.horizon, args.window, device,
    #                              args.seglen,args.normalize)  # SPLITS THE DATA IN TRAIN AND VALIDATION SET, ALONG WITH OTHER THINGS, SEE CODE FOR MORE

    #load data
    # all_station,station_ind = read_waterdata.read_df()
    # all_station, station_ind = read_meodata.read_df()
    all_station, station_ind = read_data.read_df()

    Data = utils.Data_utility(all_station,station_ind,'CNN', 0.6, 0.2, args.horizon, args.window, device,
                              args.normalize)  # SPLITS THE DATA IN TRAIN AND VALIDATION SET, ALONG WITH OTHER THINGS, SEE CODE FOR MORE

    train(Data,args)


