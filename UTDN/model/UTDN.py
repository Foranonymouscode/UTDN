import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# device = torch.device('cpu')
class Model(nn.Module):
    def __init__(self, args, data, device):
        super(Model, self).__init__()
        self.device = device
        self.prelen = data.prelen
        self.window_length = args.window  # window, read about it after understanding the flow of the code...What is window size? --- temporal window size (default 24 hours * 7)
        self.seglen = args.seglen
        self.original_columns = data.m  # the number of columns or features
        self.hidden_state_features = args.hidden_state_features
        self.output_num = args.predict_fea
        self.predict = data.predicted
        self.num_layers_lstm = args.num_layers_lstm
        self.lstm = nn.LSTM(input_size=self.original_columns, hidden_size=self.hidden_state_features,
                            num_layers=self.num_layers_lstm,
                            batch_first=True,
                            bidirectional=False)

        self.BN = nn.BatchNorm1d(self.hidden_state_features, affine=False)
        self.mlp = nn.Linear(self.hidden_state_features, self.output_num)
        self.cos = nn.CosineSimilarity(dim=3, eps=1e-6)

        '''
        ditailed cnn
        '''
        # self.dil1 = nn.Conv1d(12,32,kernel_size=2,dilation=1)
        # self.dil2 = nn.Conv1d(in_channels=32,out_channels=32,kernel_size=2,
        #                         dilation=2)
        # self.dil3 = nn.Conv1d(in_channels=32,out_channels=32,kernel_size=2,
        #                         dilation=4)
        # self.dil4 = nn.Conv1d(in_channels=32,out_channels=12,kernel_size=2,
        #                         dilation=8)
        # self.conv_gate = nn.Conv1d(in_channels=12,out_channels=12,kernel_size=
        #     1)
        self.dil1 = nn.Conv2d(self.original_columns, 32, kernel_size=(
            self.window_length // self.seglen, 2), dilation=1)
        self.dil2 = nn.Conv2d(in_channels=32, out_channels=12, kernel_size=(
            1, 2),
                              dilation=2)
        self.dil3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(
            1, 2),
                              dilation=4)
        self.dil4 = nn.Conv2d(in_channels=32, out_channels=12, kernel_size=(
            1, 2),
                              dilation=8)
        self.conv_gate = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(
            1, 1))

        self.dil1_col = nn.Conv2d(self.original_columns, 32, kernel_size=(
            self.seglen, 2), dilation=1)
        self.dil2_col = nn.Conv2d(in_channels=32, out_channels=12, kernel_size=(
            1, 2), dilation=2)

        #global SE
        self.conv_g = nn.Conv2d(in_channels=35, out_channels=35, kernel_size=(
            1, 1), dilation=1)

        self.scale = data.scale

        self.fc1 = nn.Linear(56+35, 1)  # orin 12*2
        self.drop1 = nn.Dropout(0.2)
        # self.fc2 = nn.Linear(56,1)
        self.fc2 = nn.Linear(56+35, self.prelen)
        self.judge_model1 = nn.Linear((self.seglen - 1) * 2, 2)
        self.judge_model_sigm = nn.Linear((self.seglen - 1) * 2, 1)

    def forward(self, x_local,x_global, T,type = 'long'):
        x  = x_local
        batch_size = x.size(0)
        # x = x#
        # B,T,F->B,F,T
        # x = x.transpose(1,2)

        x_now = x.clone()[:, :, -1, -2]

        # gumbel
        test = self.scale.expand(x.size(0), self.predict)
        test = test.unsqueeze(2).unsqueeze(3)
        all_day = x[:, :, :-1, :-1] * test
        target_day = x[:, :, -1, :-1].unsqueeze(2)  # B,C,1,23-->B,C,D,23
        target_day = target_day.expand(-1, -1, all_day.size(2), -1) * test
        # 2 B,C,D,H ->cat B,C*D,H-->B,C*D,2H-->MLP-->B,C*D,2-->B,C,D,2 gumbel
        # mlp
        score = torch.cat((all_day, target_day), dim=3)  # B,C,D-1,23*2
        score = self.judge_model1(score)
        o_prob = torch.softmax(score, dim=-1)
        prob_hard, prob_soft = self.gumbel_softmax(torch.log(o_prob + 1e-24), T, hard=True)
        prob_soft_new = prob_hard * prob_soft
        att_score = prob_soft_new[:, :, :, 0]
        add = torch.ones_like(target_day[:, :, 1, 1]).unsqueeze(2)
        # att_score = torch.cat((att_score,add),dim=2).unsqueeze(3)
        att_score = torch.cat((att_score, add), dim=2)

        # score, indice = att_score.sort(dim=2)
        x_01 = (att_score == 0).long()
        # print(x_01.dtype)
        _, idx = x_01.sort(2, descending=True)

        avg_score = torch.sum(att_score, dim=2)
        att_score = att_score.unsqueeze(3)
        x = torch.mul(x, att_score)

        index_x1 = torch.arange(0, x.size(0), 1).unsqueeze(1).unsqueeze(2).expand(x.size(0), x.size(1), x.size(2))
        index_x2 = torch.arange(0, x.size(1), 1).unsqueeze(0).unsqueeze(2).expand(x.size(0), x.size(1), x.size(2))
        sorted_X = x[index_x1, index_x2, idx]
        sorted_X = sorted_X[:, :, -8:]
        # score = self.judge_model_sigm(score)
        # att_score = torch.sigmoid(score).squeeze()
        # add = torch.ones_like(target_day[:, :, 1, 1]).unsqueeze(2)
        # zero = torch.zeros_like(att_score)
        # ones = torch.ones_like(att_score)
        # att_score = torch.where(att_score < 0.5, zero, ones)
        # att_score = torch.cat((att_score, add), dim=2).unsqueeze(3)

        # padding with self-feature
        # all_day = x[:, :, :, :-1] * test
        # target_day = x[:, :, -1, :-1].unsqueeze(2)  # B,C,1,23-->B,C,D,23
        # target_day = target_day.expand(-1, -1, all_day.size(2), -1) * test
        # dayatt_score = self.cos(all_day.squeeze(), target_day.squeeze())  # B,1,D,24->B,D,24->B,D
        # # zero = torch.FloatTensor(dayatt_score.size(0), dayatt_score.size(1),dayatt_score.size(2)).fill_(float('-inf')).to(device)
        # zero = torch.zeros_like(dayatt_score)
        # ones = torch.ones_like(dayatt_score)
        # dayatt_score = torch.where(dayatt_score < 0.98, zero, ones)
        # att_score = dayatt_score.unsqueeze(3)

        # x = torch.mul(x, att_score)

        x_filter = torch.relu(self.dil1(sorted_X))
        x_row = torch.relu(self.dil2(x_filter))
        # x_filter = torch.relu(self.dil3(x_filter))
        # x_row = torch.relu(self.dil4(x_filter))

        x_col = torch.relu(self.dil1_col(x.transpose(2, 3)))
        x_col = torch.relu(self.dil2_col(x_col)).transpose(2, 3)
        # x_col = torch.mul(x_col,att_score)
        # x_col = x_col[:,:,-1].squeeze()

        # ablation
        # output = x_col.squeeze()
        # x_row = x_row.squeeze()
        # output = x_row[:, :, -1]

        output = x_row * x_col
        # output = output.squeeze()[:,:,-1]
        output = output[:, :, -1, -1].squeeze()
        # output = F.max_pool2d(output, (output.size(2), output.size(3))).squeeze()
        # batch_size, features, time_steps -> batch_size, features
        # take only the last timestep

        x = torch.cat([output, x_now], dim=1)
        x = F.relu(self.fc1(x))
        output_local = self.drop1(x)

        '''
        spatial cnn for global_x
        input:x_global: B,35,T,F
        '''
        x_g = self.conv_g(x_global)
        score = F.avg_pool2d(x_g,(x_g.size(2),x_g.size(3))).squeeze()
        score = F.softmax(score,dim=1).unsqueeze(2).unsqueeze(3)
        x_g = torch.mul(x_g, score)
        output_global = F.max_pool2d(x_g,(x_g.size(2),x_g.size(3))).squeeze()

        output = torch.cat([output_global, output_local], dim=1)
        if type == 'long':
            output = self.fc2(output)
        else:
            output = self.fc1(output)
        return output, avg_score

    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape).to(self.device)
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature, input_seq_tensor=None):
        '''
        :param logits: # Batch*Len*2
        :param temperature:
        :param input_seq_tensor:  K*B
        :return:
        '''
        sample = self.sample_gumbel(logits.size(), eps=1e-20)  # n_items+1  *  2

        sample_logits = sample
        y = logits + sample_logits
        # y = logits + self.sample_gumbel(logits.size())
        # print(y.shape,temperature)
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature, hard=False, input_seq_tensor=None):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(logits, temperature, input_seq_tensor)

        if not hard:
            return y  # .view(-1, latent_dim * categorical_dim)

        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        # y_hard = (y_hard - y).detach() + y
        return y_hard, y  # .view(-1, latent_dim * categorical_dim)

