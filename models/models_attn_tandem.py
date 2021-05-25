#!/usr/bin/python3
# coding: utf-8
# @Time    : 2020/11/5 10:27

import random
import operator
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import get_dict_info_batch


def mask_log_softmax(x, mask, log_flag=True):
    maxes = torch.max(x, 1, keepdim=True)[0]
    x_exp = torch.exp(x - maxes) * mask
    x_exp_sum = torch.sum(x_exp, 1, keepdim=True)
    if log_flag:
        output_custom = torch.log(x_exp / x_exp_sum)
    else:
        output_custom = x_exp / x_exp_sum
    return output_custom

class Extra_MLP(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self.pro_input_dim = parameters.pro_input_dim
        self.pro_output_dim = parameters.pro_output_dim
        self.fc_out = nn.Linear(self.pro_input_dim, self.pro_output_dim)

    def forward(self, x):
        out = torch.tanh(self.fc_out(x))
        return out


class Encoder(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self.hid_dim = parameters.hid_dim
        self.pro_output_dim = parameters.pro_output_dim
        self.online_features_flag = parameters.online_features_flag
        self.pro_features_flag = parameters.pro_features_flag

        input_dim = 3
        if self.online_features_flag:
            input_dim = input_dim + parameters.online_dim

        self.rnn = nn.GRU(input_dim, self.hid_dim)
        self.dropout = nn.Dropout(parameters.dropout)

        if self.pro_features_flag:
            self.extra = Extra_MLP(parameters)
            self.fc_hid = nn.Linear(self.hid_dim + self.pro_output_dim, self.hid_dim)

    def forward(self, src, src_len, pro_features):
        # src = [src len, batch size, 3]
        # if only input trajectory, input dim = 2; elif input trajectory + behavior feature, input dim = 2 + n
        # src_len = [batch size]

        packed_embedded = nn.utils.rnn.pack_padded_sequence(src, src_len)
        packed_outputs, hidden = self.rnn(packed_embedded)

        # packed_outputs is a packed sequence containing all hidden states
        # hidden is now from the final non-padded element in the batch

        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)
        # outputs is now a non-packed sequence, all hidden states obtained
        #  when the input is a pad token are all zeros

        # outputs = [src len, batch size, hid dim * num directions]
        # hidden = [n layers * num directions, batch size, hid dim]

        # initial decoder hidden is final hidden state of the forwards and backwards
        #  encoder RNNs fed through a linear layer

        # hidden = [1, batch size, hidden_dim]
        # outputs = [src len, batch size, hidden_dim * num directions]
            
        if self.pro_features_flag:
            extra_emb = self.extra(pro_features)
            extra_emb = extra_emb.unsqueeze(0)
            # extra_emb = [1, batch size, extra output dim]
            hidden = torch.tanh(self.fc_hid(torch.cat((extra_emb, hidden), dim=2)))
            # hidden = [1, batch size, hid dim]

        return outputs, hidden


class Attention(nn.Module):
    # TODO update to more advanced attention layer.
    def __init__(self, parameters):
        super().__init__()
        self.hid_dim = parameters.hid_dim

        self.attn = nn.Linear(self.hid_dim * 2, self.hid_dim)
        self.v = nn.Linear(self.hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, attn_mask):
        # hidden = [1, bath size, hid dim]
        # encoder_outputs = [src len, batch size, hid dim * num directions]
        src_len = encoder_outputs.shape[0]
        # repeat decoder hidden sate src_len times
        hidden = hidden.repeat(src_len, 1, 1)
        hidden = hidden.permute(1, 0, 2)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # hidden = [batch size, src len, hid dim]
        # encoder_outputs = [batch size, src len, hid dim * num directions]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy = [batch size, src len, hid dim]

        attention = self.v(energy).squeeze(2)
        # attention = [batch size, src len]
        attention = attention.masked_fill(attn_mask == 0, -1e10)
        # using mask to force the attention to only be over non-padding elements.

        return F.softmax(attention, dim=1)


class DecoderMulti(nn.Module):
    def __init__(self, parameters):
        super().__init__()

        self.id_size = parameters.id_size
        self.id_emb_dim = parameters.id_emb_dim
        self.hid_dim = parameters.hid_dim
        self.pro_output_dim = parameters.pro_output_dim
        self.online_dim = parameters.online_dim
        self.rid_fea_dim = parameters.rid_fea_dim

        self.attn_flag = parameters.attn_flag
        self.dis_prob_mask_flag = parameters.dis_prob_mask_flag  # final softmax
        self.online_features_flag = parameters.online_features_flag
        self.tandem_fea_flag = parameters.tandem_fea_flag

        self.emb_id = nn.Embedding(self.id_size, self.id_emb_dim)
        
        rnn_input_dim = self.id_emb_dim + 1
        fc_id_out_input_dim = self.hid_dim
        fc_rate_out_input_dim = self.hid_dim
        
        type_input_dim = self.id_emb_dim + self.hid_dim
        self.tandem_fc = nn.Sequential(
                          nn.Linear(type_input_dim, self.hid_dim),
                          nn.ReLU()
                          )
        
        if self.attn_flag:
            self.attn = Attention(parameters)
            rnn_input_dim = rnn_input_dim + self.hid_dim 

        if self.online_features_flag:
            rnn_input_dim = rnn_input_dim + self.online_dim  # 5 poi and 5 road network
            
        if self.tandem_fea_flag:
            fc_rate_out_input_dim = self.hid_dim + self.rid_fea_dim
            
        self.rnn = nn.GRU(rnn_input_dim, self.hid_dim)
        self.fc_id_out = nn.Linear(fc_id_out_input_dim, self.id_size)
        self.fc_rate_out = nn.Linear(fc_rate_out_input_dim, 1)
        self.dropout = nn.Dropout(parameters.dropout)
        
        
    def forward(self, input_id, input_rate, hidden, encoder_outputs, attn_mask,
                pre_grid, next_grid, constraint_vec, pro_features, online_features, rid_features):

        # input_id = [batch size, 1] rid long
        # input_rate = [batch size, 1] rate float. 
        # hidden = [1, batch size, hid dim]
        # encoder_outputs = [src len, batch size, hid dim * num directions]
        # attn_mask = [batch size, src len]
        # pre_grid = [batch size, 3]
        # next_grid = [batch size, 3]
        # constraint_vec = [batch size, id_size], [id_size] is the vector of reachable rid
        # pro_features = [batch size, profile features input dim]
        # online_features = [batch size, online features dim]
        # rid_features = [batch size, rid features dim]

        input_id = input_id.squeeze(1).unsqueeze(0)  # cannot use squeeze() bug for batch size = 1
        # input_id = [1, batch size]
        input_rate = input_rate.unsqueeze(0)
        # input_rate = [1, batch size, 1]
        embedded = self.dropout(self.emb_id(input_id))
        # embedded = [1, batch size, emb dim]

        if self.attn_flag:
            a = self.attn(hidden, encoder_outputs, attn_mask)
            # a = [batch size, src len]
            a = a.unsqueeze(1)
            # a = [batch size, 1, src len]
            encoder_outputs = encoder_outputs.permute(1, 0, 2)
            # encoder_outputs = [batch size, src len, hid dim * num directions]
            weighted = torch.bmm(a, encoder_outputs)
            # weighted = [batch size, 1, hid dim * num directions]
            weighted = weighted.permute(1, 0, 2)
            # weighted = [1, batch size, hid dim * num directions]

            if self.online_features_flag:
                rnn_input = torch.cat((weighted, embedded, input_rate, 
                                       online_features.unsqueeze(0)), dim=2)
            else:
                rnn_input = torch.cat((weighted, embedded, input_rate), dim=2)
        else:
            if self.online_features_flag:
                rnn_input = torch.cat((embedded, input_rate, online_features.unsqueeze(0)), dim=2)
            else:
                rnn_input = torch.cat((embedded, input_rate), dim=2)

        output, hidden = self.rnn(rnn_input, hidden)
        
        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # seq len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        assert (output == hidden).all()

        # pre_rid
        if self.dis_prob_mask_flag:
            prediction_id = mask_log_softmax(self.fc_id_out(output.squeeze(0)), 
                                             constraint_vec, log_flag=True)
        else:
            prediction_id = F.log_softmax(self.fc_id_out(output.squeeze(0)), dim=1)
            # then the loss function should change to nll_loss()

        # pre_rate
        max_id = prediction_id.argmax(dim=1).long()
        id_emb = self.dropout(self.emb_id(max_id))
        rate_input = torch.cat((id_emb, hidden.squeeze(0)),dim=1)
        rate_input = self.tandem_fc(rate_input)  # [batch size, hid dim]
        if self.tandem_fea_flag:
            prediction_rate = torch.sigmoid(self.fc_rate_out(torch.cat((rate_input, rid_features), dim=1)))
        else:
            prediction_rate = torch.sigmoid(self.fc_rate_out(rate_input))

        # prediction_id = [batch size, id_size]
        # prediction_rate = [batch size, 1]

        return prediction_id, prediction_rate, hidden

class Seq2SeqMulti(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder  # Encoder
        self.decoder = decoder  # DecoderMulti
        self.device = device

    def forward(self, src, src_len, trg_id, trg_rate, trg_len,
                pre_grids, next_grids, constraint_mat, pro_features, 
                online_features_dict, rid_features_dict,
                teacher_forcing_ratio=0.5):
        """
        src = [src len, batch size, 3], x,y,t
        src_len = [batch size]
        trg_id = [trg len, batch size, 1]
        trg_rate = [trg len, batch size, 1]
        trg_len = [batch size]
        pre_grids = [trg len, batch size, 3]
        nex_grids = [trg len, batch size, 3]
        constraint_mat = [trg len, batch size, id_size]
        pro_features = [batch size, profile features input dim]
        online_features_dict = {rid: online_features} # rid --> grid --> online features
        rid_features_dict = {rid: rn_features}
        teacher_forcing_ratio is probability to use teacher forcing
        e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        Return:
        ------
        outputs_id: [seq len, batch size, id_size(1)] based on beam search
        outputs_rate: [seq len, batch size, 1]
        """
        max_trg_len = trg_id.size(0)
        batch_size = trg_id.size(1)

        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hiddens = self.encoder(src, src_len, pro_features)

        if self.decoder.attn_flag:
            attn_mask = torch.zeros(batch_size, max(src_len))  # only attend on unpadded sequence
            for i in range(len(src_len)):
                attn_mask[i][:src_len[i]] = 1.
            attn_mask = attn_mask.to(self.device)
        else:
            attn_mask = None

        outputs_id, outputs_rate = self.normal_step(max_trg_len, batch_size, trg_id, trg_rate, trg_len,
                                                    encoder_outputs, hiddens, attn_mask,
                                                    online_features_dict,
                                                    rid_features_dict,
                                                    pre_grids, next_grids, constraint_mat, pro_features,
                                                    teacher_forcing_ratio)

        return outputs_id, outputs_rate

    def normal_step(self, max_trg_len, batch_size, trg_id, trg_rate, trg_len, encoder_outputs, hidden,
                    attn_mask, online_features_dict, rid_features_dict,
                    pre_grids, next_grids, constraint_mat, pro_features, teacher_forcing_ratio):
        """
        Returns:
        -------
        outputs_id: [seq len, batch size, id size]
        outputs_rate: [seq len, batch size, 1]
        """
        # tensor to store decoder outputs
        outputs_id = torch.zeros(max_trg_len, batch_size, self.decoder.id_size).to(self.device)
        outputs_rate = torch.zeros(trg_rate.size()).to(self.device)

        # first input to the decoder is the <sos> tokens
        input_id = trg_id[0, :]
        input_rate = trg_rate[0, :]
        for t in range(1, max_trg_len):
            # insert input token embedding, previous hidden state, all encoder hidden states
            #  and attn_mask
            # receive output tensor (predictions) and new hidden state
            if self.decoder.online_features_flag:
                online_features = get_dict_info_batch(input_id, online_features_dict).to(self.device)
            else:
                online_features = torch.zeros((1, batch_size, self.decoder.online_dim))
            if self.decoder.tandem_fea_flag:
                rid_features = get_dict_info_batch(input_id, rid_features_dict).to(self.device)
            else:
                rid_features = None
            prediction_id, prediction_rate, hidden = self.decoder(input_id, input_rate, hidden, encoder_outputs,
                                                                     attn_mask, pre_grids[t], next_grids[t],
                                                                     constraint_mat[t], pro_features, online_features,
                                                                     rid_features)

            # place predictions in a tensor holding predictions for each token
            outputs_id[t] = prediction_id
            outputs_rate[t] = prediction_rate

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1_id = prediction_id.argmax(1)
            top1_id = top1_id.unsqueeze(-1)  # make sure the output has the same dimension as input

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input_id = trg_id[t] if teacher_force else top1_id
            input_rate = trg_rate[t] if teacher_force else prediction_rate

        # max_trg_len, batch_size, trg_rid_size
        outputs_id = outputs_id.permute(1, 0, 2)  # batch size, seq len, rid size
        outputs_rate = outputs_rate.permute(1, 0, 2)  # batch size, seq len, 1
        for i in range(batch_size):
            outputs_id[i][trg_len[i]:] = 0
            outputs_id[i][trg_len[i]:, 0] = 1  # make sure argmax will return eid0
            outputs_rate[i][trg_len[i]:] = 0
        outputs_id = outputs_id.permute(1, 0, 2)
        outputs_rate = outputs_rate.permute(1, 0, 2)

        return outputs_id, outputs_rate

