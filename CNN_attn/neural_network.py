# -*- coding: utf-8 -*-
"""
Spyder Editor

This file stores the neural network class
"""
import torch.nn as nn
import torch
import metrics
import math

# loss function
bceloss = nn.BCELoss()

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        #self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            old x: Tensor, shape [seq_len, batch_size, embedding_dim]
            swawp to: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:x.size(1)].swapaxes(0, 1)
        return x

class NeuralNetwork(nn.Module):
    def __init__(self, nlayers_conv, h, f, ps, fcs, p, mha_p):
        super(NeuralNetwork, self).__init__()

        # define convolutional layers
        layers = []

        for i in range(nlayers_conv):
            if len(layers) == 0:
                layers.append(nn.Conv1d(4, h, f))
                layers.append(nn.BatchNorm1d(h))
                layers.append(nn.ReLU())
                layers.append(nn.MaxPool1d(ps))

            else:
                layers.append(nn.Conv1d(h, h, f))
                layers.append(nn.BatchNorm1d(h))
                layers.append(nn.ReLU())
                layers.append(nn.MaxPool1d(ps))
        self.convlayers = nn.Sequential(*layers)
        '''positional encoding for MHA'''
        self.positionalencoding = PositionalEncoding(int(h))
        '''MHA'''
        self.multihead_attn = nn.MultiheadAttention(h, 10, dropout=mha_p, batch_first=True)
        fc_layers_head = [nn.Flatten(),
                          nn.LazyLinear(fcs),
                          nn.Dropout(p=p),
                          nn.ReLU(),
                          nn.Linear(fcs, 1),
                          nn.Sigmoid()]

        fc_layers_testis = [nn.Flatten(),
                            nn.LazyLinear(fcs),
                            nn.Dropout(p=p),
                            nn.ReLU(),
                            nn.Linear(fcs, 1),
                            nn.Sigmoid()]

        self.fclayers_head = nn.Sequential(*fc_layers_head)
        self.fclayers_testis = nn.Sequential(*fc_layers_testis)
        self.attn_weights = None

    def forward(self, x):
        # conv
        convl = self.convlayers(x)
        # change of dimension to [batch, length, embedding]
        convl_pe = self.positionalencoding(convl.swapaxes(1, 2))
        attn_output, self.attn_weights = self.multihead_attn(convl_pe, convl_pe, convl_pe)
        # tissue spec
        head_out = self.fclayers_head(attn_output).t()[0]
        testis_out = self.fclayers_testis(attn_output).t()[0]

        return head_out, testis_out

    def training_step(self, batch):
        x, y = batch
        head_out, testis_out = self(x)  # Generate predictions
        loss_head = bceloss(head_out, y[:, 0])  # Calculate loss
        loss_testis = bceloss(testis_out, y[:, 1])
        return loss_head + loss_testis

    def validation_step(self, batch):
        with torch.no_grad():
            x, y = batch
            head_out, testis_out = self(x)  # Generate predictions
            return {'head_out': head_out,
                    'head_y': y[:, 0],
                    'testis_out': testis_out,
                    'testis_y': y[:, 1]}

    def validation_epoch_end(self, outputs):
        tot_outputhead = torch.cat([x['head_out'] for x in outputs])
        tot_yhead = torch.cat([x['head_y'] for x in outputs])
        tot_outputtestis = torch.cat([x['testis_out'] for x in outputs])
        tot_ytestis = torch.cat([x['testis_y'] for x in outputs])
        loss = bceloss(tot_outputhead, tot_yhead).detach() + bceloss(tot_outputtestis, tot_ytestis).detach()
        auc_head = metrics.auc(tot_outputhead.cpu().detach().numpy(), tot_yhead.cpu().detach().numpy())
        auc_testis = metrics.auc(tot_outputtestis.cpu().detach().numpy(), tot_ytestis.cpu().detach().numpy())
        print("val_loss: {:.4f}, head_auc: {:.4f}, testis_auc: {:.4f}".
              format(loss.item(), auc_head.item(), auc_testis.item()))
        return {"loss": loss.detach().item(), "head_auc": auc_head.item(), "testis_auc": auc_testis.item()}
        # return {'val_loss': epoch_loss.item(), 'val_cor': epoch_acc.item()}

    # def epoch_end(self, epoch, result):
    #    print("Epoch [{}], val_loss: {:.4f}, val_auc: {:.4f}".format(epoch, result['val_loss'], result['val_auc']))