# -*- coding: utf-8 -*-

import math, copy, time
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Function
import _pickle as cPickle
import evaluate as eva
from sparsemax import Sparsemax
import torch.autograd as autograd
from timeit import default_timer as timer


parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='client')
parser.add_argument('--port', type=str, default='65053')
parser.add_argument('--path', type=str, default='/Desktop/DeepWmaxSAT/')
parser.add_argument('--epoch', type=int, default=40)
parser.add_argument('--preepoch', type=int, default=100)
parser.add_argument('--ini_r', type=float, default=1.)

args = parser.parse_args()
opt = vars(args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    """
    Encoder is a multi-layer transformer with BiGRU
    """

    def __init__(self, encoder, classifier, src_embed, pos_emb, dropout):
        super(Model, self).__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.src_embed = src_embed
        self.pos_emb = pos_emb
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, pos_ind, src_mask, entity=False, rel=False):
        memory, attn = self.encode(src, src_mask, pos_ind)
        output_chunk, output_batch = self.decode(memory, src_mask, entity, rel)
        return output_chunk, output_batch, attn

    def encode(self, src, src_mask, pos_ind):
        X = torch.cat((self.dropout(self.src_embed(src)), self.pos_emb[pos_ind]), dim=2)
        return self.encoder(X, src_mask)

    def decode(self, memory, src_mask, entity, rel):
        return self.classifier(memory, src_mask, entity, rel)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N, d_in, d_h):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.gru = nn.GRU(d_in, d_h, batch_first=True, dropout=0.1, bidirectional=True)

    def forward(self, X, mask):
        x, _ = self.gru(X)
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
            attn = layer.self_attn.attn
        return x, attn


class LayerNorm(nn.Module):
    "Construct a layernorm module."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        # return x + self.dropout(sublayer(self.norm(x)))
        return x + sublayer(x)


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        # d_k is the output dimension for each head
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab, emb):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model, _weight=emb)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x)


class PositionalEncodingEmb(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=100):
        super(PositionalEncodingEmb, self).__init__()
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)
        self.position = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(max_len + 1, d_model)), requires_grad=True)

    def forward(self, x):
        if x.size(1) < self.max_len:
            p = self.position[torch.arange(x.size(1))]
            p = p.unsqueeze(0)
            p = p.expand(x.size(0), p.size(1), p.size(2))
        else:
            i = torch.cat((torch.arange(self.max_len), -1 * torch.ones((x.size(1) - self.max_len), dtype=torch.long)))
            p = self.position[i]
            p = p.unsqueeze(0)
            p = p.expand(x.size(0), p.size(1), p.size(2))
        return self.dropout(x + p)


# The decoder is also composed of a stack of N=6 identical layers.
class Classifier(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, sch_k, d_in, d_h, d_e, h, dchunk_out, drel_out, vocab, dropout):
        super(Classifier, self).__init__()
        self.sch_k = sch_k
        self.vocab = vocab
        self.d_in = d_in
        self.d_h = d_h
        self.h = h
        self.dchunk_out = dchunk_out
        self.lstm = nn.LSTM(3 * d_in, d_h, batch_first=True, dropout=0.5, bidirectional=True)
        self.gru = nn.GRU(3 * d_in, d_h, batch_first=True, dropout=0.1, bidirectional=True)
        self.chunk_out = nn.Linear(2 * d_h + d_e, dchunk_out)
        self.softmax = nn.Softmax(dim=1)
        self.pad = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(1, d_in)), requires_grad=True)
        self.label_emb = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(dchunk_out + 1, d_e)), requires_grad=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, src_mask, entity=False, rel=False):

        # transformer with label GRU
        X = self.dropout(X)
        X_pad = torch.cat((self.pad.repeat(X.size(0), 1, 1), X, self.pad.repeat(X.size(0), 1, 1)), dim=1)
        l = X_pad[:, :-2, :]
        m = X_pad[:, 1:-1, :]
        r = X_pad[:, 2:, :]

        # transformer with Elman GRU
        output_h, _ = self.gru(torch.cat((l, m, r), dim=2))
        # output_h, _ = self.gru(X)
        # concatenate label embedding
        output_batch = torch.Tensor().to(device)
        output_h_ = output_h.permute(1, 0, 2)
        hi = torch.cat((output_h_[0, :, :], self.label_emb[-1].repeat(output_h_.size(1), 1)), dim=1)
        output_chunki = self.softmax(self.chunk_out(hi)).view(hi.size(0), 1, -1)
        output_batch = torch.cat((output_batch, output_chunki), dim=1)
        chunki = torch.argmax(self.softmax(self.chunk_out(hi)), dim=1)
        chunk_batch = torch.argmax(self.softmax(self.chunk_out(hi)), dim=1).view(-1, 1)
        for h in output_h_[1:, :, :]:
            hi = torch.cat((h, self.label_emb[chunki]), dim=1)
            output_chunki = self.softmax(self.chunk_out(hi)).view(hi.size(0), 1, -1)
            output_batch = torch.cat((output_batch, output_chunki), dim=1)
            chunki = torch.argmax(self.softmax(self.chunk_out(hi)), dim=1)
            chunk_batch = torch.cat((chunk_batch, chunki.view(-1, 1)), dim=1)
        output_chunk = output_batch.view(-1, self.dchunk_out)

        #if entity and not rel:
        return output_chunk, output_batch


def make_model(vocab, pos, emb, sch_k=1.0, N=2, d_in=350, d_h=100, d_e=25, d_p=50, dchunk_out=5,
               drel_out=2, d_model=200, d_emb=300, d_ff=200, h=10, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    pos_emb = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(pos, d_p)), requires_grad=True)
    position = PositionalEncodingEmb(d_emb, dropout)
    model = Model(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N, d_emb+d_p, d_h),
        Classifier(sch_k, d_model, d_h, d_e, h, dchunk_out, drel_out, vocab, 0.1),
        nn.Sequential(Embeddings(d_emb, vocab, emb), c(position)),
        pos_emb, dropout)

    return model


class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, src, pos,  parse, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        self.trg = trg
        self.pos = pos
        self.parse = parse
        self.ntokens = torch.tensor(self.src != pad, dtype=torch.float).data.sum()


def data_batch(idxs_src, labels_src, pos_src, parse_src, batchlen):
    assert len(idxs_src) == len(labels_src)
    batches = [idxs_src[x: x + batchlen] for x in range(0, len(idxs_src), batchlen)]
    label_batches = [labels_src[x: x + batchlen] for x in range(0, len(labels_src), batchlen)]
    pos_batches = [pos_src[x: x + batchlen] for x in range(0, len(idxs_src), batchlen)]
    parse_batches = [parse_src[x: x + batchlen] for x in range(0, len(idxs_src), batchlen)]
    for batch, label, pos, parse in zip(batches, label_batches, pos_batches, parse_batches):
        # compute length of longest sentence in batch
        batch_max_len = max([len(s) for s in batch])
        batch_data = 0 * np.ones((len(batch), batch_max_len))
        batch_labels = 0 * np.ones((len(batch), batch_max_len))
        batch_pos = 0 * np.ones((len(batch), batch_max_len))
        batch_parse = parse

        # copy the data to the numpy array
        for j in range(len(batch)):
            cur_len = len(batch[j])
            batch_data[j][:cur_len] = batch[j]
            batch_labels[j][:cur_len] = label[j]
            batch_pos[j][:cur_len] = pos[j]

        # since all data are indices, we convert them to torch LongTensors
        batch_data, batch_labels, batch_pos = torch.LongTensor(batch_data), torch.LongTensor(batch_labels), torch.LongTensor(batch_pos)
        # convert Tensors to Variables
        yield Batch(batch_data, batch_pos, batch_parse, batch_labels, 0)


class MixingFunc(Function):
    # Apply the Mixing method to the input probabilities.

    @staticmethod
    def forward(ctx, in_S, in_z, in_is_input, in_rule_idx, max_iter, eps, prox_lam):
        B = in_z.size(0)
        n, m = in_S.size(0), in_S.size(1)
        ctx.max_iter, ctx.eps, ctx.prox_lam = max_iter, eps, prox_lam

        ctx.z = torch.zeros(B, n, device=device)
        ctx.S = torch.zeros(n, m, device=device)
        ctx.is_input = torch.zeros(B, n, dtype=torch.float, device=device)
        ctx.rule_idx = torch.zeros(B, m, m, dtype=torch.float, device=device)

        ctx.z[:] = in_z.data
        ctx.S[:] = in_S.data
        ctx.is_input[:] = in_is_input.data
        ctx.rule_idx[:] = in_rule_idx.data

        ctx.g = torch.zeros(B, n, k, device=device)
        ctx.niter = torch.zeros(B, dtype=torch.int, device=device)
        ctx.v_t = F.normalize(torch.randn(k, 1, device=device, dtype=torch.float).normal_().view(1, -1))
        ctx.V = torch.zeros(B, n, k, device=device)

        count = 0
        for z_b, is_input_b, rule_idx_b, g_b in zip(ctx.z, ctx.is_input, ctx.rule_idx, ctx.g):
            z_b = z_b.unsqueeze(1).to(device)
            new_z = z_b.clone().to(device)
            #v_rand = F.normalize(torch.randn(k, 1, device=device, dtype=torch.float).normal_().view(1, -1))
            ctx.V[count] = (- torch.cos(pi * new_z)) * ctx.v_t + torch.sin(pi * new_z) * \
                           (F.normalize(torch.randn(k, 1, device=device, dtype=torch.float).normal_().view(1, -1))\
                            .mm(torch.diag(torch.ones(k, device=device)) - ctx.v_t.t().mm(ctx.v_t))).to(device)
            ctx.V[count] = torch.mul(ctx.V[count].float(), is_input_b.unsqueeze(1).float()).to(device).float()
            # init vo with random vector
            for i in (is_input_b.squeeze() == 0).nonzero():
                #v_rand = F.normalize(torch.randn(k, 1, device=device, dtype=torch.float).normal_().view(1, -1))
                i = i.squeeze()
                ctx.V[count] = ctx.V[count].index_add_(0, i, F.normalize(torch.randn(k, 1, device=device, dtype=torch.float).normal_().view(1, -1)))

            masked_S_t = rule_idx_b.mm(ctx.S.t()).to(device)
            new_S = masked_S_t[masked_S_t.sum(dim=1) != 0].t().to(device)
            new_g_b = g_b.clone().to(device)
            if new_S.size(1) != 0:
                omega = new_S.t().mm(ctx.V[count]).to(device)

                # while not converged: forward pass coord descent
                for it in range(max_iter):
                    early_stop = 0
                    for i in (is_input_b.squeeze() == 0).nonzero():
                        i = i.squeeze()
                        new_g_b[i] = new_S[i].unsqueeze(0).mm(omega).squeeze() - new_S[i].norm().pow(2) * ctx.V[count][i]
                        v_pre = ctx.V[count][i].clone().to(device)
                        if new_g_b[i].norm()!=0:
                            ctx.V[count][i] = - new_g_b[i] / new_g_b[i].norm()
                        omega += new_S[i].unsqueeze(0).t().mm((ctx.V[count][i] - v_pre).unsqueeze(0))
                # output V: vo --> zo
                for i in (is_input_b.squeeze() == 0).nonzero():
                    i = i.squeeze()
                    cos = - ctx.v_t.mm(ctx.V[count][i].unsqueeze(0).t()).squeeze().to(device)
                    cos = torch.clamp(cos, -1 + ctx.eps, 1 - ctx.eps)
                    new_z[i] = torch.acos(cos) / pi
            ctx.V[count] = F.normalize(ctx.V[count])
            ctx.z[count] = new_z.squeeze()
            ctx.g[count] = new_g_b.clone()
            count += 1
        return ctx.z.clone()

    @staticmethod
    def backward(ctx, dz):
        if torch.any(dz != dz):
            res = torch.zeros_like(dz)
        else:
            #dz[dz!=dz]=0
            dz.clamp(0,1)
            B = dz.size(0)
            ctx.U = torch.zeros(B, n, k, device=device)
            ctx.dz = torch.zeros(B, n, device=device)
            ctx.dz[:] = dz.data
            #print("in dz:", ctx.dz)
            ctx.dV = torch.zeros(B, n, k, device=device)

            dz_new = ctx.dz.clone().to(device)
            count = 0
            for dz_b, is_input_b, dV_b, U_b, z_b, rule_idx_b, g_b in zip(dz_new, ctx.is_input, ctx.dV, ctx.U, ctx.z, ctx.rule_idx, ctx.g):
                # dzo -> dvo
                #print("dz_b_0", dz_b)
                for i in (is_input_b.squeeze() == 0).nonzero():
                    i = i.squeeze()
                    dV_b[i] = (dz_b[i] / pi / (torch.sin(pi * z_b[i]))) * ctx.v_t

                masked_S_t = rule_idx_b.mm(ctx.S.t()).to(device)
                new_S = masked_S_t[masked_S_t.sum(dim=1) != 0].t().to(device)
                #print("new_S:", new_S)
                if new_S.size(1) != 0:
                    Phi_b = (new_S.t()).mm(U_b).to(device)

                    # while not converged: backward pass coord descent
                    for it in range(ctx.max_iter):
                        #early_stop = 0
                        for i in (is_input_b.squeeze() == 0).nonzero():
                            i = i.squeeze()
                            dg = new_S[i].unsqueeze(0).mm(Phi_b).squeeze() - new_S[i].norm().pow(2) * U_b[i] - dV_b[i]
                            P = torch.diag(torch.ones(k, device=device)) - ctx.V[count][i].unsqueeze(0).t().mm(ctx.V[count][i].unsqueeze(0))
                            u_pre = U_b[i].clone().to(device)
                            if g_b[i].norm()!=0:
                                U_b[i] = - (dg / g_b[i].norm()).unsqueeze(0).mm(P).to(device)
                            #else:
                                #U_b[i] = u_pre
                            Phi_b += new_S[i].unsqueeze(0).t().mm((U_b[i] - u_pre).unsqueeze(0)).to(device)

                    sum_uws = torch.zeros(new_S.size(1), k, device=device)
                    for i in (is_input_b.squeeze() == 0).nonzero():
                        i = i.squeeze()
                        sum_ums_step = new_S[i].unsqueeze(0).t().mm(U_b[i].unsqueeze(0))
                        sum_uws += sum_ums_step

                    v_rand = F.normalize(torch.randn(k, 1, device=device, dtype=torch.float).normal_().view(1, -1))
                    dv_dz = pi * (torch.sin(pi * z_b).unsqueeze(1) * ctx.v_t + torch.cos(pi * z_b).unsqueeze(1) *
                                  (v_rand.mm(torch.diag(torch.ones(k, device=device)) - ctx.v_t.t().mm(ctx.v_t))))
                    dz_input = torch.diagonal(new_S.mm(sum_uws).mm(dv_dz.t()))

                    dz_new[count] = torch.mul(dz_input.squeeze(), is_input_b) + torch.mul(dz_b, (1 - is_input_b))
                    #ctx.dz[count][0] = 0
                    ctx.dz[count][11:] = 0
                count += 1
            res = ctx.dz.clone()
            #print("out dz:", ctx.dz.clone())

        return None, res, None, None, None, None, None


class RuleAttention(nn.Module):
    """ Applies attention mechanism on the `context` -- S using the `query` -- z.

    Args:
        dimensions: #var

        attention_type (str, optional): How to compute the attention score:
            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`

    Example:
         attention = Attention(#var)
         query = torch.randn(#batch, 1, #var)
         context = torch.randn(#batch, #rule, #var)
         attn_scores = attention(query, context)
    """

    def __init__(self, dimensions, attention_type='dot', max_type='sparse'):
        super(RuleAttention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')
        if max_type not in ['sparse', 'soft']:
            raise ValueError('Invalid max type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False).to(device)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False).to(device)
        self.max_type = max_type
        if self.max_type == 'sparse':
            self.softmax = Sparsemax(dim=-1)
        if self.max_type == 'soft':
            self.softmax = nn.Softmax(dim=-1)
        self.tanh = torch.nn.Tanh().cuda()

    def forward(self, query, context):
        """
        Args:
            query (:class:`torch.FloatTensor` [#batch, 1, #var]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [#batch, #rule, #var]): Data
                over which to apply the attention mechanism.

        Returns:
            * **scores** (:class:`torch.FloatTensor`  [#batch, 1, #rules]):
              Tensor containing attention scores.
        """
        batch_size, output_len, dimensions = query.size() # [#batch, 1, #var]
        query_len = context.size(1) # [#rule]

        if self.attention_type == "general":
            query = query.reshape(batch_size * output_len, dimensions).to(device)
            query = self.linear_in(query).to(device)
            query = query.reshape(batch_size, output_len, dimensions).to(device)

        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous()).to(device)

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, query_len).to(device)
        attention_weights = self.softmax(attention_scores).to(device)
        attention_weights = attention_weights.view(batch_size, output_len, query_len).to(device)

        # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        mix = torch.bmm(attention_weights, context).to(device)

        # concat -> (batch_size * output_len, 2*dimensions)
        combined = torch.cat((mix, query), dim=2).to(device)
        combined = combined.view(batch_size * output_len, 2 * dimensions).to(device)

        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        output = self.linear_out(combined.to(device)).view(batch_size, output_len, dimensions).to(device)
        output = self.tanh(output).to(device)

        return attention_weights.view(batch_size * output_len, query_len)


class MaxsatLayer(nn.Module):
    """Apply a MaxSAT layer to complete the input probabilities.

    Args:
        aux: Number of auxiliary variables.
        max_iter: Maximum number of iterations for solving
            the inner optimization problem.
            Default: 40
        eps: The stopping threshold for the inner optimizaiton problem.
            The inner Mixing method will stop when the function decrease
            is less then eps times the initial function decrease.
            Default: 1e-4
        prox_lam: The diagonal increment in the backward linear system
            to make the backward pass more stable.
            Default: 1e-2
        weight_normalize: Set true to perform normlization for init weights.
            Default: True

    Inputs: (z, is_input)
        **z** of shape `(batch, n)`:
            Float tensor containing the probabilities (must be in [0,1]).
        **is_input** of shape `(batch, n)`:
            Int tensor indicating which **z** is a input.

    Outputs: shape same as "output_chunk"

    """

    def __init__(self, model, in_S, aux=0, max_iter=80, eps=1e-4, prox_lam=1e-2, weight_normalize=True):
        super(MaxsatLayer, self).__init__()
        self.model = model
        self.S = in_S
        self.n, self.m, self.k = in_S.size(0), in_S.size(1), 32
        self.thre = nn.Parameter(torch.tensor(0.6, device=device, requires_grad=True))
        self.r = nn.Parameter(torch.tensor(opt['ini_r'], device=device, requires_grad=True))
        self.w = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(1, self.m)), requires_grad=True)
        self.aux, self.max_iter, self.eps, self.prox_lam = aux, max_iter, eps, prox_lam

    def forward(self, src, pos_ind, src_mask, parse, entity=True, rel=False):
        trans_out, output_batch, _ = self.model(src, pos_ind, src_mask, entity, rel)
        label = torch.argmax(output_batch, dim=2)
        new_output_batch = output_batch.clone()
        z = []
        is_input = []
        for sentence, out, tag, parse_sen in zip(src, output_batch, label, parse):
            for [x, x_pos, y, y_pos, r] in parse_sen:
                if (tag[y] in [1, 2] and y_pos in ['PROPN', 'NOUN'] and x_pos in ['PROPN', 'NOUN'] and r in ['nn'] and (x - y == 1)) or \
                    (tag[y] in [1, 2] and x_pos == 'ADJ' and y_pos in ['PROPN', 'NOUN'] and r in ['nsubj']) or \
                        (tag[y] in [1, 2] and y_pos == 'ADJ' and x_pos in ['PROPN', 'NOUN'] and r in ['amod'] and (x - y == 1)):
                    z_1 = out[x].to(device)
                    z_2 = out[y].to(device)
                    z_3 = torch.tensor([x_pos == 'ADJ', x_pos in ['PROPN', 'NOUN'], y_pos == 'ADJ', y_pos in ['PROPN', 'NOUN']]).float().to(device)
                    z_4 = torch.tensor([tag[x-1] in[1,2], tag[x-1] not in[1,2], tag[x-1] in[3,4], tag[x-1] not in[3,4]]).float().to(device)
                    z_5 = torch.tensor([r == 'nn', r == 'conj', r == 'nsubj', r == 'amod']).float().to(device)
                    z.append(torch.cat((z_1, z_2, z_4, z_3, z_5), 0))
                    is_input.append(torch.cat([torch.zeros(5).to(device), torch.ones(n-6).to(device)]))
                if (tag[x] in [3, 4] and y_pos == 'ADJ' and x_pos == 'ADJ' and r in ['conj']) or \
                    (tag[x] in [1, 2] and y_pos in ['PROPN', 'NOUN'] and x_pos in ['PROPN', 'NOUN'] and r in ['conj']):
                    z_1 = out[x].to(device)
                    z_2 = out[y].to(device)
                    z_3 = torch.tensor([x_pos == 'ADJ', x_pos in ['PROPN', 'NOUN'], y_pos == 'ADJ', y_pos in ['PROPN', 'NOUN']]).float().to(device)
                    z_4 = torch.tensor([tag[y-1] in[1,2], tag[y-1] not in[1,2], tag[y-1] in[3,4], tag[y-1] not in[3,4]]).float().to(device)
                    z_5 = torch.tensor([r == 'nn', r == 'conj', r == 'nsubj', r == 'amod']).float().to(device)
                    z.append(torch.cat((z_1, z_2, z_4, z_3, z_5), 0))
                    is_input.append(torch.cat([torch.ones(5).to(device), torch.zeros(5).to(device), torch.ones(n-11).to(device)]))

        if z != []:
            z = torch.stack(z).to(device)
            is_input = torch.stack(is_input).to(device)

            if (z!=z).any():
                print("warning!")
            else:
                B = z.size(0)
                z = torch.cat([torch.ones(B, 1, device=device), z, torch.zeros(B, 0, device=device)], dim=1)
                is_input = torch.cat([torch.ones(B, 1, device=device), is_input, torch.zeros(B, 0, device=device)], dim=1)
                query = z[:, 1:11].unsqueeze(1).to(device)
                context = torch.abs(S_t_body[:, 1:11]).repeat(B, 1, 1).to(device)
                rule_attention_sparse = RuleAttention(query.size(2), 'dot', 'sparse')
                attn_weights = rule_attention_sparse(query, context).squeeze().to(device)

                if B == 1:
                    attn_weights = attn_weights.unsqueeze(0).to(device)

                rule_idx_vec = torch.zeros_like(attn_weights).to(device)

                for i in range(B):
                    for j in range(self.m):
                        if torch.all(torch.eq(torch.abs(z[:, 15:])[i], torch.abs(S_t[:, 15:])[j])):
                            if torch.all(torch.eq(torch.abs(z[:, 11:13])[i], torch.abs(S_t[:, 11:13])[j])) or \
                                    torch.all(torch.eq(torch.abs(z[:, 13:15])[i], torch.abs(S_t[:, 13:15])[j])) or\
                                    torch.all(torch.eq(torch.zeros_like(torch.abs(S_t[:, 11:15])[j]), torch.abs(S_t[:, 11:15])[j])):
                                rule_idx_vec[i, j] = 1


                rule_idx_vec *= attn_weights
                rule_idx_vec = F.normalize(rule_idx_vec)


                rule_idx = torch.empty(B, self.m, self.m)
                for i in range(B):
                    rule_idx[i] = torch.diag(rule_idx_vec[i])

                z_new = MixingFunc.apply(self.S, z, is_input, rule_idx, self.max_iter, self.eps, self.prox_lam).to(device)

                z_msout = z_new[:, 1:11].to(device)

                count_a = 0
                count_z = 0

                for sentence, out, tag, parse_sen in zip(src, output_batch, label, parse):
                    for [x, x_pos, y, y_pos, r] in parse_sen:
                        if (tag[y] in [1, 2] and y_pos in ['PROPN', 'NOUN'] and x_pos in ['PROPN', 'NOUN'] and r in ['nn'] and (x - y == 1)) or \
                           (tag[y] in [1, 2] and x_pos == 'ADJ' and y_pos in ['PROPN', 'NOUN'] and r in ['nsubj']) or \
                            (tag[y] in [1, 2] and y_pos == 'ADJ' and x_pos in ['PROPN', 'NOUN'] and r in ['amod'] and (x - y == 1)) or \
                            (tag[x] in [3, 4] and y_pos == 'ADJ' and x_pos == 'ADJ' and r in ['conj']) or \
                            (tag[x] in [1, 2] and y_pos in ['PROPN', 'NOUN'] and x_pos in ['PROPN', 'NOUN'] and r in ['conj']):
                                new_output_batch[count_a][x] = z_msout[count_z][0:5]
                                new_output_batch[count_a][y] = z_msout[count_z][5:10]
                                count_z += 1
                    count_a += 1

        features = gate(output_batch, new_output_batch, self.r).to(device)

        return features.clone()


def gate(h,x,r):
    features = (r * h + (1-r)*x).to(device)
    features = features.float()
    return features


def log_sum_exp(vec, m_size):
    """
    calculate log of exp sum
    args:
        vec (batch_size, vanishing_dim, hidden_dim) : input tensor
        m_size : hidden_dim
    return:
        batch_size, hidden_dim
    """
    _, idx = torch.max(vec, 1)  # B * 1 * M
    max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)  # B * M
    return max_score.view(-1, m_size) + torch.log(torch.sum(torch.exp(vec - max_score.expand_as(vec)), 1)).view(-1, m_size)  # B * M


class CRF(nn.Module):
    def __init__(self, tagset_size, gpu):
        super(CRF, self).__init__()
        self.gpu = gpu
        self.tagset_size = tagset_size
        init_transitions = torch.zeros(self.tagset_size+2, self.tagset_size+2)
        init_transitions[:,START_TAG] = -10000.0
        init_transitions[STOP_TAG,:] = -10000.0
        #init_transitions[:,0] = -10000.0
        #init_transitions[0,:] = -10000.0
        if self.gpu:
            init_transitions = init_transitions.cuda()
        self.transitions = nn.Parameter(init_transitions)

    def _calculate_PZ(self, feats, mask):
        """
            input:
                feats: (batch, seq_len, self.tag_size+2)
                masks: (batch, seq_len)
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)
        assert(tag_size == self.tagset_size+2)
        mask = mask.transpose(1,0).contiguous()
        ins_num = seq_len * batch_size
        ## be careful the view shape, it is .view(ins_num, 1, tag_size) but not .view(ins_num, tag_size, 1)
        feats = feats.transpose(1,0).contiguous().view(ins_num,1, tag_size).expand(ins_num, tag_size, tag_size).to(device)
        scores = feats + self.transitions.view(1,tag_size,tag_size).expand(ins_num, tag_size, tag_size).to(device)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size).to(device)
        # build iter
        seq_iter = enumerate(scores)
        _, inivalues = next(seq_iter)  # bat_size * from_target_size * to_target_size
        partition = inivalues[:, START_TAG, :].clone().view(batch_size, tag_size, 1)  # bat_size * to_target_size
        # partition = partition + self.transitions[START_TAG,:].view(1, tag_size, 1).expand(batch_size, tag_size, 1)
        # iter over last scores
        for idx, cur_values in seq_iter:
            # previous to_target is current from_target
            # partition: previous results log(exp(from_target)), #(batch_size * from_target)
            # cur_values: bat_size * from_target * to_target
            cur_values = cur_values + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size).to(device)
            cur_partition = log_sum_exp(cur_values, tag_size).to(device)
            mask_idx = mask[idx, :].view(batch_size, 1).expand(batch_size, tag_size).to(device)
            mask_idx = mask_idx.bool()
            masked_cur_partition = cur_partition.masked_select(mask_idx).to(device)
            mask_idx = mask_idx.contiguous().view(batch_size, tag_size, 1).to(device)
            partition.masked_scatter_(mask_idx, masked_cur_partition).to(device)
        cur_values = self.transitions.view(1,tag_size, tag_size).expand(batch_size, tag_size, tag_size).to(device) + \
                     partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size).to(device)
        cur_partition = log_sum_exp(cur_values, tag_size).to(device)
        final_partition = cur_partition[:, STOP_TAG]
        return final_partition.sum(), scores

    def _viterbi_decode(self, feats, mask):
        """
            input:
                feats: (batch, seq_len, self.tag_size+2)
                mask: (batch, seq_len)
            output:
                decode_idx: (batch, seq_len) decoded sequence
                path_score: (batch, 1) corresponding score for each sequence (to be implementated)
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)
        assert(tag_size == self.tagset_size+2)
        ## calculate sentence length for each sentence
        length_mask = torch.sum(mask.long(), dim = 1).view(batch_size,1).long().to(device)
        ## mask to (seq_len, batch_size)
        mask = mask.transpose(1,0).contiguous()
        ins_num = seq_len * batch_size
        ## be careful the view shape, it is .view(ins_num, 1, tag_size) but not .view(ins_num, tag_size, 1)
        feats = feats.transpose(1,0).contiguous().view(ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size).to(device)
        scores = feats + self.transitions.view(1,tag_size,tag_size).expand(ins_num, tag_size, tag_size).to(device)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size).to(device)

        seq_iter = enumerate(scores)
        back_points = list()
        partition_history = list()
        # mask = 1 + (-1)*mask
        mask = (1 - mask.long()).byte()
        _, inivalues = next(seq_iter)  # bat_size * from_target_size * to_target_size
        partition = inivalues[:, START_TAG, :].clone().view(batch_size, tag_size)  # bat_size * to_target_size
        partition_history.append(partition)
        for idx, cur_values in seq_iter:
            cur_values = cur_values + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
            partition, cur_bp = torch.max(cur_values, 1)
            partition_history.append(partition)
            mask = mask.bool()
            cur_bp.masked_fill_(mask[idx].view(batch_size, 1).expand(batch_size, tag_size), 0).to(device)
            back_points.append(cur_bp)
        partition_history = torch.cat(partition_history, 0).view(seq_len, batch_size, -1).transpose(1,0).contiguous().to(device) ## (batch_size, seq_len. tag_size)
        last_position = length_mask.view(batch_size,1,1).expand(batch_size, 1, tag_size) -1
        last_partition = torch.gather(partition_history, 1, last_position).view(batch_size,tag_size,1).to(device)
        last_values = last_partition.expand(batch_size, tag_size, tag_size).to(device) + self.transitions.view(1,tag_size, tag_size).expand(batch_size, tag_size, tag_size).to(device)
        _, last_bp = torch.max(last_values, 1)
        pad_zero = autograd.Variable(torch.zeros(batch_size, tag_size)).long().to(device)
        #pad_zero = pad_zero.to(device)
        back_points.append(pad_zero.to(device))
        back_points = torch.cat(back_points).view(seq_len, batch_size, tag_size)

        pointer = last_bp[:, STOP_TAG]
        insert_last = pointer.contiguous().view(batch_size,1,1).expand(batch_size,1, tag_size).to(device)
        back_points = back_points.transpose(1,0).contiguous().to(device)
        back_points.scatter_(1, last_position, insert_last).to(device)
        back_points = back_points.transpose(1,0).contiguous().to(device)
        decode_idx = autograd.Variable(torch.LongTensor(seq_len, batch_size))
        if self.gpu:
            decode_idx = decode_idx.to(device)
        decode_idx[-1] = pointer.detach()
        for idx in range(len(back_points)-2, -1, -1):
            pointer = torch.gather(back_points[idx], 1, pointer.contiguous().view(batch_size, 1)).to(device)
            decode_idx[idx] = pointer.detach().view(batch_size).to(device)
        path_score = None
        decode_idx = decode_idx.transpose(1,0)
        return path_score, decode_idx

    def forward(self, feats):
        path_score, best_path = self._viterbi_decode(feats)
        return path_score, best_path

    def _score_sentence(self, scores, mask, tags):
        """
            input:
                scores: variable (seq_len, batch, tag_size, tag_size)
                mask: (batch, seq_len)
                tags: tensor  (batch, seq_len)
            output:
                score: sum of score for gold sequences within whole batch
        """
        # Gives the score of a provided tag sequence
        batch_size = scores.size(1)
        seq_len = scores.size(0)
        tag_size = scores.size(2)
        new_tags = autograd.Variable(torch.LongTensor(batch_size, seq_len))
        if self.gpu:
            new_tags = new_tags.cuda()
        for idx in range(seq_len):
            if idx == 0:
                new_tags[:, 0] = (tag_size - 2)*tag_size + tags[:, 0]

            else:
                new_tags[:, idx] = tags[:, idx-1]*tag_size + tags[:, idx]

        end_transition = self.transitions[:,STOP_TAG].contiguous().view(1, tag_size).expand(batch_size, tag_size).to(device)
        length_mask = torch.sum(mask.long(), dim = 1).view(batch_size,1).long().to(device)
        end_ids = torch.gather(tags, 1, length_mask - 1).to(device)

        end_energy = torch.gather(end_transition.to(device), 1, end_ids.to(device)).to(device)

        new_tags = new_tags.transpose(1,0).contiguous().view(seq_len, batch_size, 1).to(device)
        tg_energy = torch.gather(scores.view(seq_len, batch_size, -1), 2, new_tags).view(seq_len, batch_size).to(device)  # seq_len * bat_size
        mask = mask.bool()
        tg_energy = tg_energy.masked_select(mask.transpose(1,0)).to(device)

        gold_score = tg_energy.sum().to(device) + end_energy.sum().to(device)
        return gold_score

    def neg_log_likelihood_loss(self, feats, mask, tags):
        batch_size = feats.size(0)
        forward_score, scores = self._calculate_PZ(feats, mask)
        gold_score = self._score_sentence(scores, mask, tags)
        return forward_score - gold_score

    def _viterbi_decode_nbest(self, feats, mask, nbest):
        """
            input:
                feats: (batch, seq_len, self.tag_size+2)
                mask: (batch, seq_len)
            output:
                decode_idx: (batch, nbest, seq_len) decoded sequence
                path_score: (batch, nbest) corresponding score for each sequence (to be implementated)
                nbest decode for sentence with one token is not well supported, to be optimized
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)
        assert(tag_size == self.tagset_size+2)
        length_mask = torch.sum(mask.long(), dim = 1).view(batch_size,1).long().to(device)
        mask = mask.transpose(1,0).contiguous()
        ins_num = seq_len * batch_size
        feats = feats.transpose(1,0).contiguous().view(ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)
        scores = feats + self.transitions.view(1,tag_size,tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size).to(device)

        seq_iter = enumerate(scores)
        back_points = list()
        partition_history = list()
        mask = (1 - mask.long()).byte()
        _, inivalues = next(seq_iter)  # bat_size * from_target_size * to_target_size
        partition = inivalues[:, START_TAG, :].clone()  # bat_size * to_target_size
        partition_history.append(partition.view(batch_size, tag_size, 1).expand(batch_size, tag_size, nbest))
        for idx, cur_values in seq_iter:
            if idx == 1:
                cur_values = cur_values.view(batch_size, tag_size, tag_size) + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
            else:
                cur_values = cur_values.view(batch_size, tag_size, 1, tag_size).expand(batch_size, tag_size, nbest, tag_size) + partition.contiguous().view(batch_size, tag_size, nbest, 1).expand(batch_size, tag_size, nbest, tag_size)
                cur_values = cur_values.view(batch_size, tag_size*nbest, tag_size)
                # print "cur size:",cur_values.size()
            partition, cur_bp = torch.topk(cur_values, nbest, 1)
            if idx == 1:
                cur_bp = cur_bp*nbest
            partition = partition.transpose(2,1).to(device)
            cur_bp = cur_bp.transpose(2,1).to(device)
            partition_history.append(partition)
            mask = mask.bool()
            cur_bp.masked_fill_(mask[idx].view(batch_size, 1, 1).expand(batch_size, tag_size, nbest), 0).to(device)
            back_points.append(cur_bp)
        partition_history = torch.cat(partition_history,0).view(seq_len, batch_size, tag_size, nbest).transpose(1,0).contiguous() ## (batch_size, seq_len, nbest, tag_size)
        last_position = length_mask.view(batch_size,1,1,1).expand(batch_size, 1, tag_size, nbest) - 1
        last_partition = torch.gather(partition_history, 1, last_position).view(batch_size, tag_size, nbest, 1)
        last_values = last_partition.expand(batch_size, tag_size, nbest, tag_size) + self.transitions.view(1, tag_size, 1, tag_size).expand(batch_size, tag_size, nbest, tag_size)
        last_values = last_values.view(batch_size, tag_size*nbest, tag_size)
        end_partition, end_bp = torch.topk(last_values, nbest, 1)
        end_bp = end_bp.transpose(2,1)
        # end_bp: (batch, tag_size, nbest)
        pad_zero = autograd.Variable(torch.zeros(batch_size, tag_size, nbest)).long()
        if self.gpu:
            pad_zero = pad_zero.cuda()
        back_points.append(pad_zero)
        back_points = torch.cat(back_points).view(seq_len, batch_size, tag_size, nbest)

        pointer = end_bp[:, STOP_TAG, :] ## (batch_size, nbest)
        insert_last = pointer.contiguous().view(batch_size, 1, 1, nbest).expand(batch_size, 1, tag_size, nbest)
        back_points = back_points.transpose(1,0).contiguous()
        back_points.scatter_(1, last_position, insert_last)

        back_points = back_points.transpose(1,0).contiguous()
        decode_idx = autograd.Variable(torch.LongTensor(seq_len, batch_size, nbest))
        if self.gpu:
            decode_idx = decode_idx.cuda()
        decode_idx[-1] = pointer.data/nbest
        for idx in range(len(back_points)-2, -1, -1):
            new_pointer = torch.gather(back_points[idx].view(batch_size, tag_size*nbest), 1, pointer.contiguous().view(batch_size,nbest))
            decode_idx[idx] = new_pointer.data/nbest
            pointer = new_pointer + pointer.contiguous().view(batch_size,nbest)*mask[idx].view(batch_size,1).expand(batch_size, nbest).long()

        path_score = None
        decode_idx = decode_idx.transpose(1,0).to(device)
        scores = end_partition[:, :, STOP_TAG]
        max_scores,_ = torch.max(scores, 1).to(device)
        minus_scores = scores - max_scores.view(batch_size,1).expand(batch_size, nbest)
        path_score = F.softmax(minus_scores, 1).to(device)
        return path_score, decode_idx


class TransCrf(nn.Module):
    def __init__(self, model, num_tags):
        super(TransCrf, self).__init__()
        self.model = model
        self.crf = CRF(num_tags, False)
        self.hidden2tag = nn.Linear(num_tags, num_tags+2)
        # num_tags = 5

    def loss(self, src, pos_ind, src_mask, batch_labels, parse):
        features = self.model(src, pos_ind, src_mask, parse).to(device) #B, L, C
        tags = batch_labels.clone().to(device) #B, L
        features = features.float().to(device)
        outs = torch.cat([features, torch.ones(features.size(0), features.size(1),2).to(device)],dim=2).to(device)
        masks = src_mask.squeeze().to(device) #B, L
        if masks.dim()==1:
            masks=masks.unsqueeze(0)
        loss = self.crf.neg_log_likelihood_loss(outs, masks, tags).to(device)
        return loss

    def forward(self, src, pos_ind, src_mask, parse):
        features = self.model(src, pos_ind, src_mask, parse).to(device) #B, L, C
        features = features.float().to(device)
        outs = torch.cat([features, torch.ones(features.size(0), features.size(1), 2).to(device)], dim=2).to(device)
        masks = src_mask.squeeze().unsqueeze(0).to(device)  # B, L
        if masks.dim()==1:
            masks=masks.unsqueeze(0)
        scores, tag_seq = self.crf._viterbi_decode(outs, masks)
        tag_seq = [torch.tensor(item.clone()) for sublist in tag_seq for item in sublist]
        return scores, tag_seq


def run_crf_epoch(data_iter, loss_compute_crf):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0

    for i, batch in enumerate(data_iter):
        loss_crf = loss_compute_crf(batch.src.to(device), batch.pos.to(device), batch.src_mask.to(device),
                                    batch.trg.to(device), batch.parse)
        total_loss += loss_crf.item()
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        elapsed = time.time() - start
        print("Epoch Step: %d Loss_crf: %f Tokens per Sec: %f" %
              (i, loss_crf, tokens / elapsed))
        start = time.time()
        tokens = 0

    return total_loss


def predict_crf(data_iter, model, entity=False, rel=False):
    labels_all = []

    with torch.no_grad():
        if entity and not rel:
            for batch in data_iter:
                _, label = model(batch.src.to(device), batch.pos.to(device), batch.src_mask.to(device), batch.parse)
                labels_all.append(label)
            return labels_all


class CrfLoss:
    def __init__(self, model, opt=None):
        self.opt = opt
        self.model = model

    def __call__(self, src, pos_ind, src_mask, batch_labels, parse):

        loss = self.model.loss(src, pos_ind, src_mask, batch_labels, parse).to(device)
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            with torch.no_grad():
                model_maxsat.r.clamp_(0, 1)
            self.opt.zero_grad()
        return loss


def run_trans_epoch(data_iter, model, loss_compute, entity=False, rel=False):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0

    for i, batch in enumerate(data_iter):
        out_chunk, _, _ = model.forward(batch.src.to(device), batch.pos.to(device), batch.src_mask.to(device), batch.parse)
        loss = loss_compute(out_chunk, batch.trg.to(device), \
                            batch.ntokens.to(device), batch.src_mask.to(device))
        total_loss += loss.item()
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        elapsed = time.time() - start
        print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
              (i, loss, tokens / elapsed))
        start = time.time()
        tokens = 0

    return total_loss


def predict(data_iter, model, entity=False, rel=False):
    labels_all = []

    with torch.no_grad():
        if entity and not rel:
            for batch in data_iter:
                out, _, _ = model.forward(batch.src.to(device), batch.pos.to(device), \
                                       batch.src_mask.to(device), entity, rel)
                label = torch.argmax(out, dim=1)
                labels_all.append(label)
            return labels_all


class TransLoss:
    def __init__(self, opt=None):
        self.opt = opt

    def __call__(self, x, y, norm, mask):
        # losses_flat: (batch * max_len, 1)
        losses_flat = -torch.log(torch.gather(x.contiguous(), dim=1, index=y.contiguous().view(-1, 1)))
        # losses: (batch, max_len)
        losses = losses_flat.view(*y.size())

        # mask: (batch, max_len)
        mask = mask.squeeze()
        loss = (losses * mask.float()).sum() / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.zero_grad()
        return loss


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr/(1+decay_rate*epoch)
    print(" Learning rate is set as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


criterion = nn.NLLLoss()

emb = cPickle.load(open(opt['path'] + "data/Aspect_Opinion/embedding300_laptop", "rb"))
idxs_train = cPickle.load(open(opt['path'] +"data/Aspect_Opinion/idx_laptop.train", "rb"))
idxs_test = cPickle.load(open(opt['path'] +"data/Aspect_Opinion/idx_laptop.test", "rb"))
labels_train = cPickle.load(open(opt['path'] +"data/Aspect_Opinion/labels_laptop.train", "rb"))
labels_test = cPickle.load(open(opt['path'] +"data/Aspect_Opinion/labels_laptop.test", "rb"))
pos_train, pos_test = cPickle.load(open(opt['path'] +"data/Aspect_Opinion/pos_laptop.tag", "rb"))
parse_train = cPickle.load(open(opt['path'] +"data/Aspect_Opinion/parse_pos_laptop.train", "rb"))
parse_test = cPickle.load(open(opt['path'] +"data/Aspect_Opinion/parse_pos_laptop.test", "rb"))

pos_dic = ['pad']
posind_train, posind_test = [], []
for pos_line in pos_train:
    pos_ind = []
    for pos in pos_line:
        if pos not in pos_dic:
            pos_dic.append(pos)
        pos_ind.append(pos_dic.index(pos))
    posind_train.append(pos_ind)
for pos_line in pos_test:
    pos_ind = []
    for pos in pos_line:
        if pos not in pos_dic:
            pos_dic.append(pos)
        pos_ind.append(pos_dic.index(pos))
    posind_test.append(pos_ind)

START_TAG = -2
STOP_TAG = -1


#S_t is the rule matrix transpose
S_t = torch.tensor([
    #rule1
    # B-ASP(Y) or I-ASP(Y) & POS['PROPN', 'NOUN'](X) & POS['PROPN', 'NOUN'](Y) & nn(X,Y) --> I-ASP(X)
    [-1.,   0.,  0.,  1.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0.,  0.,     0., -1., 0., -1.,    -1.,  0.,  0.,  0.],
    [-1.,  -1.,  0.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0.,  0.,     0., -1., 0., -1.,    -1.,  0.,  0.,  0.],
    [-1.,   0., -1.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0.,  0.,     0., -1., 0., -1.,    -1.,  0.,  0.,  0.],
    [-1.,   0.,  0.,  0., -1.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0.,  0.,     0., -1., 0., -1.,    -1.,  0.,  0.,  0.],
    [-1.,   0.,  0.,  0.,  0., -1.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0.,  0.,     0., -1., 0., -1.,    -1.,  0.,  0.,  0.],

    [-1.,   0.,  0.,  1.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0.,  0.,     0., -1., 0., -1.,    -1.,  0.,  0.,  0.],
    [-1.,  -1.,  0.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0.,  0.,     0., -1., 0., -1.,    -1.,  0.,  0.,  0.],
    [-1.,   0., -1.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0.,  0.,     0., -1., 0., -1.,    -1.,  0.,  0.,  0.],
    [-1.,   0.,  0.,  0., -1.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0.,  0.,     0., -1., 0., -1.,    -1.,  0.,  0.,  0.],
    [-1.,   0.,  0.,  0.,  0., -1.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0.,  0.,     0., -1., 0., -1.,    -1.,  0.,  0.,  0.],

    #rule2
    # [B-OPN(X) or I-OPN(X)] & NOT OPN(X-1) & POS['ADJ'](X) & POS['ADJ'](Y) & conj(X,Y) --> B-OPN(Y)
    # [B-OPN(X) or I-OPN(X)] & OPN(X-1) & POS['ADJ'](X) & POS['ADJ'](Y) & conj(X,Y) --> I-OPN(Y)
    [-1.,   0.,  0.,  0., -1.,  0.,     0.,  0.,  0.,  1.,  0.,    0.,  0.,  0., -1.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0., -1.,  0.,    -1.,  0.,  0.,  0.,  0.,    0.,  0.,  0., -1.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0., -1.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0., -1.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0., -1.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0., -1.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0., -1.,  0.,     0.,  0.,  0.,  0., -1.,    0.,  0.,  0., -1.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],

    [-1.,   0.,  0.,  0., -1.,  0.,    -1.,  0.,  0.,  0.,  0.,    0.,  0., -1.,  0.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0., -1.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0., -1.,  0.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0., -1.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0., -1.,  0.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0., -1.,  0.,     0.,  0.,  0., -1.,  0.,    0.,  0., -1.,  0.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0., -1.,  0.,     0.,  0.,  0.,  0.,  1.,    0.,  0., -1.,  0.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],

    [-1.,   0.,  0.,  0.,  0., -1.,    -1.,  0.,  0.,  0.,  0.,    0.,  0.,  0., -1.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0.,  0., -1.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0., -1.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0.,  0., -1.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0., -1.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0.,  0., -1.,     0.,  0.,  0.,  1.,  0.,    0.,  0.,  0., -1.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0.,  0., -1.,     0.,  0.,  0.,  0., -1.,    0.,  0.,  0., -1.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],

    [-1.,   0.,  0.,  0.,  0., -1.,    -1.,  0.,  0.,  0.,  0.,    0.,  0., -1.,  0.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0.,  0., -1.,     0., -1.,  0.,  0.,  0.,    0.,  0., -1.,  0.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0.,  0., -1.,     0.,  0., -1.,  0.,  0.,    0.,  0., -1.,  0.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0.,  0., -1.,     0.,  0.,  0., -1.,  0.,    0.,  0., -1.,  0.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0.,  0., -1.,     0.,  0.,  0.,  0.,  1.,    0.,  0., -1.,  0.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],

    #rule3
    # [B-ASP(X) or I-ASP(X)] & NOT ASP(X-1) & POS['PROPN', 'NOUN'](X) & POS['PROPN', 'NOUN'](Y) & nsubj(X,Y) --> B-ASP(X)
    # [B-ASP(X) or I-ASP(X)] & ASP(X-1) & POS['PROPN', 'NOUN'](X) & POS['PROPN', 'NOUN'](Y) & nsubj(X,Y) --> I-ASP(X)
    [-1.,   0., -1.,  0.,  0.,  0.,     0.,  1.,  0.,  0.,  0.,    0., -1.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0., -1.,  0.,  0.,  0.,    -1.,  0.,  0.,  0.,  0.,    0., -1.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0., -1.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0., -1.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0., -1.,  0.,  0.,  0.,     0.,  0.,  0., -1.,  0.,    0., -1.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0., -1.,  0.,  0.,  0.,     0.,  0.,  0.,  0., -1.,    0., -1.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],

    [-1.,   0., -1.,  0.,  0.,  0.,     0.,  0.,  1.,  0.,  0.,   -1.,  0.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0., -1.,  0.,  0.,  0.,    -1.,  0.,  0.,  0.,  0.,   -1.,  0.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0., -1.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,   -1.,  0.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0., -1.,  0.,  0.,  0.,     0.,  0.,  0., -1.,  0.,   -1.,  0.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0., -1.,  0.,  0.,  0.,     0.,  0.,  0.,  0., -1.,   -1.,  0.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],

    [-1.,   0.,  0., -1.,  0.,  0.,     0.,  1.,  0.,  0.,  0.,    0., -1.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0., -1.,  0.,  0.,    -1.,  0.,  0.,  0.,  0.,    0., -1.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0., -1.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0., -1.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0., -1.,  0.,  0.,     0.,  0.,  0., -1.,  0.,    0., -1.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0., -1.,  0.,  0.,     0.,  0.,  0.,  0., -1.,    0., -1.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],

    [-1.,   0.,  0., -1.,  0.,  0.,     0.,  0.,  1.,  0.,  0.,   -1.,  0.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0., -1.,  0.,  0.,    -1.,  0.,  0.,  0.,  0.,   -1.,  0.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0., -1.,  0.,  0.,     0., -1.,  0.,  0.,  0.,   -1.,  0.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0., -1.,  0.,  0.,     0.,  0.,  0., -1.,  0.,   -1.,  0.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0., -1.,  0.,  0.,     0.,  0.,  0.,  0., -1.,   -1.,  0.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],


    #rule4
    # [B-ASP(Y) or I-ASP(Y)] & NOT OPN(X-1) & POS['ADJ'](X) & POS['PROPN', 'NOUN'](Y) & conj(X,Y) --> B-OPN(Y)
    # [B-ASP(Y) or I-ASP(Y)] & OPN(X-1) & POS['ADJ'](X) & POS['PROPN', 'NOUN'](Y) & conj(X,Y) --> I-OPN(Y)
    [-1.,   0.,  0.,  0.,  1.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0., -1.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,  -1.,  0.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0., -1.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0., -1.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0., -1.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0.,  0., -1.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0., -1.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0.,  0.,  0.,  0., -1.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0., -1.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],

    [-1.,   0.,  0.,  0.,  0.,  1.,     0., -1.,  0.,  0.,  0.,    0.,  0., -1.,  0.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,  -1.,  0.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0., -1.,  0.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0., -1.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0., -1.,  0.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0.,  0., -1.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0., -1.,  0.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0.,  0.,  0., -1.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0., -1.,  0.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],

    [-1.,   0.,  0.,  0.,  1.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0., -1.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,  -1.,  0.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0., -1.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0., -1.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0., -1.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0.,  0., -1.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0., -1.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0.,  0.,  0.,  0., -1.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0., -1.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],

    [-1.,   0.,  0.,  0.,  0.,  1.,     0.,  0., -1.,  0.,  0.,    0.,  0., -1.,  0.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,  -1.,  0.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0., -1.,  0.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0., -1.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0., -1.,  0.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0.,  0., -1.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0., -1.,  0.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0.,  0.,  0., -1.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0., -1.,  0.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],


    #rule5
    # B-ASP(Y) or I-ASP(Y) & POS['PROPN', 'NOUN'](X) & POS['ADJ'](Y) & amod(X,Y) --> I-ASP(X)
    [-1.,   0.,  0.,  1.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0.,  0.,     0.,-1., -1.,  0.,    0.,  0.,  0., -1.],
    [-1.,  -1.,  0.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0.,  0.,     0.,-1., -1.,  0.,    0.,  0.,  0., -1.],
    [-1.,   0., -1.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0.,  0.,     0.,-1., -1.,  0.,    0.,  0.,  0., -1.],
    [-1.,   0.,  0.,  0., -1.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0.,  0.,     0.,-1., -1.,  0.,    0.,  0.,  0., -1.],
    [-1.,   0.,  0.,  0.,  0., -1.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0.,  0.,     0.,-1., -1.,  0.,    0.,  0.,  0., -1.],

    [-1.,   0.,  0.,  1.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0.,  0.,     0.,-1., -1.,  0.,    0.,  0.,  0., -1.],
    [-1.,  -1.,  0.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0.,  0.,     0.,-1., -1.,  0.,    0.,  0.,  0., -1.],
    [-1.,   0., -1.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0.,  0.,     0.,-1., -1.,  0.,    0.,  0.,  0., -1.],
    [-1.,   0.,  0.,  0., -1.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0.,  0.,     0.,-1., -1.,  0.,    0.,  0.,  0., -1.],
    [-1.,   0.,  0.,  0.,  0., -1.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0.,  0.,     0.,-1., -1.,  0.,    0.,  0.,  0., -1.]
    ], device=device)

#S_t_body is the matrix that contains only variables corresponding to rule bodies
S_t_body = torch.tensor([
    #rule1
    [-1.,   0.,  0.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0.,  0.,     0., -1., 0., -1.,    -1.,  0.,  0.,  0.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0.,  0.,     0., -1., 0., -1.,    -1.,  0.,  0.,  0.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0.,  0.,     0., -1., 0., -1.,    -1.,  0.,  0.,  0.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0.,  0.,     0., -1., 0., -1.,    -1.,  0.,  0.,  0.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0.,  0.,     0., -1., 0., -1.,    -1.,  0.,  0.,  0.],

    [-1.,   0.,  0.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0.,  0.,     0., -1., 0., -1.,    -1.,  0.,  0.,  0.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0.,  0.,     0., -1., 0., -1.,    -1.,  0.,  0.,  0.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0.,  0.,     0., -1., 0., -1.,    -1.,  0.,  0.,  0.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0.,  0.,     0., -1., 0., -1.,    -1.,  0.,  0.,  0.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0.,  0.,     0., -1., 0., -1.,    -1.,  0.,  0.,  0.],

    #rule2
    [-1.,   0.,  0.,  0., -1.,  0.,     0.,  0.,  0.,  0.,  0.,    0.,  0.,  0., -1.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0., -1.,  0.,     0.,  0.,  0.,  0.,  0.,    0.,  0.,  0., -1.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0., -1.,  0.,     0.,  0.,  0.,  0.,  0.,    0.,  0.,  0., -1.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0., -1.,  0.,     0.,  0.,  0.,  0.,  0.,    0.,  0.,  0., -1.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0., -1.,  0.,     0.,  0.,  0.,  0.,  0.,    0.,  0.,  0., -1.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],

    [-1.,   0.,  0.,  0., -1.,  0.,     0.,  0.,  0.,  0.,  0.,    0.,  0., -1.,  0.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0., -1.,  0.,     0.,  0.,  0.,  0.,  0.,    0.,  0., -1.,  0.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0., -1.,  0.,     0.,  0.,  0.,  0.,  0.,    0.,  0., -1.,  0.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0., -1.,  0.,     0.,  0.,  0.,  0.,  0.,    0.,  0., -1.,  0.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0., -1.,  0.,     0.,  0.,  0.,  0.,  0.,    0.,  0., -1.,  0.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],

    [-1.,   0.,  0.,  0.,  0., -1.,     0.,  0.,  0.,  0.,  0.,    0.,  0.,  0., -1.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0.,  0., -1.,     0.,  0.,  0.,  0.,  0.,    0.,  0.,  0., -1.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0.,  0., -1.,     0.,  0.,  0.,  0.,  0.,    0.,  0.,  0., -1.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0.,  0., -1.,     0.,  0.,  0.,  0.,  0.,    0.,  0.,  0., -1.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0.,  0., -1.,     0.,  0.,  0.,  0.,  0.,    0.,  0.,  0., -1.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],

    [-1.,   0.,  0.,  0.,  0., -1.,     0.,  0.,  0.,  0.,  0.,    0.,  0., -1.,  0.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0.,  0., -1.,     0.,  0.,  0.,  0.,  0.,    0.,  0., -1.,  0.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0.,  0., -1.,     0.,  0.,  0.,  0.,  0.,    0.,  0., -1.,  0.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0.,  0., -1.,     0.,  0.,  0.,  0.,  0.,    0.,  0., -1.,  0.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0.,  0., -1.,     0.,  0.,  0.,  0.,  0.,    0.,  0., -1.,  0.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],

    #rule3
    [-1.,   0., -1.,  0.,  0.,  0.,     0.,  0.,  0.,  0.,  0.,    0., -1.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0., -1.,  0.,  0.,  0.,     0.,  0.,  0.,  0.,  0.,    0., -1.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0., -1.,  0.,  0.,  0.,     0.,  0.,  0.,  0.,  0.,    0., -1.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0., -1.,  0.,  0.,  0.,     0.,  0.,  0.,  0.,  0.,    0., -1.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0., -1.,  0.,  0.,  0.,     0.,  0.,  0.,  0.,  0.,    0., -1.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],

    [-1.,   0., -1.,  0.,  0.,  0.,     0.,  0.,  0.,  0.,  0.,   -1.,  0.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0., -1.,  0.,  0.,  0.,     0.,  0.,  0.,  0.,  0.,   -1.,  0.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0., -1.,  0.,  0.,  0.,     0.,  0.,  0.,  0.,  0.,   -1.,  0.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0., -1.,  0.,  0.,  0.,     0.,  0.,  0.,  0.,  0.,   -1.,  0.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0., -1.,  0.,  0.,  0.,     0.,  0.,  0.,  0.,  0.,   -1.,  0.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],

    [-1.,   0.,  0., -1.,  0.,  0.,     0.,  0.,  0.,  0.,  0.,    0., -1.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0., -1.,  0.,  0.,     0.,  0.,  0.,  0.,  0.,    0., -1.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0., -1.,  0.,  0.,     0.,  0.,  0.,  0.,  0.,    0., -1.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0., -1.,  0.,  0.,     0.,  0.,  0.,  0.,  0.,    0., -1.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0., -1.,  0.,  0.,     0.,  0.,  0.,  0.,  0.,    0., -1.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],

    [-1.,   0.,  0., -1.,  0.,  0.,     0.,  0.,  0.,  0.,  0.,   -1.,  0.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0., -1.,  0.,  0.,     0.,  0.,  0.,  0.,  0.,   -1.,  0.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0., -1.,  0.,  0.,     0.,  0.,  0.,  0.,  0.,   -1.,  0.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0., -1.,  0.,  0.,     0.,  0.,  0.,  0.,  0.,   -1.,  0.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0., -1.,  0.,  0.,     0.,  0.,  0.,  0.,  0.,   -1.,  0.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],


    #rule4
    [-1.,   0.,  0.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0., -1.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0., -1.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0., -1.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0., -1.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0., -1.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],

    [-1.,   0.,  0.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0., -1.,  0.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0., -1.,  0.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0., -1.,  0.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0., -1.,  0.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0., -1.,  0.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],

    [-1.,   0.,  0.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0., -1.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0., -1.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0., -1.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0., -1.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0., -1.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],

    [-1.,   0.,  0.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0., -1.,  0.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0., -1.,  0.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0., -1.,  0.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0., -1.,  0.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0., -1.,  0.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],


    #rule5
    [-1.,   0.,  0.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0.,  0.,     0.,-1., -1.,  0.,    0.,  0.,  0., -1.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0.,  0.,     0.,-1., -1.,  0.,    0.,  0.,  0., -1.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0.,  0.,     0.,-1., -1.,  0.,    0.,  0.,  0., -1.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0.,  0.,     0.,-1., -1.,  0.,    0.,  0.,  0., -1.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0.,  0.,     0.,-1., -1.,  0.,    0.,  0.,  0., -1.],

    [-1.,   0.,  0.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0.,  0.,     0.,-1., -1.,  0.,    0.,  0.,  0., -1.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0.,  0.,     0.,-1., -1.,  0.,    0.,  0.,  0., -1.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0.,  0.,     0.,-1., -1.,  0.,    0.,  0.,  0., -1.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0.,  0.,     0.,-1., -1.,  0.,    0.,  0.,  0., -1.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0.,  0.,     0.,-1., -1.,  0.,    0.,  0.,  0., -1.]
    ], device=device)

S = S_t.t()
# n: number of rules, m: number of logical variables
n, m, k = S.size(0), S.size(1), 32
S = S * ((.5 / (n + 1 + m)) ** 0.5)
pi = torch.tensor([math.pi], device=device)

emb = torch.tensor(emb, dtype=torch.float)
model_trans = make_model(emb.shape[0], len(pos_dic), emb, sch_k=1.0, N=2)
model_trans = model_trans.to(device)

f_out = open(opt['path'] + "result/Aspect-Opinion/DeepWMaxSAT_laptop_ada_2e-3_5e-4.txt", "w")


label_map = {0: 'O', 1: 'B-ASP', 2: 'I-ASP', 3: 'B-OPN', 4: 'I-OPN'}

model_maxsat = MaxsatLayer(model_trans, S, S_t)
model_crf = TransCrf(model_maxsat, num_tags=5)

optimizer_trans = optim.Adadelta(model_trans.parameters())
optimizer = optim.Adadelta(model_crf.parameters())


for epoch in range(opt['preepoch']):
    model_trans.train()
    loss = run_trans_epoch(data_batch(idxs_train, labels_train, posind_train, parse_train, 25), model_trans,
                           TransLoss(optimizer_trans), entity=True, rel=False)

    model_trans.eval()
    labels_predict = predict(data_batch(idxs_test, labels_test, posind_test, parse_test, 1), model_trans, entity=True, rel=False)

    labels_test_map = [label_map[item] for sub in labels_test for item in sub]
    labels_predict_map = [label_map[t.item()] for sub in labels_predict for t in sub]

    print("Pre-training epoch: %d" %epoch)
    print(eva.f1_score(labels_test_map, labels_predict_map))
    report = eva.classification_report(labels_test_map, labels_predict_map)
    print(report)
    print(loss)

    f_out.write("epoch: " + str(epoch) + "\n")
    f_out.write("performance on entity extraction:" + "\n")
    f_out.write("precision: " + str(eva.precision_score(labels_test_map, labels_predict_map)))
    f_out.write("\t" + "recall: " + str(eva.recall_score(labels_test_map, labels_predict_map)))
    f_out.write("\t" + "f1: " + str(eva.f1_score(labels_test_map, labels_predict_map)))
    f_out.write("\n" + report)
    f_out.write("\n")

for epoch in range(opt['epoch']):
    optimizer_crf = lr_decay(optimizer, epoch, 5e-4, 2e-3)
    model_trans.train()
    model_maxsat.train()
    model_crf.train()
    loss = run_crf_epoch(data_batch(idxs_train, labels_train, posind_train, parse_train, 25), CrfLoss(model_crf, optimizer_crf))
    model_trans.eval()
    model_maxsat.eval()
    model_crf.eval()
    labels_predict = predict_crf(data_batch(idxs_test, labels_test, posind_test, parse_test, 1),
                                              model_crf, entity=True, rel=False)
    labels_test_map = [label_map[item] for sub in labels_test for item in sub]
    labels_predict_map = [label_map[t.item()] for sub in labels_predict for t in sub]

    print("Training epoch: %d" %epoch)
    print(eva.f1_score(labels_test_map, labels_predict_map))
    report = eva.classification_report(labels_test_map, labels_predict_map)
    print(report)

    f_out.write("Training epoch: " + str(epoch) + "\n")
    f_out.write("performance on entity extraction:" + "\n")
    f_out.write("precision: " + str(eva.precision_score(labels_test_map, labels_predict_map)))
    f_out.write("\t" + "recall: " + str(eva.recall_score(labels_test_map, labels_predict_map)))
    f_out.write("\t" + "f1: " + str(eva.f1_score(labels_test_map, labels_predict_map)))
    f_out.write("\n" + report)
    f_out.write("\n")

f_out.close()






