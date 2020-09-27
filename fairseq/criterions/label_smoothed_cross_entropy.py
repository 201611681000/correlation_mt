# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import sys
import numpy as np
import torch

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion



def label_smoothed_nll_loss(samples, sent_len, lprobs, cand_probs, target, cand, epsilon, ignore_index=None, reduce=True):
    lambda_p = 1e4
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    if cand.dim() == cand_probs.dim() - 1:
        cand = cand.unsqueeze(-1)

    nll_loss = -lprobs.gather(dim=-1, index=target)
    cand_loss = -cand_probs.gather(dim=-1, index=cand)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)

    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        cand_pad_mask = cand.eq(ignore_index)

        nll_loss.masked_fill_(pad_mask, 0.)
        cand_loss.masked_fill_(cand_pad_mask, 0.)
        smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    def cal_pearson(x, y):
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)

        a = torch.sqrt(torch.sum(vx ** 2))* torch.sqrt(torch.sum(vy ** 2))
        b = torch.sum(vx*vy)
        r = b / a
        return r

    cand_loss_trans = cand_loss.view(-1,sent_len)
    cand_loss = cand_loss_trans.sum(axis=1)
    
    torch_bleu = torch.tensor(samples['bleu']).cuda()

    pearson = cal_pearson(torch_bleu,cand_loss)
    
    if len(cand_loss) > 200:
        print ("probs:{}\tbleu:{}".format(cand_loss, torch_bleu))
        print (pearson)
        
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()

    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    
    total_loss = loss + lambda_p*pearson

    return total_loss, loss, nll_loss, pearson


@register_criterion('label_smoothed_cross_entropy')
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, task, sentence_avg, label_smoothing):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--lamba-for-corr', default=0., type=float, metavar='D',
                            help='epsilon for the weight of correlation, 0 means no adding correlation')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output_tgt, net_output_cand = model(**sample['net_input'])

        #net_output_tgt = model(sample['net_input']['src_tokens'], sample['net_input']['src_lengths'], sample['net_input']['prev_output_tokens'])
        #net_output_cand = model(sample['net_input']['src_tokens'], sample['net_input']['src_lengths'], sample['net_input']['prev_cand_output_tokens'])

        total_loss, loss, nll_loss, pearson = self.compute_loss(model, net_output_tgt, net_output_cand, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'total_loss':total_loss.data,
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'pearson': pearson.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return total_loss, sample_size, logging_output

    def compute_loss(self, model, tgt_output, cand_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(tgt_output, log_probs=True)
        cand_lprobs = model.get_normalized_probs(cand_output, log_probs=True)
        sent_len = len(cand_lprobs[0])

        lprobs = lprobs.view(-1, lprobs.size(-1))
        cand_lprobs = cand_lprobs.view(-1, cand_lprobs.size(-1))

        target = model.get_targets(sample, tgt_output).view(-1, 1)
        cand = model.get_cands(sample, cand_output).view(-1, 1)

        total_loss, loss, nll_loss, pearson = label_smoothed_nll_loss(
            sample, sent_len, lprobs, cand_lprobs, target, cand, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        return total_loss, loss, nll_loss, pearson

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        total_loss_sum = sum(log.get('total_loss', 0) for log in logging_outputs)
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        cor_pearson = sum(log.get('pearson', 0) for log in logging_outputs)

        metrics.log_scalar('total_loss', total_loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_scalar('pearson', cor_pearson, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
