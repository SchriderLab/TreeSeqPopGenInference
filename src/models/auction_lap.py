#!/usr/bin/env python3

"""
    auction_lap.py
    
    From
        https://dspace.mit.edu/bitstream/handle/1721.1/3265/P-2108-26912652.pdf;sequence=1
"""

from __future__ import print_function, division

import sys
import torch
import numpy as np

def auction_lap(X, eps=None, compute_score=True):
    """
        X: n-by-n matrix w/ integer entries
        eps: "bid size" -- smaller values means higher accuracy w/ longer runtime
    """
    eps = 1 / X.shape[0] if eps is None else eps
    
    # --
    # Init
    
    cost     = torch.zeros((1, X.shape[1]))
    curr_ass = torch.ones(X.shape[0]).long() - 1
    bids     = torch.zeros(X.shape)
    
    indices = torch.LongTensor(np.array(list(range(X.shape[0])))).to(X.device)
    
    if X.is_cuda:
        cost, curr_ass, bids = cost.cuda(), curr_ass.cuda(), bids.cuda()
    
    counter = 0
    while (curr_ass == -1).any():
        counter += 1
        
        # --
        # Bidding
        
        unassigned = torch.nonzero((curr_ass == -1).int())[:,0]
        print(unassigned.shape)
        
        value = X[unassigned] - cost
        top_value, top_idx = value.topk(2, dim=1)
        
        first_idx = top_idx[:,0]
        first_value, second_value = top_value[:,0], top_value[:,1]
        
        bid_increments = first_value - second_value + eps
        
        bids_ = bids[unassigned]
        bids_.zero_()
        bids_.scatter_(
            dim=1,
            index=first_idx.contiguous().view(-1, 1),
            src=bid_increments.view(-1, 1)
        )
        
        # --
        # Assignment
        
        have_bidder = (bids_ > 0.).int().sum(dim=0).nonzero()
        
        high_bids, high_bidders = bids_[:,have_bidder].max(dim=0)
        high_bidders = unassigned[high_bidders.squeeze()]
        
        cost[:,have_bidder] += high_bids
        
        curr_ass[(curr_ass.view(-1, 1) == have_bidder.view(1, -1)).sum(dim=1)] = -1
        curr_ass[high_bidders] = have_bidder.squeeze()
    
    score = None
    if compute_score:
        score = int(X.gather(dim=1, index=curr_ass.view(-1, 1)).sum())
    
    return score, curr_ass, counter
