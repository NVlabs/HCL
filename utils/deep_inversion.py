# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch

class DeepInversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.mean = None
        self.var = None
        self.use_stored_stats = False
        self.capture_bn_stats = False


    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch_in = input[0].shape[1]
        nch_out = output.shape[1]

        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch_in, -1]).var(1, unbiased=False)

        in_mean = input[0].mean([0, 2, 3])
        in_var = input[0].permute(1, 0, 2, 3).contiguous().view([nch_in, -1]).var(1, unbiased=False)

        out_mean = output.mean([0, 2, 3])
        out_var = output.permute(1, 0, 2, 3).contiguous().view([nch_out, -1]).var(1, unbiased=False)

        if self.capture_bn_stats:
            self.out_mean = out_mean.clone().detach()
            self.out_var = out_var.clone().detach()

        if not self.use_stored_stats:
            r_feature = torch.norm(module.running_var.data.type(in_var.type()) - in_var, 2) + torch.norm(
            module.running_mean.data.type(in_mean.type()) - in_mean, 2)
        else:
            r_feature = torch.norm(self.out_var - out_var, 2) + torch.norm(self.out_mean - out_mean, 2)

        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()

