# coding: utf-8
# 2021/8/2 @ tongshiwei
import logging
import torch
from torch.nn import DataParallel


def set_device(_net, ctx, *args, **kwargs):  # pragma: no cover
    """code from longling v1.3.26"""
    if ctx == "cpu":
        if not isinstance(_net, DataParallel):
            _net = DataParallel(_net)
        return _net.cpu()
    elif any(map(lambda x: x in ctx, ["cuda", "gpu"])):
        if not torch.cuda.is_available():
            try:
                torch.ones((1,), device=torch.device("cuda: 0"))
            except AssertionError as e:
                raise TypeError("no cuda detected, noly cpu is supported, the detailed error msg:%s" % str(e))
        if torch.cuda.device_count() >= 1:
            if ":" in ctx:
                ctx_name, device_ids = ctx.split(":")
                assert ctx_name in ["cuda", "gpu"], "the equipment should be 'cpu', 'cuda' or 'gpu', now is %s" % ctx
                device_ids = [int(i) for i in device_ids.strip().split(",")]
                try:
                    if not isinstance(_net, DataParallel):
                        return DataParallel(_net, device_ids).cuda
                    return _net.cuda(device_ids)
                except AssertionError as e:
                    logging.error(device_ids)
                    raise e
            elif ctx in ["cuda", "gpu"]:
                if not isinstance(_net, DataParallel):
                    _net = DataParallel(_net)
                return _net.cuda()
            else:
                raise TypeError("the equipment should be 'cpu', 'cuda' or 'gpu', now is %s" % ctx)
        else:
            logging.error(torch.cuda.device_count())
            raise TypeError("0 gpu can be used, use cpu")
    else:
        if not isinstance(_net, DataParallel):
            return DataParallel(_net, device_ids=ctx).cuda()
        return _net.cuda(ctx)
