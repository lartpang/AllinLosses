# python3
# @File: log.py
# --coding:utf-8--
# @Author:axjing
# @Time: 2021年09月16日17
# 说明:

from segment_losses import __all__
from segment_losses import *
import torch
print(__all__)

if __name__ == '__main__':
    import logging

    # 获取logger实例，如果参数为空则返回root logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    output = torch.zeros(2, 3, 64, 64, 64, device="cpu")
    output[:, 1, 12:20, 12:20, 12:20] = 1
    output[:, 2, 8:10, 8:10, 8:10] = 1

    target = torch.zeros(2, 64, 64, 64, device="cpu")
    target[:, 10:20, 10:20, 10:20] = 1
    target[:, 8:10, 8:10, 8:10] = 2

    # print(TopKLoss()(output, target))
    print(len(__all__))
    for i in range(len(__all__)):
        criterion = eval(__all__[i])
        loss_val = criterion()(output, target)
        if isinstance(loss_val, tuple):
            loss_total, loss1, loss2 = loss_val
            logger.info(
                "\t{}:\t\t\tTotal:{:.4f} | loss1:{:.4f} | loss2:{:.4f}".format(__all__[i], loss_total.data, loss1.data,
                                                                               loss2.data))
        else:
            logger.info("\t{}:\t\t\t{:.4f}".format(__all__[i], loss_val.data))
    print("*" * 30 + "\n |\t\tEnd Of Program\t\t|\n" + "*" * 30)
