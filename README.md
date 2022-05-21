# AllinLosses

![](https://gitee.com/axjing/AnImg/raw/master/AllinLosses.png)

Deep learning of all loss functions
https://github.com/axjing/AllinLosses
>该仓库为常用效果较好的损失函数的集装箱，Pytotch实现，持续更新，保证每个方法都方便高效的使用，借鉴开源社区中的相关代码，实现接口、框架、方法的统一，方便使用与研究，欢迎star,另外也欢迎star:[AIAnClub](https://github.com/axjing/AIAnClub): https://github.com/axjing/AIAnClub
它为流行模型的算法实现

## 更新内容：
- 与分割任务有关的所有loss函数，经过严格测试，兼容2d和3d分割模型训练，拿来即用:
  - GeneralizedDiceWithFocalLoss
  - FocalLoss
  - FocalLossV2
  - CrossentropyND
  - CrossentropyNDTopK
  - TopKThreshold
  - WeightedCrossEntropyLoss
  - WeightedCrossEntropyLossV2
  - DistPenalizedCE
  - TopKLoss
  - SoftDiceLoss
  - DiceWithTopKLoss
  - ExpLogLoss
  - GeneralizedDiceLoss
  - PenaltyGeneralizedDiceLoss
  - IoULoss
  - SensitivitySpecifityLoss
  - TverskyLoss
  - FocalTverskyLoss
  - AsymLoss
  - DiceWithCrossentropyNDLoss
  - DiceWithFocalLoss
  - LovaszSoftmax
  - BoudaryLoss
  - DiceWithBoundaryLoss
  - GeneralizedDiceWithBoundaryLoss
  - HDLoss
  - DiceWithHDLoss
  - log_cosh_smooth `loss平滑`
  - BECLoss


## 使用方法
TODO:
1. **clone**
```shell
git clone https://github.com/axjing/AllinLosses.git
cd ./AllinLosses
pip install -r requirements.txt
```

2. **import**

```python
from AllinLosses import *
```

3. **测试所有损失函数**

```python
# test.py
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
```
**OUTPUT:**
```python
['FocalLoss', 'CrossentropyND', 'CrossentropyNDTopK', 'BWeightedCrossEntropyLoss', 'WeightedCrossEntropyLoss', 'WeightedCrossEntropyLossV2', 'TopKLoss', 'TopKThreshold', 'TverskyLoss', 'AsymLoss', 'DistPenalizedCE', 'PenaltyGeneralizedDiceLoss', 'SensitivitySpecifityLoss', 'SoftDiceLoss', 'SoftDiceLossV2', 'IoULoss', 'GeneralizedDiceLoss', 'GeneralizedDiceLossV2', 'GeneralizedDiceLossV3', 'ExpLogLoss', 'LovaszSoftmax', 'BoudaryLoss', 'HDLoss', 'FocalTverskyLoss', 'DiceWithHDLoss', 'DiceWithBoundaryLoss', 'DiceWithBCELoss', 'DiceWithFocalLoss', 'GeneralizedDiceWithBoundaryLoss', 'GeneralizedDiceWithBCELoss', 'GeneralizedDiceWithFocalLoss', 'DiceWithCrossentropyNDLoss']
32
INFO:__main__:  FocalLoss:                      11.4899
INFO:__main__:  CrossentropyND:                 1.0975
INFO:__main__:  CrossentropyNDTopK:                     1.0975
INFO:__main__:  BWeightedCrossEntropyLoss:                      0.6929
INFO:__main__:  WeightedCrossEntropyLoss:                       1.0975
INFO:__main__:  WeightedCrossEntropyLossV2:                     1.0975
C:\Users\axjing\anaconda3\envs\jing\lib\site-packages\torch\nn\_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='none' instead.
  warnings.warn(warning.format(ret))
INFO:__main__:  TopKLoss:                       1.0986
INFO:__main__:  TopKThreshold:                  nan
INFO:__main__:  TverskyLoss:                    0.4666
INFO:__main__:  AsymLoss:                       0.4657
INFO:__main__:  DistPenalizedCE:                        1.0975
INFO:__main__:  PenaltyGeneralizedDiceLoss:                     0.9996
INFO:__main__:  SensitivitySpecifityLoss:                       0.0496
INFO:__main__:  SoftDiceLoss:                   0.4409
INFO:__main__:  SoftDiceLossV2:                 0.4409
INFO:__main__:  IoULoss:                        0.4958
INFO:__main__:  GeneralizedDiceLoss:                    0.0019
INFO:__main__:  GeneralizedDiceLossV2:                  0.0020
INFO:__main__:  GeneralizedDiceLossV3:                  0.0020
INFO:__main__:  ExpLogLoss:                     1.9782
INFO:__main__:  LovaszSoftmax:                  0.4960
INFO:__main__:  BoudaryLoss:                    -0.0019
INFO:__main__:  HDLoss:                 0.0025
INFO:__main__:  FocalTverskyLoss:                       1.3327
INFO:__main__:  DiceWithHDLoss:                 Total:0.0049 | loss1:0.8300 | loss2:0.0025
INFO:__main__:  DiceWithBoundaryLoss:                   Total:0.0065 | loss1:0.8300 | loss2:-0.0019
INFO:__main__:  DiceWithBCELoss:                        Total:1.5229 | loss1:0.8300 | loss2:0.6929
INFO:__main__:  DiceWithFocalLoss:                      Total:1.3175 | loss1:0.8300 | loss2:0.4875
INFO:__main__:  GeneralizedDiceWithBoundaryLoss:                        Total:-0.0018 | loss1:0.0020 | loss2:-0.0019
INFO:__main__:  GeneralizedDiceWithBCELoss:                     Total:1.6928 | loss1:0.9999 | loss2:0.6929
INFO:__main__:  GeneralizedDiceWithFocalLoss:                   Total:1.5285 | loss1:0.9999 | loss2:0.4875
INFO:__main__:  DiceWithCrossentropyNDLoss:                     Total:1.9275 | loss1:0.8300 | loss2:1.0975
******************************
 |              End Of Program          |
******************************
```
4. 使用方法

```python
from AllinLosses.segment_losses import *
if __name__ == "__main__":
    output = torch.zeros(2, 2, 64, 64,64)
    output[:, 0, 10:20, 10:20, 10:20] = 0
    output[:, 1, 12:20, 12:20, 12:20] = 1

    target = torch.zeros(2, 64, 64,64)
    # target[:, 5:15, 5:15, 5:15] = 1
    target[:, 10:20, 10:20, 10:20] = 1

    dice_loss = SoftDiceLoss(smooth=1e-5)
    dice_lv = dice_loss(output, target)
    print(dice_lv)
    
    gdl = GeneralizedDiceLoss()
    dice_lv = gdl(output, target)
    print(dice_lv)
    BDL = BoudaryLoss()
    print(BDL(output, target))
    para = dict(batch_dice=False, do_bg=True, smooth=1e-5)
    DBDL = DiceWithBoundaryLoss(para)
    print(DBDL(output, target))
    DHDL = DiceWithHDLoss(para)
    print(DHDL(output, target))
    GDBL = GeneralizedDiceWithBoundaryLoss()
    print(GDBL(output, target))

    GDFL=GeneralizedDiceWithFocalLoss()
    print(GDFL(output, target))

```

## 不同损失函数使用场景->待更新

## 关于loss函数的讲解：
- [损失函数优缺点适用场景对比分析](https://mp.weixin.qq.com/s/hrxFWmPdZkZmyA9PdJZkxw)

- [机器学习损失函数及优化方法](https://mp.weixin.qq.com/s/AVDlh5fTJqqOE4E5jRf6BQ)

- [GeneralisedDiceLoss解决类别极不均衡问题](https://mp.weixin.qq.com/s/CE_Lhg6-KKu61trQ1DlwGQ)

- [pytorch反向传播|数据加载|数据集使用|自定义数据集构建|训练过程|模型加载与保存](https://mp.weixin.qq.com/s/5_VQiKmidH_ZkfaIKE9BUA)

- [深度学习模型训练中常见问题总结](https://mp.weixin.qq.com/s/iceVwKaJCDE57jadceofNg)

- [Loss出现NaN的原因及解决方法​](https://mp.weixin.qq.com/s/7STgxx_TJM8W-J3E7FX30Q)

- [Loss震荡剧烈原因及解决办法​](https://mp.weixin.qq.com/s/onVwjNEhciOqnxqFCSSoyQ)


## 参考文献

所有参考文献均在代码注释中写明，方式使用时理解、阅读

>![欢迎关注：【未来现相】微信公众号！！！](https://gitee.com/axjing/AnImg/raw/master/20210808192034.png)

>未来现相WX公众号主要以下服务：
>1. 提供专业的AI解决方案及相关产品落地；
>2. 项目互助，科技创业；
>3. 每日更新AI算法研究等精品内容；
>4. 为IT从业者提供完备的职业发展规划，解决人生不同阶段的难题；
>5. **一起干，自律，自由，有知识，有力量，有未来，一起富裕​;**
