# AllinLosses
Deep learning of all loss functions
https://github.com/axjing/AllinLosses
>该仓库为常用效果较好的损失函数的集装箱，Pytotch实现，持续更新，保证每个方法都方便高效的使用，借鉴开源社区中的相关代码，实现接口、框架、方法的统一，方便使用与研究，欢迎star,另外也欢迎star:[AIAnClub](https://github.com/axjing/AIAnClub): https://github.com/axjing/AIAnClub,它为流行模型的算法实现
![](https://gitee.com/axjing/AnImg/raw/master/AllinLosses.png)


## 更新内容：
- 与分割任务有关的所有loss函数，经过严格测试，兼容2d和3d分割模型训练，拿来即用:
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

## 使用方法
TODO:
1. **clone**
```shell
git clone https://github.com/axjing/AllinLosses.git
```

2. **import**

```python
from AllinLosses import *
```

3. instantiation

## 关于loss函数的讲解：
- [损失函数优缺点适用场景对比分析](https://mp.weixin.qq.com/s/hrxFWmPdZkZmyA9PdJZkxw)

- [机器学习损失函数及优化方法](https://mp.weixin.qq.com/s/AVDlh5fTJqqOE4E5jRf6BQ)

- [GeneralisedDiceLoss解决类别极不均衡问题](https://mp.weixin.qq.com/s/CE_Lhg6-KKu61trQ1DlwGQ)

- [pytorch反向传播|数据加载|数据集使用|自定义数据集构建|训练过程|模型加载与保存](https://mp.weixin.qq.com/s/5_VQiKmidH_ZkfaIKE9BUA)

- [深度学习模型训练中常见问题总结](https://mp.weixin.qq.com/s/iceVwKaJCDE57jadceofNg)

- [Loss出现NaN的原因及解决方法​](https://mp.weixin.qq.com/s/7STgxx_TJM8W-J3E7FX30Q)

- [Loss震荡剧烈原因及解决办法​](https://mp.weixin.qq.com/s/onVwjNEhciOqnxqFCSSoyQ)



>![欢迎关注：【未来现相】微信公众号！！！](https://gitee.com/axjing/AnImg/raw/master/20210808192034.png)

>未来现相WX公众号主要以下服务：
>1. 提供专业的AI解决方案及相关产品落地；
>2. 项目互助，科技创业；
>3. 每日更新AI算法研究等精品内容；
>4. 为IT从业者提供完备的职业发展规划，解决人生不同阶段的难题；
>5. **一起干，自律，自由，有知识，有力量，有未来，一起富裕​;**