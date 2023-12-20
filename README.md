# RecCheatSheet
- 推荐模型优化方案与实现
## RankingLosses
由于自然推荐场景的隐式反馈形式(1/0), 搜索场景的LTRLoss形式在自然推荐场景精排模型上有一定的限制性, 但是可以使用在粗排模型蒸馏精排模型的优化中。

### PairWise
- BPR Loss
- DValue-Weighted Distill Ranking Loss
- Bucket PairWise Ranking Loss
  - [美团-美团搜索粗排优化的探索与实践](https://tech.meituan.com/2022/08/11/coarse-ranking-exploration-practice.html)
  - [阿里-全链路联动-面向最终目标的全链路一致性建模](https://zhuanlan.zhihu.com/p/413240790)
### ListWise
- ListNet/ListMLE
  - [ICML-2007-Learning to Rank: From Pairwise Approach to Listwise Approach](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2007-40.pdf)
- LambdaLoss:
  -  [CIKM-2018-The LambdaLoss Framework for Ranking Metric Optimization](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/1e34e05e5e4bf2d12f41eb9ff29ac3da9fdb4de3.pdf)
  -  [大众点评搜索基于知识图谱的深度学习排序实践](https://tech.meituan.com/2019/01/17/dianping-search-deeplearning.html)

- ApproxMetricLoss(NDCG/MRR)
  - [微软-A General Approximation Framework for Direct
Optimization of Information Retrieval Measures](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2008-164.pdf)


### 多Loss联合蒸馏

### MTL多Loss学习

- Multi-task learning using uncertainty to weigh losses for scene geometry and semantics.
- GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks
- End-to-End Multi-Task Learning with Attention

### 精排-粗排联合训练/共享底层/Embedding蒸馏
- 阿里-Rocket Launching：A Universal and Efficient Framework for Training Well-Performing Light Net

## 双塔模型结构优化

## 特征交叉
- 华为-IntTower: the Next Generation of Two-Tower Model for Pre-Ranking System
### 专家混合网络
- 腾讯-Mixture of Virtual-Kernel Experts for Multi-Objective User Profile Modeling
- Google-Empowering Long-tail Item Recommendation through Cross Decoupling Network (CDN)

### 千人千模
- 快手-POSO: Personalized Cold Start Modules for Large-scale Recommender Systems.



