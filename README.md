# 解决方案
采用3中召回方式：itemcf 召回，binetwork 召回和基于 word2vec 的 i2i 召回。合并去重并删除没有召回到真实商品的用户数据后，利用特征工程+ LGB 二分类模型进行排序。

# 复现步骤
操作系统：ubuntu 16.04  
```
pip install requirements.txt
cd code
bash test.sh
```