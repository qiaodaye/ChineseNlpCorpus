# ChineseNlpCorpus
搜集、整理、发布 中文 自然语言处理 语料/数据集，与 有志之士 共同 促进 中文 自然语言处理 的 发展。

## 情感/观点/评论 倾向性分析

| 数据集 | 数据概览 | 下载地址 |
| ----- | -------- | ------- |
| ChnSentiCorp_htl_all | 7000 多条酒店评论数据，5000 多条正向评论，2000 多条负向评论 | [点击查看](./datasets/ChnSentiCorp_htl_all/intro.ipynb) |
| waimai_10k | 某外卖平台收集的用户评价，正向 4000 条，负向 约 8000 条 | [点击查看](./datasets/waimai_10k/intro.ipynb) |
| online_shopping_10_cats | 10 个类别，共 6 万多条评论数据，正、负向评论各约 3 万条，<br /> 包括书籍、平板、手机、水果、洗发水、热水器、蒙牛、衣服、计算机、酒店 | [点击查看](./datasets/online_shopping_10_cats/intro.ipynb) |
| weibo_senti_100k | 10 万多条，带情感标注 新浪微博，正负向评论约各 5 万条 | [点击查看](./datasets/weibo_senti_100k/intro.ipynb) |
| simplifyweibo_4_moods | 36 万多条，带情感标注 新浪微博，包含 4 种情感，<br /> 其中喜悦约 20 万条，愤怒、厌恶、低落各约 5 万条 | [点击查看](./datasets/simplifyweibo_4_moods/intro.ipynb) |
| dmsc_v2 | 28 部电影，超 70 万 用户，超 200 万条 评分/评论 数据 | [点击查看](./datasets/dmsc_v2/intro.ipynb) |
| yf_dianping | 24 万家餐馆，54 万用户，440 万条评论/评分数据 | [点击查看](./datasets/yf_dianping/intro.ipynb) |
| yf_amazon | 52 万件商品，1100 多个类目，142 万用户，720 万条评论/评分数据 | [点击查看](./datasets/yf_amazon/intro.ipynb) |

## 中文命名实体识别

| 数据集 | 数据概览 | 下载地址 |
| ----- | -------- | ------- |
| dh_msra | 5 万多条中文命名实体识别标注数据（包括地点、机构、人物） | [点击查看](./datasets/dh_msra/intro.ipynb) |

## 推荐系统

| 数据集 | 数据概览 | 下载地址 |
| ----- | -------- | ------- |
| ez_douban | 5 万多部电影（3 万多有电影名称，2 万多没有电影名称），2.8 万 用户，280 万条评分数据 | [点击查看](./datasets/ez_douban/intro.ipynb) |
| dmsc_v2 | 28 部电影，超 70 万 用户，超 200 万条 评分/评论 数据 | [点击查看](./datasets/dmsc_v2/intro.ipynb) |
| yf_dianping | 24 万家餐馆，54 万用户，440 万条评论/评分数据 | [点击查看](./datasets/yf_dianping/intro.ipynb) |
| yf_amazon | 52 万件商品，1100 多个类目，142 万用户，720 万条评论/评分数据 | [点击查看](./datasets/yf_amazon/intro.ipynb) |

## FAQ 问答系统

| 数据集 | 数据概览 | 下载地址 |
| ----- | -------- | ------- |
| 保险知道 | 8000 多条保险行业问答数据，包括用户提问、网友回答、最佳回答 | [点击查看](./datasets/baoxianzhidao/intro.ipynb) |
| 安徽电信知道 | 15.6 万条电信问答数据，包括用户提问、网友回答、最佳回答 | [点击查看](./datasets/anhuidianxinzhidao/intro.ipynb) |
| 金融知道 | 77 万条金融行业问答数据，包括用户提问、网友回答、最佳回答 | [点击查看](./datasets/financezhidao/intro.ipynb) |
| 法律知道 | 3.6 万条法律问答数据，包括用户提问、网友回答、最佳回答 | [点击查看](./datasets/lawzhidao/intro.ipynb) |
| 联通知道 | 20.3 万条联通问答数据，包括用户提问、网友回答、最佳回答 | [点击查看](./datasets/liantongzhidao/intro.ipynb) |
| 农行知道 | 4 万条农业银行问答数据，包括用户提问、网友回答、最佳回答 | [点击查看](./datasets/nonghangzhidao/intro.ipynb) |
| 保险知道 | 58.8 万条保险行业问答数据，包括用户提问、网友回答、最佳回答 | [点击查看](./datasets/baoxianzhidao/intro.ipynb) |



## 加入我们

- 愿景：以人工智能产品和技术服务 30 亿人
- 团队：极客精神、技术驱动，做有温度的技术，让世界更美好
- 产品：面向细分行业领域的自动营销机器人，客户需求旺盛，产品前景无限
- 职位：[自然语言人机交互应用研究](./docs/recruit/researcher.md)、[自然语言处理算法工程师](./docs/recruit/engineer.md)、[系统架构师（人工智能产品）](./docs/recruit/architect.md)

![](./docs/images/recruit/recruit_banner.png)




情感分析语料：
中文nlp语料：https://github.com/SophonPlus/ChineseNlpCorpus

	句子级别的情绪识别，可以看成是文本分类的子任务，可以尝试使用文本分类的方法进行解决：
	textCNN
论文：Convolutional Neural Networks for Sentence Classification
参考代码：https://github.com/DongjunLee/text-cnn-tensorflow
模型：
 
Embedding:句子矩阵，每一行是一个词的向量
Convolution:卷积核=[2,3,4]的卷积层，每个卷积层输出两个channel
MaxPooling:将每个卷积层的两个channel池化成定长向量
FC&&softmax:pooling层后的向量全连接，softmax计算类别概率
	charCNN
论文：Character-level Convolutional Networks for Text Classification
代码：https://github.com/zhangxiangxiao/Crepe
模型：
 
Char Embedding:字符使用one-hot进行编码
Conv&Pooling:常见的conv和pooling，6层
FC:3层，每层之间有dropout
本文设计了两种神经网络，一大一小，都是上述C&P和FC层组成，区别在于特征长度和FC层神经元的个数
	fastText
论文：Bag of Tricks for Efficient Text Classification
代码：https://github.com/facebookresearch/fastText
模型：
 
首先将文本转化为向量，然后输入一个线性分类器得到一个 hidden，这个 hidden 是预定义类的概率分布，最后使用 softmax 计算概率。这里的 ouput 是一个 label，即分类的结果。
	bi-LSTM
资料：https://zhuanlan.zhihu.com/p/47802053
参考代码：https://github.com/albertwy/BiLSTM/
模型：
 
	bi-LSTM+ATT
论文：Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification
代码：https://github.com/SeoSangwoo/Attention-Based-BiLSTM-relation-extraction
模型：
 
Input:原始句子的输入
Embedding:词映射到向量
Lstm:提取高级特征
Attention:先产生权重向量，并与LSTM层的每一个时间节点上词级别的特征相乘，合并得到句子级别的特征向量
Output:softmax计算类别概率
	RCNN
论文：Recurrent Convolutional Neural Networks for Text Classification
代码：https://github.com/roomylee/rcnn-text-classification
模型：
 
Embedding:词映射向量
Bi-RNN(lstm or gru):词向量输入双向RNN分别得到正向和反向向量，拼接[正向向量，词向量，反向向量]，输入tanh激活函数
MaxPooling:池化成定长向量
Output: softmax计算类别概率
	Transformer
论文：Attention Is All You Need
论文中给出的代码：https://github.com/tensorflow/tensor2tensor
参考代码：https://github.com/tensorflow/models/tree/master/official/transformer
模型：
 
左边是Encoder，右边是Decoder，在文本分类任务中只需要Encoder。
Embedding：词映射向量
Positional Encoding：将位置编号，每个编号对应一个向量，结合词向量和位置向量
Encoder：N=6。每一层包含两个子层，第一层是多头注意力层（Muti-Head Attention），第二层就是全连接层（Feed-Forward），每个子层使用残差连接（Add&Norm）
Linear&softmax：根据encoder输出的向量，计算类别概率
	BERT
论文：BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
代码：https://github.com/google-research/bert
模型：
  
模型整体就是多层的双向Transformer，这是一个预训练的通用语言模型。可以根据具体应用场景，用自己的的训练数据，微调（fine-tuning）模型之后直接使用。
9、ERNIE
	百度没有发表论文，项目代码链接。ERNIE是对标BERT的，也是预训练的通用语言模型。BERT学习原始的语言信号，ERNIE直接对先验语义知识单元（比如词，实体以及实体关系等）进行建模。ERNIE在中文任务全面超越了BERT。














二、aspect级别的情绪识别，需要对句子进行细粒度的情感分析，难度和复杂度更高，可参考的论文和方法：
	DMN
论文：Aspect Level Sentiment Classification with Deep Memory Network
参考代码：https://github.com/abhimanu/MemNet4Sentiment
模型：
 
Word embedding:一部分是aspect vec，一部分是context vecs，也是memory。
Computational layer:一部分是attention layer，一部分是linear layer，这两部分求和作为输出。Attention layer的输入时memory，输出是memory中比较重要的部分，linear layer输入就是aspect vec。
Softmax：计算分类概率
	RNN
论文：Adaptive Recursive Neural Network for Target-dependent Twitter Sentiment Classification
代码：
模型：
	Hierarchical DL
论文：Aspect Specific Sentiment Analysis using Hierarchical Deep Learning
	PhraseRNN:
论文：PhraseRNN: Phrase Recursive Neural Network for Aspect-based Sentiment Analysis
	LSTM
论文：Effective LSTMs for Target-Dependent Sentiment Classification
	TNet
论文：Transformation Networks for Target-Oriented Sentiment Classification
代码：https://github.com/lixin4ever/TNet
模型：
 
左侧是整体模型图，整体分三层，自下向上是bi-lstm层，CPT层，卷积层：
Bi-lstm：对句子整体进行建模，获得h_i^0
CPT：调整target的表示，更好的突出语义信息
卷积：与位置相关的卷积层，先根据对象词与上下文的位置信息进行加权，再进行卷积核为1的卷积和maxpooling操作，再通过softmax进行分类
右侧是CPT单元图，由两部分组成，一部分是target调整机制，一部分是上下文信息保存机制：
     Target(TST)：target的embedding经过bi-lstm之后获得h_i^t，然后与上层每个词的h_i^l进行相似度计算，得到的结果r_i^t，将r_i^t和h_i^l拼接起来通过fc层得到输出，送到下一个结构LF/AS。
     Context-preserve(LF/AS)：LF将TST的输出与CPT的输入求和，得到结果传到下一层。AS通过一个门结构来控制哪些有用信息被传入下一层。这两种方法是独立使用的。
  
	可参考的比赛
AI Challennger细粒度用户评论情感分析竞赛


