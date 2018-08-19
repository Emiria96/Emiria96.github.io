---
layout: post
title: 推荐系统（一）：Factorization Machine
author: Scalsol
category: [ rs ]
---

因为某些契机了解到了Factorization Machine，就以此为切入点来了解一下推荐系统的一些模型吧。  
**原始论文：**[Factorization Machine](https://cseweb.ucsd.edu/classes/fa17/cse291-b/reading/Rendle2010FM.pdf)  
<br />
### 1. 什么是Factorization Machine?  
FM是用于解决当自变量非常稀疏时的一类预测问题的，它通过把计算不同特征的特征权值并通过对权值进行内积操作来达到捕捉高维依赖的效果，按照论文中所述，FM具有以下优良性质：
1. 在自变量非常稀疏的时候，FM能够对参数进行估计，而与此同时SVM则会失败。
2. FM的计算具有线性复杂度，并且可以只通过Primal形式来进行求解，并且不像SVM那样依赖于支持向量。在Netflix这样的大数据集中，其仍然有优良表现。
3. FM是一个非常广义的预测模型，可以用于任意实值特征向量的情形。并且其余目前表现最优异，但都有其局限性的模型都可以通过设计输入向量的形式来进行模拟。比如MF, SVD++, PITF, FPMC.

### 2. 模型形式
普通的线性预测模型具有以下形式：  
$$\hat{y}(x):=w_0+\sum\limits_{i=1}^n w_ix_i$$

加入对高阶项的依赖，比如多项式SVM，将其形式不是非常严谨的写成以下形式：
$$\hat{y}(x):=w_0+\sum\limits_{i=1}^n w_ix_i + \sum\limits_{i=1}^n\sum\limits_{j=i+1}^n w_{ij} x_i x_j$$

对线性模型进行扩展，加入对高阶项的依赖，则得到二阶的FM的模型描述如下：  
$$\hat{y}(x):=w_0+\sum\limits_{i=1}^n w_ix_i + \sum\limits_{i=1}^n\sum\limits_{j=i+1}^n \langle v_i,v_j \rangle x_i x_j$$

这之中，$w_0$, $\textbf{w}$, $\textbf{V}$都是需要估计的参数。

简单解释一下为什么这样的模型就能够一定程度解决数据稀疏带来的问题。因为对于每一个特征$x_i$，只要有足够多的数据中这项特征出现了，那么就可以对其参数$v_i$进行估计。而与此同时，简单的二阶SVM则不能在数据量少，特征稀疏的情况下进行有效的训练。比如若$x_i$和$x_j$都并未同时出现时，我们无法得到$w_{ij}$这一项。所以FM在我看来，实际上是通过减小模型的capacity来达到解除依赖的效果，使得参数的估计可以单独进行。  
<br />
### 3. 模型使用
简单的通过FM的原表达式来计算预测值是平方阶的，然而通过一系列变换可以将其改进为线性。
$$\sum\limits_{i=1}^n\sum\limits_{j=i+1}^n \langle v_i,v_j \rangle x_i x_j=\frac{1}{2}\sum_{f=1}^k \left(\left(\sum\limits_{i=1}^n v_{i,f}x_i\right)^2-\sum\limits_{i=1}^n v_{i,f}^2x_i^2\right)$$  
<br />
### 4. 模型训练
通过SGD就可以对参数进行训练。给定损失函数即可。$\hat{y}(x)$对于各项参数的偏导列表如下：  
$$\frac{\partial}{\partial \theta}\hat{y}(x)=
\left\{
	\begin{aligned}
		&1&\text{if }\theta \text{ is } w_0\\
		&x_i&\text{if }\theta \text{ is } w_i\\
		&x_i\sum_{j=1}^n v_{j,f}x_j-v_{i,f}x_i^2&\text{if }\theta \text{ is } v_{i,f}
	\end{aligned}
\right.
$$

在libFM中则给出了其他许多种训练方法，比如MCMC等。  
<br />
### 5. 加入更高阶的依赖项
此时模型可以写成  
$$\hat{y}(x):=w_0+\sum\limits_{i=1}^n w_ix_i + 
			\sum\limits_{l=2}^d\sum\limits_{i_1=1}^n\cdots\sum\limits_{i_l=i_{l-1}+1}^n\left(\prod\limits_{j=1}^l x_{i_j}\right)\left(\sum\limits_{f=1}^{k_l}\prod\limits_{j=1}^lv_{i_j,f}^{(l)}\right)$$
			
<br />		
### 6. 模型实战
了解了基本原理，那么接下来就是要把它用到实际任务中去。这里我采用的数据集是[MovieLens100k Dataset](https://grouplens.org/datasets/movielens/100k/)。数据包含4列，分别是用户ID，电影ID，打分，时间。为了使用FM，我们首先需要将每条数据处理成一个稀疏向量。设n, m分别是用户数和电影数，则我们生成的向量中，前n列是代表用户的one-hot编码，后m列是代表电影的one-hot编码。实际上可以再增加一组变量代表上一次评分的电影但是这里为了简单就不加进去了。处理方法使用了`scipy.sparse.csr.csr_matrix`这个函数。
{% highlight py %}
def vectorize_dic(dic, ix=None, p=None, n=0, g=0):
    """
    dic -- dictionary of feature lists. Keys are the name of features
    ix -- index generator (default None)
    p -- dimension of featrure space (number of columns in the sparse matrix) (default None)
    """
    if ix is None:
        ix = dict()

    nz = n * g

    col_ix = np.empty(nz, dtype=int)

    i = 0
    sums = np.zeros(g)

    for k, lis in dic.items():
        for t in range(len(lis)):
            ix[str(lis[t]) + str(k)] = ix.get(str(lis[t]) + str(k), sums[i])
            if ix[str(lis[t]) + str(k)] == sums[i]:
                sums[i] += 1

            col_ix[i + t * g] = ix[str(lis[t]) + str(k)] + (i == 1) * sums[0]
        i += 1
    row_ix = np.repeat(np.arange(0, n), g)

    data = np.ones(nz)
    if p is None:
        p = len(ix)

    ixx = np.where(col_ix < p)
    return csr.csr_matrix((data[ixx], (row_ix[ixx], col_ix[ixx])), shape=(n, p)), ix
	
train = pd.read_csv(os.path.join(args.data_dir, 'ua.base'), delimiter='\t', names=names)
test = pd.read_csv(os.path.join(args.data_dir, 'ua.test'), delimiter='\t', names=names)

x_train, ix = vectorize_dic({'users': train['user'].values,
                             'items': train['item'].values}, n=len(train.index), g=2)

x_test, ix = vectorize_dic({'users': test['user'].values,
                            'items': test['item'].values}, ix, x_train.shape[1], n=len(test.index), g=2)

x_train = x_train.todense()
x_test = x_test.todense()
y_train = train['rating'].values
y_test = test['rating'].values
{% endhighlight %}
这样子`x_train`和`x_test`就是代表用户和电影的稀疏特征向量了。接下来为了方便，我使用pytorch构建了整个模型（不想写反向传播），模型代码如下：
{% highlight py %}
class FM(nn.Module):
    def __init__(self, k, p):
        super(FM, self).__init__()
        self.k = k
        self.p = p

        self.w0 = nn.Parameter(torch.zeros(1, dtype=torch.float))
        self.W = nn.Parameter(torch.zeros(self.p, dtype=torch.float))
        self.V = nn.Parameter(0.01 * torch.randn((p, k), dtype=torch.float))

    def forward(self, x):
        linear_terms = self.w0 + torch.matmul(x, self.W)
        pair_interactions = 0.5 * ((torch.matmul(x, self.V)
                                    - torch.matmul(torch.pow(x, 2), torch.pow(self.V, 2)))).sum(1)

        return linear_terms + pair_interactions
{% endhighlight %}
训练使用了带动量的SGD和带正则项的L2损失函数。最后得到的MSE误差约为0.96，也就是离正确值基本上相差1，并不是令人满意的结果。但是作为练手，也当我真的了解了FM这个模型了吧。