# 利用Pytorch写一个简单的多层神经网络

------

首先我们参考pytorch 提供的一个example code
https://github.com/pytorch/examples/blob/master/mnist/main.py
在这个仓库里面还有很多其他的代码值得我们去学习与借鉴

在这里我们就编写一个最简单的多层神经网络，并在MNIST上面测试我们的模型

## 原始代码解析

### 数据加载
pytorch内置就有MNIST数据集的相关接口
```python
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)
```
### GPU加速
当我们需要GPU加速的时候，需要把模型的参数，以及运算中涉及的Tensor放到GPU里面
```python
model = Net()
if args.cuda:
    model.cuda()
    
...

        if args.cuda:
            data, target = data.cuda(), target.cuda()  # 注意这里和模型的.cuda()用法有些不同！
```
### 训练过程
```python
def train(epoch):
    model.train()  # 训练函数的初始化，后面会详细介绍
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target) # 把输入的input和label 转化为Variable
        optimizer.zero_grad() # 这句话必须加
        output = model(data) # 获取模型的output
        loss = F.nll_loss(output, target) # 计算损失函数
        loss.backward() # 反向传播
        optimizer.step() # 更新梯度
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0])) # 打印log

def test():
    model.eval() # 设置model的状态为训练模式，后面会详细介绍
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum() # 当GPU的数据与CPU的数据运算时需要先放到CPU里面在参与运算 .cpu()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, args.epochs + 1): # 设定训练轮数
    train(epoch)
    test()
```
### 定义模型

好了，最重要的部分来了，由于pytroch自带反向传播以及梯度计算功能，所以我们很多情况下只需要定义模型的参数以及正向的传播过程就可以了
```python
class Net(nn.Module): # 定义的模型是torch.nn.Module的子类
    def __init__(self):
        super(Net, self).__init__() # 调用父类的构造函数
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x): # 神经网络的正向计算
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
```

我们可以看到我们定义的网络都是torch.nn.Module的子类
回忆上面我们在模型训练和评价前会调用model.eval()和model.train()这两个函数
那么这两个函数有什么用呢？
我们知道dropout是用于神经网络中防止过拟合的一个很有效的方法，我们一般会在训练的过程中加入dropout，但是在评价模型的时候我们不希望加入dropout，因为这样在评价的时候对结果会有很大的随机性！
所以我们可以看到在forward的过程中有这样一个函数 F.dropout(x, training=self.training)，中间把self.training作为一个参数
那么我们可以看看torch.nn.Module的定义：
我们找到了train和eval这两个方法
```python
    def train(self, mode=True):
        """Sets the module in training mode.

        This has any effect only on modules such as Dropout or BatchNorm.

        Returns:
            Module: self
        """
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def eval(self):
        """Sets the module in evaluation mode.

        This has any effect only on modules such as Dropout or BatchNorm.
        """
        return self.train(False)
```

我们可以看到self.traing 表示模型是处于训练中或者评价中，那么我们就可以通过这种方式让model在训练的时候dropout生效，而在评价过程中无效
所以当我们在自己写模型的时候需要一些初始化的时候可以重载这个方法

## 修改代码为一个两层的神经网络

我们发现，其实我们只需要修改model的构造函数以及forward这两个函数就可以了
需要注意的是：
在forward过程中，x的size是batch_size * in_channel * 28 * 28的，因为输入的数据是原先给cnn用的，所以是一个二维的信息
所以我们需要用x = x.view(x.size(0), -1)修改一下x的大小
```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear_0 = nn.Linear(28 * 28, 100)
        self.linear_1 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = self.linear_0(x)
        x = self.linear_1(x)
        return F.log_softmax(x)
```

## 修改一个有初始参数设置的多层神经网络

向刚才的方法，每次调整参数都需要直接修改代码，那么我们想可不可以直接把参数传入模型的构造函数里，这样我们就可以更加方便的调整参数了
想了2秒钟后就开始写了下面这个代码（写了不止2s 233）

```python
class Net(nn.Module):
    def __init__(self, layer_size): # 比如这里初始参数可以是[28 * 28, 100, 10], [28 * 28, 200, 100, 50, 10]等等
        super(Net, self).__init__()
        
        self.linear_list = [nn.Linear(layer_size[i], layer_size[i + 1]) for i in range(len(layer_size) - 1)] # 用一个列表保存我们的这些线性层

        self.layer_count = len(layer_size) - 1

    def forward(self, x):
        x = x.view(x.size(0), -1)

        for linear in self.linear_list:
            x = linear(x)

        return F.log_softmax(x)
```

然后运行，发现有问题
根据debug的信息，我们的模型没有参数

为了解决这个问题我们可以看一下pytorch.nn.Parameters 的文档http://pytorch.org/docs/0.3.0/nn.html#torch.nn.Module

Parameters are Variable subclasses, that have a very special property when used with Module s - when they’re assigned as Module attributes they are automatically added to the list of its parameters, and will appear e.g. in parameters() iterator. Assigning a Variable doesn’t have such effect. This is because one might want to cache some temporary state, like last hidden state of the RNN, in the model. If there was no such class as Parameter, these temporaries would get registered too.

发现我们在定义模型的时候，子模型或者参数必须以成员的形式，而不能以一个包含参数的列表的形式！

我们再看看pytorch.nn.Module的\__setattr__方法
关于\__setarrt__是一个什么东西：https://docs.python.org/2/reference/datamodel.html#object.\__setattr__


```python
    def __setattr__(self, name, value):
        def remove_from(*dicts):
            for d in dicts:
                if name in d:
                    del d[name]
        params = self.__dict__.get('_parameters')
        if isinstance(value, Parameter):
            if params is None:
                raise AttributeError(
                    "cannot assign parameters before Module.__init__() call")
            remove_from(self.__dict__, self._buffers, self._modules)
            self.register_parameter(name, value)
        elif params is not None and name in params:
            if value is not None:
                raise TypeError("cannot assign '{}' as parameter '{}' "
                                "(torch.nn.Parameter or None expected)"
                                .format(torch.typename(value), name))
            self.register_parameter(name, value)
        else:
            modules = self.__dict__.get('_modules')
            if isinstance(value, Module):
                if modules is None:
                    raise AttributeError(
                        "cannot assign module before Module.__init__() call")
                remove_from(self.__dict__, self._parameters, self._buffers)
                modules[name] = value
            elif modules is not None and name in modules:
                if value is not None:
                    raise TypeError("cannot assign '{}' as child module '{}' "
                                    "(torch.nn.Module or None expected)"
                                    .format(torch.typename(value), name))
                modules[name] = value
            else:
                buffers = self.__dict__.get('_buffers')
                if buffers is not None and name in buffers:
                    if value is not None and not torch.is_tensor(value):
                        raise TypeError("cannot assign '{}' as buffer '{}' "
                                        "(torch.Tensor or None expected)"
                                        .format(torch.typename(value), name))
                    buffers[name] = value
                else:
                    object.__setattr__(self, name, value)
```

我们可以看到，原来我刚刚那么写是有问题的

然后我们可以这样写

```python
class Net(nn.Module):
    def __init__(self, layer_size):
        super(Net, self).__init__()

        for i in range(len(layer_size) - 1):
            self.__setattr__("linear_{}".format(i), nn.Linear(layer_size[i], layer_size[i + 1]))
        self.layer_count = len(layer_size) - 1

    def forward(self, x):
        x = x.view(x.size(0), -1)

        for i in range(self.layer_count):
            x = self.__getattr__("linear_{}".format(i))(x)

        return F.log_softmax(x)
```

完事了，有时间我把视频给减一减，压缩压缩再传上去
