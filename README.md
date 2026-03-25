# Taiyi
A simple tool for observing the internal dynamics of neural networks.
<h1 align="center">Debug Tools</h1>
<p align="right">作者：DLCV Team




> 这是一个监测网络训练过程中指标的轻量级工具，即插即用，也可以方便的自己编写自己想要观察的指标。
>
> 工具的两个重要概念（维度）是模块和指标，模块指的是pytorch中的神经网络模块例如Linear、Conv2d，指标是指想要观察的指标例如权重范数等等。可以针对单个模块设置多个观察指标，同一个指标可以被多个模块使用（如果计算支持）。
>
> 工具的架构借鉴了软件工程中的三层架构：1.数据准备（通过注册hook获取指标计算需要的数据），2.指标计算（将1中得到的基础数据进行二次加工）3.界面（用户与工具交互的接口）。此外，工具还将指标计算与数据展示解耦，使用者可以根据自己熟悉的可视化工具将计算得到的数据进行展示。

[TOC]

## 1. 使用方法

### 1.1 安装

```bash
// 下载源代码
git clone https://github.com/DLCV-BUAA/Taiyi.git
// 进入下载目录
cd 到下载目录，与setup.py文件同级
// 在安装前可以激活你想要安装的环境后运行安装代码
pip install -e ./  或者 python setup.py develop
```

### 1.2 使用

```python
from Taiyi.taiyi.monitor import Monitor  # 工具接口类
import wandb							 # 展示工具，用wandb展示（可选，计算结果可以通过调用Monitor.get_output()方法获取）
from Taiyi.visualize import Visualization# 可选（可以不用）

"""
编写想要观察模块以及对应的观察指标
基本格式：{
	模块1：[[指标1,对应设置],[指标2,对应设置]],
	模块2：[[指标1,对应设置],[指标2,对应设置]]
}
观察模块写法支持  
1. 'fc1'：model中自己定义的模块名称 
2. 'Conv2d':pytorch中的模块类名称的字符串形式  
3. nn.BatchNorm2d：pytorch中的模块类名称

注意： 2，3方式是模糊搜索，会将所有满足条件的模块都进行观察，例如2是模型中所有的卷积模块都观察

观察指标写法支持 
注意：指标的对应设置（例如 linear(5, 2)）是指每隔多少step（minibatch）计算一次指标，linear是指线性（目前仅支持linear方式），5是指每隔5个step计算一次，2是指从第2 个step开始计数也就是第2、7、12...个step的时候进行计算
1. 'MeanTID':指标的类名称的字符串形式,这种形式默认对应设置是linear(0, 0)
2. ['MeanTID']同1
3. ['MeanTID', 'linear(5, 0)']：使用MeanTID指标，这个指标计算从0开始，每隔5个计算一次
""" 
config = {
        # 'Conv2d': ['InputSndNorm', 'OutputGradSndNorm', 'WeightStatistic'],
        nn.BatchNorm2d: [['MeanTID', 'linear(5,0)'],'InputSndNorm'],
        nn.Linear: [['WeightGradNorm', 'linear(100,0)'], ['WeightGradRange', 'linear(100,0)']]
}
# prepare model
model = Model()
# init monitor
monitor = Monitor(model, config)
# vis = Visualization(monitor, wandb)
step = 0  # 第几个step
for epoch in range(epochs):
    for data in dataloader:
        y = model(data)
        loss.backward()
        ###########################
        monitor.track(step)
        step += 1
        # vis.show(step) 可选
        ###########################
# vis.close()
monitor.get_output() # 可以自己定义save方式，或者在vis.show()方法中自己添加，或者等作者想起来的时候添加
```



## 2.支持的指标

> 指标分为两类一类是singlestep，一类是multistep
>
> singlestep：是指指标计算在一个step内就可以完成的指标
>
> multistep：是指标计算需要多个step结果聚合在一起才能计算完成的指标

### 2.1 Single Step Quantity

| Name                | 描述                       | 实现                                         | Extension                     | cite                                                         |
| ------------------- | -------------------------- | -------------------------------------------- | ----------------------------- | ------------------------------------------------------------ |
| InputCovMaxEig      | 输入协方差矩阵的最大特征值 |1. data = module.input.cov_matrix_eigs<br />2. Max Eig value | ForwardInputEigOfCovExtension | [https://arxiv.org/pdf/2002.10801.pdf](https://arxiv.org/pdf/2207.12598.pdf) |
| InputCovStableRank  | 输入协方差矩阵的稳定秩     |1. data = module.input.cov_matrix_eigs<br />2. Eigs sum /  Max Eig| ForwardInputEigOfCovExtension |                                                              |
| InputCovCondition20 | 输入协方差矩阵的20%条件数  |1. data = module.input.cov_matrix_eigs<br />2. top20% Eig values| ForwardInputEigOfCovExtension | [https://arxiv.org/pdf/2002.10801.pdf](https://arxiv.org/pdf/2207.12598.pdf) |
| InputCovCondition50 | 输入协方差矩阵的50%条件数  |1. data = module.input.cov_matrix_eigs<br />2. top50% Eig values| ForwardInputEigOfCovExtension | [https://arxiv.org/pdf/2002.10801.pdf](https://arxiv.org/pdf/2207.12598.pdf) |
| InputCovCondition80 | 输入协方差矩阵的80%条件数  |1. data = module.input.cov_matrix_eigs<br />2. top80%  Eig values| ForwardInputEigOfCovExtension | [https://arxiv.org/pdf/2002.10801.pdf](https://arxiv.org/pdf/2207.12598.pdf) |
| WeightNorm          | 权重二范数                 | 1. data = module.weight<br />2. norm(2)      |                               |                                                              |
| InputMean           | 输入的每个channel的均值    | 1. data = module.input<br />2. mean          | ForwardInputExtension         |                                                              |
| OutputGradSndNorm   | 输出梯度二范数             | 1. data = module.output_grad<br />2. norm(2) | BackwardOutputExtension       |                                                              |
| InputSndNorm        | 输出二范数                 | 1. data = module.input<br />2. norm(2)       | ForwardInputExtension         |                                                              |

Additional weight-gradient quantities:

| Name | Meaning | Implementation |
| --- | --- | --- |
| `WeightGradNorm` | L2 norm of `module.weight.grad` | `module.weight.grad.norm(2)` |
| `WeightGradRange` | Range summary of `module.weight.grad` | `min / max / abs max` |
| `OutputGradRange` | Range summary of `module.output_grad` | `min / max / abs max` |

### 2.2 Multi Step Quantity

| Name    | 描述                                  | 实现                                                         | Extension             | cite                             |
| ------- | ------------------------------------- | ------------------------------------------------------------ | --------------------- | -------------------------------- |
| MeanTID | BN模块中batch的训练和推理时mean的差异 | 1.data = module.input<br />2.datas = [data.mean]<br />3.diff_data = [d-module.running_mean for d in datas]<br />4. stack diff_data<br />5. diff_data.norm(-1)/(sqrt(running_var).norm(-1))<br />6. mean (思路详见论文公式) | ForwardInputExtension | https://arxiv.org/abs/2210.05153 |
| VarTID  | BN模块中batch的训练和推理时var的差异  | 1. data = input<br />2. datas = [sqrt(data.var(1))]<br />3. sigma = sqrt(running_var)<br />4. diff_data = [d-sigma for d in datas]<br />5. stack diff_data<br />6. diff_data(-1)/(sigma.norm(-1))<br />7. mean(思路详见论文公式) | ForwardInputExtension | https://arxiv.org/abs/2210.05153 |



## 3. 开发者

### 3.1 Extension

> e.g. 计算InputMean指标时需要知道当前模块的输入，但是当前模块获取不到，因此需要通过注册hook先获取input，Extension的作用就是获取input的hook并将获取的input作为当前模块的一个属性。Extension根据获取方式分为forward Extension和backward Extension

#### 3.1.1 Forward Extension

| Name                          | 描述                       | 实现                                                 |
| ----------------------------- | -------------------------- | ---------------------------------------------------- |
| ForwardInputExtension         | 获取模块的输入             | 1.return input[0]                                    |
| ForwardOutputExtension        | 获取模块的输出             | 1.return output[0]                                   |
| ForwardInputEigOfCovExtension | 获取输入协方差矩阵的特征值 | 1.cal_cov_matrix(data)<br />2.retrun cal_eig(matrix) |



#### 3.1.2 Backward Extension

| Name                            | 描述                               | 实现                                                         |
| ------------------------------- | ---------------------------------- | ------------------------------------------------------------ |
| BackwardInputExtension          | 获取模块输入的梯度                 | 1. return grad_input[0]                                      |
| BackwardOutputExtension         | 获取模块输出的梯度                 | 1. return grad_output[0]                                     |
| BackwardOutputEigOfCovExtension | 获取模块输出梯度协方差矩阵的特征值 | 1. cal_cov_matrix(grad_output)<br />2. return cal_eig(matrix) |



#### 3.1.3 utils

| Function       | 描述                 | 实现                              |
| -------------- | -------------------- | --------------------------------- |
| cal_cov_matrix | 计算数据的协方差矩阵 | return torch.cov(data.T)          |
| cal_eig        | 计算数据的特征值     | return torch.linalg.eigvals(data) |

### 3.2 Visualization

> visualize的作用是将monitor的output提取出来,在wandb中进行展示，主要有三个函数
>
> 1. show(step)：将第step个数据提取出来进行展示，可以在这个函数中进行数据的存储
> 2. log_ext()：等价于wandb.log()
> 3. close()：wandb.finish()
>
> 注：monitor的output的数据组织方式
>
> {
>
> ​	module_name:{
>
> ​		quantity:{
>
> ​			step:{
>
> ​				data
>
> ​			}
>
> ​		}
>
> ​	}
>
> }
 
## 4. Weight Gradient Monitoring

Taiyi can monitor per-layer weight-gradient statistics for modules that expose a `weight` parameter, such as `nn.Linear`.

### 4.1 Built-in quantities

| Name | Meaning | Output fields |
| --- | --- | --- |
| `WeightGradNorm` | L2 norm of `module.weight.grad` | scalar |
| `WeightGradRange` | Range summary of `module.weight.grad` | `min`, `max`, `abs_max` |

### 4.2 Example config

```python
config = {
    "Linear": [
        ["OutputGradRange", "linear(100,0)"],
        ["WeightGradNorm", "linear(100,0)"],
        ["WeightGradRange", "linear(100,0)"],
    ],
}
monitor = Monitor(model, config)
```

### 4.3 Notes

- These quantities are only available after `loss.backward()`.
- If `module.weight.grad` is `None`, Taiyi skips the current step.
- If `module.output_grad` is `None`, `OutputGradRange` skips the current step.
- `WeightGradRange` skips logging when the gradient tensor contains non-finite values.

## 5. Residual Angle Monitoring

Taiyi now supports residual-angle monitoring for residual blocks that expose a unified `residual_states` field.

### 5.1 Built-in quantities

| Name | Meaning | Required state keys |
| --- | --- | --- |
| `ResidualInputAngleMean` | Mean angle between the two inputs of a residual add | `stream`, `branch` |
| `ResidualInputAngleStd` | Std of the angle between the two inputs of a residual add | `stream`, `branch` |
| `ResidualStreamOutputAngleMean` | Mean angle between the residual stream and the residual output | `stream`, `output` |
| `ResidualStreamOutputAngleStd` | Std of the angle between the residual stream and the residual output | `stream`, `output` |
| `ResidualEnergyRatio` | Mean residual-branch energy ratio `||branch||^2 / (||stream||^2 + ||branch||^2)` | `stream`, `branch` |

Residual quantities can be grouped as:

| Category | Quantities | What they measure |
| --- | --- | --- |
| Direction | `ResidualInputAngleMean/Std` | Relative direction between skip stream and residual branch |
| Direction | `ResidualStreamOutputAngleMean/Std` | Direction change from skip stream to final residual output |
| Energy | `ResidualEnergyRatio` | Relative magnitude contribution of residual branch versus skip stream |

Definitions:

- `stream`: the skip/residual-flow tensor before addition.
- `branch`: the main branch tensor to be added to the stream.
- `output`: the tensor after the residual add, and after any post-add transform if you choose to monitor that final tensor.

### 5.2 Example config

```python
config = {
    "Block": [
        ["ResidualInputAngleMean", "linear(5,0)"],
        ["ResidualStreamOutputAngleMean", "linear(5,0)"],
    ],
}
monitor = Monitor(model, config)
```

### 5.3 How residual monitoring works

Taiyi does not infer residual structure automatically from arbitrary Python code. Instead, the monitored module must expose:

```python
self.residual_states = {
    "default": {
        "stream": stream_tensor,
        "branch": branch_tensor,
        "output": output_tensor,
    }
}
```

If a module contains multiple residual adds in one forward, expose multiple named entries:

```python
self.residual_states = {
    "attn": {
        "stream": attn_stream,
        "branch": attn_branch,
        "output": attn_output,
    },
    "mlp": {
        "stream": mlp_stream,
        "branch": mlp_branch,
        "output": mlp_output,
    },
}
```

Each quantity will compute angles independently for each entry and log them as separate metrics.

### 5.4 How to make a new residual model detectable by Taiyi

When you implement a new residual block, write the forward so the three tensors are explicit:

```python
class MyResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.act = nn.ReLU()

    def forward(self, x):
        stream = x
        branch = self.act(self.fc(x))
        output = stream + branch

        self.residual_states = {
            "default": {
                "stream": stream,
                "branch": branch,
                "output": output,
            }
        }
        return output
```

Then register it in Taiyi config by module name or module type:

```python
config = {
    "MyResidualBlock": [
        "ResidualInputAngleMean",
        "ResidualStreamOutputAngleMean",
    ]
}
```

Rules:

- `stream`, `branch`, and `output` should be tensors with the same batch dimension.
- Taiyi flattens each sample to a vector and computes sample-wise cosine angle.
- The logged `Mean` / `Std` is computed across the current batch.
- If you want to monitor the angle before a post-add activation, set `output` to the raw sum.
- If you want to monitor the angle after a post-add activation/norm, set `output` to that final tensor.

### 5.5 Compatibility checker

You can scan a model for Taiyi residual-angle compatibility with:

```python
from Taiyi.utils import check_residual_compatibility

report = check_residual_compatibility(model, sample_inputs=x)
print(report["summary"])
print(report["compatible"])
print(report["incompatible"])
```

Notes:

- If `sample_inputs` is provided, Taiyi will run one forward pass first, so runtime-created `residual_states` can be detected.
- If `sample_inputs` is omitted, the checker only inspects modules as they are currently stored in memory.
