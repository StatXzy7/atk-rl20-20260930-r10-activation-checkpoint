# 分段激活 Checkpointing

这是一个不依赖 `torch.utils.checkpoint` 的小型顺序网络实现。网络层限定为
`Linear`、无状态非线性激活和 `Dropout`，核心分段重算在 `checkpointing.py` 的
`_SegmentCheckpoint` 中直接实现。

## 用法

```python
import torch
from torch import nn
from checkpointing import CheckpointedSequential

model = CheckpointedSequential(
    nn.Linear(32, 64),
    nn.Tanh(),
    nn.Dropout(0.1),
    nn.Linear(64, 16),
    boundaries=(2,),
)
x = torch.randn(8, 32, requires_grad=True)
model(x).sum().backward()
```

`boundaries` 支持：

- 整数：均匀分段数，例如 `boundaries=3`；
- cut 元组/列表：层索引切点，例如 25 层传 `(9, 15, 21, 24)` 得到 5 个非空段；
- 单段：`boundaries=1` 只保存整个网络输入和该段起点 RNG 状态。

## 数学

设网络分为 \(S\) 段，第 \(i\) 段为 \(f_i\)，段输入为 \(a_{i-1}\)，参数为
\(\theta_i\)，则

\[
y = f_S(\theta_S, \cdot)\circ\cdots\circ f_1(\theta_1, a_0).
\]

前向在每段外构建一阶 autograd 节点，但段内运行于 `no_grad`，只持久化边界
\(a_{i-1}\) 和该段开始时的 RNG 状态。反向到第 \(i\) 段时：

1. 恢复段起点 RNG；
2. 从保存的 \(a_{i-1}\) 重新执行 \(f_i\)，重新得到 dropout mask 和局部计算图；
3. 对 \(a_{i-1}\) 和该段参数调用一次局部 `torch.autograd.grad`；
4. 把输入梯度交回前一段节点，参数梯度由自定义 Function 的参数返回值累加到模型参数。

对损失 \(L\)，局部计算使用

\[
\frac{\partial L}{\partial a_{i-1}}
= \left(\frac{\partial a_i}{\partial a_{i-1}}\right)^\top
  \frac{\partial L}{\partial a_i},
\qquad
\frac{\partial L}{\partial \theta_i}
= \left(\frac{\partial a_i}{\partial \theta_i}\right)^\top
  \frac{\partial L}{\partial a_i}.
\]

dropout 是分段前向的一部分。每段捕获前向实际使用的全局 RNG 状态，反向重放时使用
同一状态，因此 mask 与未 checkpoint 的参照前向完全一致。本实现不关闭 dropout，
也不缓存 dropout 输出或段输出。

## RNG 生命周期

对每个段：

- checkpoint 前向开始时捕获相关 CPU/CUDA generator 状态；
- checkpoint 前向正常消耗全局 RNG，因此调用者在整个前向后的 RNG 位置与参照相同；
- 反向开始时先保存调用者当前 RNG；
- 重放期间恢复为前向段起点 RNG；
- 无论重放是否成功，`finally` 都恢复反向开始时的 RNG。

因此重算消耗的随机数不会泄漏到调用者后续抽样中：前向后、反向后的继续抽样序列都与
不启用 checkpoint 的参照一致。对应测试是
`test_rng_sequence_is_unchanged_after_forward_and_backward`。

## 保存量与额外计算

设非 checkpoint 网络会为反向保存 \(M\) 的中间激活：

- checkpoint 额外前向计算约等于完整前向一遍，即总前向类计算约 \(2F\)；
- 持久边界张量约为 \(S\) 个段输入，不保存各段内部激活；
- 每个相关设备的每个段额外保存一份 RNG 状态；
- 分段越少，持久保存越少，但单段重放产生的瞬时局部激活峰值可能更大；
- 分段越多，边界保存量近似随段数增加，但每段重放更短、瞬时局部工作区更小。

真实峰值还包括参数、梯度、分配器缓存、内核工作区和段内瞬时重算张量，不能用
“保存张量字节数”直接冒充设备峰值显存。

内存代理统计分为：

- `logical_bytes`：每个张量引用可见元素的字节数之和，重复引用会重复计数；
- `referenced_storage_bytes`：每次引用指向的 backing storage 大小之和；
- `unique_storage_bytes`：按 `(device, data_ptr)` 去重后的 backing storage 字节数；
- `repeated_reference_bytes`：前两项差值，用于暴露共享 storage 上的重复引用；
- activation、parameter、rng state 分开报告。参数由模型本身持有，统计为引用而不是
  checkpoint 新复制出的参数负载。

## 支持与拒绝

明确支持：

- 一个标量损失上的一次一阶反向传播；
- `Linear`、允许名单内的无状态激活和 `nn.Dropout`；
- 输入需要或不需要梯度；
- 同一个模块/参数在同一网络中重复使用；
- CPU 和标准 CUDA generator。

明确拒绝或不支持：

- 高阶梯度、`create_graph=True`、双反向；自定义 backward 检测到梯度模式时抛出
  `RuntimeError`；
- `BatchNorm` 等带状态更新的层、未知自定义层、I/O 或其他副作用层；构造时抛出
  `ValueError`；
- `inplace=True` 激活和层内输入原地修改；未知层构造时拒绝，允许层不做原地操作；
- 用户自定义 `torch.Generator` 传入 dropout；本实现只恢复全局 PyTorch RNG；
- 多输出或分支网络；本类只表达单张量顺序网络。

## 测试

运行：

```bash
pytest -q
```

测试覆盖单段、多段、不均匀分段、重复模块参数、输入需要/不需要梯度、边界校验、
副作用层拒绝、原地层拒绝、RNG 前向/反向后序列、共享 storage 统计，以及确定性
Linear/Tanh/Linear 子问题的双精度 `torch.autograd.gradcheck`。带 dropout 的数值
正确性通过固定全局 RNG 与参照实现逐项 `torch.equal` 比较输出和梯度。

## 本机测量

运行：

```bash
python measure.py
```

本次环境是 Python 3.11、PyTorch 2.14.0+cpu、CUDA 不可用，因此没有真实设备显存
峰值可报；表格中的 `real device peak` 标记为 `n/a (CPU)`。CPU 墙钟时间仅代表此
环境，不预设 checkpoint 更快。

| model | payload logical B | unique storage B | repeated backing B | parameter refs B | wall s |
| --- | ---: | ---: | ---: | ---: | ---: |
| reference | 5,144,576 | 5,144,576 | 0 | 0 | 0.016482 |
| 1 segment | 136,128 | 136,128 | 0 | 2,138,240 | 0.025377 |
| 2 segments | 272,256 | 272,256 | 0 | 2,138,240 | 0.025437 |
| 4 segments | 544,512 | 544,512 | 0 | 2,138,240 | 0.027452 |
| 8 segments | 1,089,024 | 1,089,024 | 0 | 2,138,240 | 0.030523 |
| uneven cuts=(9,15,21,24) | 680,640 | 680,640 | 0 | 2,138,240 | 0.027230 |

单段 checkpoint 的 payload 为输入 `128*256*4=131072` 字节加 CPU RNG 状态
`5056` 字节。参照网络的代理值来自 `saved_tensors_hooks` 实际观测；checkpoint 值
来自活动 checkpoint 上下文的真实边界张量和 RNG 张量。
