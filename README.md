# 小型顺序网络的分段激活 Checkpointing（从零实现）

不调用 `torch.utils.checkpoint`，核心重算逻辑用一个自定义
`torch.autograd.Function`（`mini_checkpoint/checkpoint.py` 中的
`_CheckpointSegment`）实现。网络由 `Linear`、非线性激活和 `Dropout` 组成，
整网只做**一次** `backward()`。

## 目录

- `mini_checkpoint/checkpoint.py` — 核心：逐段保存边界 + 反向重算
- `mini_checkpoint/network.py` — `CheckpointedSequential` 与分段边界解析
- `mini_checkpoint/rng.py` — CPU/CUDA 全局 RNG 状态快照/恢复
- `mini_checkpoint/stats.py` — 真实张量的内存代理（区分 storage 与引用）
- `mini_checkpoint/benchmark.py` — 保存量、模块调用次数、计时、真实峰值（若设备允许）
- `mini_checkpoint/reference.py` — 构造对照模型的小工具
- `tests/test_checkpoint.py` — 全部正确性测试
- `README.md` — 本文档（数学与生命周期）

运行测试：`python -m pytest tests/ -q`
运行测量：`python -m mini_checkpoint.benchmark --d 256 --depth 6`

## 分段边界怎么指定

`CheckpointedSequential(*modules, segments=...)`：

- `segments=k`（整数）：把模块列表尽量均匀地切成 `k` 段（余数由前向后分配）。
- `segments=[c1, c2, ...]`：在模块 `c_i` **之前**插入边界（内部切点，严格递增）。
  例如 10 个模块给 `[2, 3, 9]` 得到 `[0:2) [2:3) [3:9) [9:10)`，用于不均匀分段。
- 边界只能落在模块之间；非法取值（0、越界、重复、非递增、非 int）会抛错。

## 数学

把整网写成复合函数，按边界切成 k 段：

```
h_0 = x
h_s = f_s(h_{s-1}; θ_s),   s = 1..k
L = ℓ(h_k)
```

普通训练时 autograd 为反向保存每一层的中间激活。Checkpoint 只保留边界
`h_0, ..., h_{k-1}`（共 k 个；`h_0=x` 调用方本来就持有），段内中间量全部丢弃。

反向时，对第 s 段，已知上游梯度 `ḡ_s = ∂L/∂h_s`：

1. 用保存的 `h_{s-1}` 在 `enable_grad` 下重跑 `f_s`，临时重建段内计算图
   `h_s = f_s(h_{s-1}; θ_s)`；
2. 对这张局部图调用一次 `torch.autograd.grad`：
   `∂L/∂h_{s-1}, ∂L/∂θ_s = autograd.grad(h_s, (h_{s-1}, θ_s), grad_outputs=ḡ_s)`；
3. 丢弃临时图；`∂L/∂h_{s-1}` 作为下一段（s-1）的上游梯度继续传播。

**段间接续（关键细节）**：传给 `apply` 的是**未 detach** 的原始边界
（保持整网 autograd 图连通）；只有在反向**重算时**，才对边界做一次
`detach()`（保留 `requires_grad`）并在该 detach 叶子上重建本段局部图。这样：

- 每个段的重算图彼此独立，同一权重被多个段复用时，不会“反向穿过一张已释放的图”；
- 局部图对 detach 叶子求出的 `∂L/∂h_{s-1}`，由 autograd 引擎接到
  “产生该边界的上一个段 Function 节点”继续传播，因此输入梯度、链式梯度不丢；
- 第一段的外部输入从不 detach，`.grad` 能到达调用方的输入叶子。

这与 `torch.utils.checkpoint`（reentrant）的 `detach_variable` 机制一致，但本实现
未调用它。

权重 θ 通过 `Function.apply` 的普通张量入参传入（段内按张量 id 去重），
因此参数梯度走 autograd 标准累加器：**同一参数被多段/多次复用时，.grad
是各段贡献之和**，与不开 checkpoint 完全一致（见
`test_shared_module_parameters`）。

对 dropout：`Dropout` 的前向是 `y_i = m_i * a_i / (1-p)`，其中 mask
`m_i∈{0,1}` 来自全局 RNG。反向是 `∂L/∂a_i = m_i/(1-p) · ∂L/∂y_i`。
重算必须用**前向当时那个 mask**，否则梯度错误。这通过恢复前向前的 RNG
状态保证（见下）。

## 保存量与额外计算（分段数量的影响）

设每个边界激活字节数为 a（等宽网络近似常数），整网一次前向的模块计算量为 F：

- 保存的边界激活：**k 个**，约 `k·a` 字节。k=1 最少（只存输入），k 越大
  保存越多；k 等于模块数时退化为几乎全保存。
- 额外计算：反向时每段恰好重算一次，**与 k 和切点位置无关，恒为约 +1 个
  完整前向 F**（前向 F + 反向重算 F，模块调用数 2 倍；见 benchmark 的
  `calls fwd+bwd` 列）。
- 切点不均匀只改变“每段临时重算峰值”的形状，不改变总保存的边界个数（仍 k 个）
  和总重算量（仍 +F）。

即：段越少越省激活内存、重算量不变；实践中按显存预算和可接受重算来选 k。

实测（CPU，`d=256, depth=6, batch=64, float32`，见 `benchmark.py`）：

```
  k  saved_refs saved_stor  saved_MiB  calls(fwd+bwd)  best_s   peak_MiB
  1       1          1       0.062         38        0.0144      N/A
  2       2          2       0.125         38        0.0149      N/A
  4       4          4       0.250         38        0.0160      N/A
  6       6          6       0.375         38        0.0173      N/A
 19      19         13       0.812         38        0.0210      N/A
eager   19         13       0.812         19        0.0101      N/A
```

- 保存量随 k 线性增长；k=19（逐模块）时引用 19 个但只有 13 个不同
  storage——eval 模式下 `Dropout` 直接透传输入（共享 storage）。这正是
  “重复引用/视图 vs 独立 storage”必须分开计数的实证。
- checkpoint 各档模块调用 38=2×19，eager 19；额外约一整个前向。
- CPU 上 checkpoint 更慢；**不把“更快”作为预设结论**。

## RNG 生命周期（为什么流不会多走或少走）

每个段在 forward 前记录 `S_before_s = get_rng_state()`。关键约定：

- **Forward**：正常执行，dropout 照常推进全局 RNG（和 eager 一样）。因此
  forward 结束时，全局流的位置与“不开 checkpoint 的 forward”**完全相同**，
  没有任何额外消耗。
- **Backward（进入点）**：记录 `S_bwd_start`（= eager 反向开始时的流位置，
  因为 forward 已把流推进到同一位置；Linear/Tanh 反向不抽随机数）。
- **重算第 s 段前**：`set_rng_state(S_before_s)`，使该段 dropout 抽到与
  forward 完全一致的 mask。
- **该段重算结束后**：无条件 `set_rng_state(S_bwd_start)`。重算消耗的随机数
  被“抵消”，全局流恢复到反向开始处。
- 下一段重复同样的“恢复到段前→重算→恢复到反向起点”。整个 backward 结束后，
  全局流位置 = eager 整次 forward+backward 后的位置。

结论：调用方在 forward 前、forward 与 backward 之间、backward 之后继续抽样，
两条路径（eager 与 checkpoint）逐比特一致。对应测试：

- `test_rng_stream_position_matches_eager`（整次 fwd+bwd 后流位置相同）
- `test_subsequent_draws_identical_after_backward`（backward 后继续抽 1000 个一致）
- `test_interleaved_backward_draws_are_consumed_like_eager`
  （forward 与 backward **之间**插入抽样，偏移量也一致）
- `test_recompute_uses_forward_mask_not_new_mask`（重算用的是前向 mask）

不是靠关闭 dropout 或缓存输出：forward 真实执行 dropout，重算在恢复的 RNG
状态下重新生成同一个 mask；段输出从不被保存。

`preserve_rng_state=False` 可用于完全无随机性的段（确定性），跳过快照/恢复。

## 内存代理指标是什么、不是什么

`tensor_mem_stats`（`stats.py`）对“真正保存的张量”统计：

- `num_refs`：张量对象（引用）个数；
- `num_storages`：按 `(device, data_ptr, nbytes)` 去重后的底层 storage 个数；
- `unique_bytes`：不同 storage 的字节数之和（共享 storage 只计一次）；
- `ref_bytes`：按每个引用 `numel*element_size` 朴素相加（视图会重复计）；
- `num_views`：与某个已统计张量共享 storage 的引用数。

它只数“显式保存的张量”，**不能冒充设备峰值显存**：不含分配器碎片/预留池、
autograd 元数据、工作区、临时 buffer 等。真实峰值另行报告：

- 有 CUDA 时 `benchmark.py` 的 `peak_MiB` 列直接输出
  `torch.cuda.max_memory_allocated()`（每轮 reset）；本机为 CPU-only 构建
  （`torch.cuda.is_available()==False`），故该列为 N/A，无设备峰值可测。
- CPU 构建下只报告 OS 进程 `ru_maxrss`（约 279 MiB，包含一切，非激活保存量），
  并明确标注不可与设备峰值类比。

## 明确支持与不支持

支持：

- 由 Linear / 激活 / Dropout 等**纯函数式**（输出只依赖输入、参数与全局 RNG）
  模块构成的顺序网络；
- 单段、多段、不均匀切点、同一模块实例重复使用（参数共享）、输入需要或不需要
  梯度、部分参数冻结（`requires_grad=False`）；
- CPU 与已初始化的 CUDA 设备的 RNG 恢复（本机无 CUDA，该路径为防御性实现）。

不支持 / 已界定：

- **高阶梯度（double backward / `create_graph=True`）**：backward 重建的局部图
  在返回后即销毁。运行时检测到 backward 处于 grad-enabled 状态会抛
  `NotImplementedError("higher-order ...")`（`test_higher_order_rejected`）。
- **任意副作用层**：除“使用全局 RNG”外有副作用的模块会在重算时**再执行一次**。
  我们不假装它只执行一次；计数器测试 `test_side_effect_layer_runs_twice_
  documented_boundary` 明确这一行为。计数器、累加器、BatchNorm 的 running
  stats、打印/IO 等属于此类，不应放进需要精确一次语义的位置。
- **原地修改输入/共享输入**：段运行器原地改输入、或后续模块原地改已保存边界，
  会被 `_version` 检查在 forward/backward 抛 `RuntimeError`
  （`test_inplace_segment_input_rejected`）。注意：对段内**新建**张量做原地
  激活（如内部 `relu_()`）是安全的，因为该张量会被一致地重建
  （`test_inplace_internal_activation_is_safe`）。

## 验证策略

- **无随机子问题用双精度 `gradcheck`**：对确定性的 Linear+Tanh 子段（单段、
  以及两段链式）做 `torch.autograd.gradcheck`（eps=1e-6），数值验证自定义
  backward 的雅可比（`test_gradcheck_*`）。
- **带 dropout 用固定 RNG 逐项对照**：同初始参数、同输入、同种子，
  `CheckpointedSequential` 与 `nn.Sequential` 在 `float64` 下**逐比特相等**
  （输出、输入梯度、每个参数梯度）；`float32` 下用紧容差
  `rtol=1e-5, atol=1e-6` 比对。
- 覆盖：单段/多段/不均匀、参数共享、输入有/无 grad、参数冻结、RNG 流、
  原地与高阶拒绝、storage 与引用计数、模块调用次数。

不下载任何模型或数据，全部使用合成张量。
