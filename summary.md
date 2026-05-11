# SmolVLM2-500M 极限边缘部署：W4A8 量化算法演进与软硬协同验证报告

## 1. 项目背景与物理约束
本项目旨在将视觉大语言模型 SmolVLM2 (500M) 部署至 KV260 FPGA 边缘计算板卡。
* **硬件物理约束**：W4A8（权重 4-bit，激活 8-bit）。
* **硬件接口特性**：128-bit AXI 接口，天然契合 `group_size=32`（16 Byte）的数据吞吐。
* **性能天花板 (FP32 无损基线)**：**60.5%**（ScienceQA 验证集）。

---

## 2. 量化策略全景对比与实现细节

### 2.1 原定硬件基线：RTN (Round-to-Nearest)
* **实现细节**：
  * 采用 `group_size=32` 的局部隔离机制。
  * 均匀量化（Uniform Quantization），提取块级最大绝对值作为线性缩放因子（Scale），直接划分为等距刻度。
* **致命缺陷（大象踩死蚂蚁）**：大模型权重存在极端的长尾分布（Outliers）。在一个 32 人的 Group 中，为了容纳 1 个极其夸张的极值（大象），等距划分的网格会被拉得极其粗糙，导致剩余 31 个代表微小注意力特征的权重（蚂蚁）全部被四舍五入为 0。
* **实验结果**：**27.0%**（证明均匀量化无法胜任大模型 W4 压缩）。

### 2.2 硬件护甲派：纯 BFP (Block Floating Point)
* **实现细节**：
  * 块级浮点机制，`block_size=32`。
  * 不提取直接的线性 Scale，而是提取组内最大值的**共享指数（Shared Exponent）**：`floor(log2(max_abs)) + 1.0`，并保留尾数（Mantissa）。
* **算法修复**：
  * **修复前 (3.5%)**：指数计算漏掉 `+1.0`，导致最大极值被拦腰斩断；且权重与激活位宽参数耦合，未能正确模拟 W4A8。
  * **修复后**：解耦参数，完善数学边界。
* **优势**：依靠浮点的动态范围，完美抗住了极值的破坏力，同时将爆炸半径隔离在 32 个元素内。
* **实验结果**：**42.5%**。

### 2.3 算法魔法派：纯 APoT (Additive Power-of-Two)
* **实现细节**：
  * 非均匀量化（Non-uniform）。刻度在 0 附近极其细密，在两端极其稀疏，完美拟合权重的钟形分布。
* **算法演进**：
  * **初代 (Per-tensor 全局量化)**：全层共享一个 Scale。大模型单层只要出现一个超级极值，整层网络降智。截断护甲（Clipping）又会切断极值导致输出乱码。**结果：23.0%**。
  * **满血版 (Group-wise APoT)**：引入硬件的 `group_size=32` 约束，将矩阵拆块，每组独立计算 Scale 并映射到 APoT 网格。实现了“大象（稀疏大网格）”与“蚂蚁（细密小网格）”的和平共处。
* **实验结果**：**36.5%**（较全局版暴涨 13.5 个百分点）。

### 2.4 终极 SOTA：Hybrid 异构软硬协同方案
* **实现细节**：
  * 深度结合 LLM 的结构异质性，打破同构量化范式：
    * **Attention 层（方向盘）**：极值较少，更依赖微小权重的密集分辨率。部署 **Group-wise APoT**，用最密集的网格捕获语义关联，同时大幅节省硬件 DSP 资源。
    * **MLP 层（发动机）**：Massive Outliers（超级极值）重灾区。保留 **BFP**，用块级浮点死死抗住极值爆炸。
* **超参微调优化**：
  * 经过消融实验验证，APoT 网格下沉深度 `max-power=6`（下潜至 0.015625）为全局最优解。放宽至 `max-power=4` 会导致 0 附近漏风，精度下降（44.5%）。
* **实验结果**：**46.5%**（反杀纯 BFP，创造 W4A8 极限约束下的最高精度）。

---

## 3. 核心算法代码重构参考 (Python / PyTorch)

### 3.1 满血版 Group-wise APoT 核心算子
```python
def apot_quantize_tensor(x: torch.Tensor, levels: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    if x.numel() == 0:
        return x
    flat = x.detach().reshape(-1)
    n = flat.numel()
    
    # 填充补齐
    rem = n % group_size
    if rem != 0:
        pad = group_size - rem
        flat = torch.cat([flat, torch.zeros(pad, device=flat.device, dtype=flat.dtype)], dim=0)

    # 拆解 Group 并提取局部 Scale
    groups = flat.reshape(-1, group_size)
    max_abs = groups.abs().amax(dim=1, keepdim=True)
    max_abs = torch.clamp(max_abs, min=1e-8)

    # 归一化与非均匀网格映射
    normalized = (groups / max_abs).clamp(-1, 1)
    q_abs = quantize_abs_to_levels(normalized.abs(), levels)
    
    # 伪量化解压恢复
    q = normalized.sign() * q_abs
    deq = q * max_abs
    out = deq.reshape(-1)[:n]
    return out.reshape_as(x)
```

### 3.2 满血版 BFP 核心算子 (修复边界溢出)
```python
def bfp_quantize_tensor(x: torch.Tensor, block_size: int = 32, mantissa_bits: int = 7) -> torch.Tensor:
    if x.numel() == 0:
        return x

    flat = x.detach().reshape(-1)
    n = flat.numel()
    rem = n % block_size
    if rem != 0:
        pad = block_size - rem
        flat = torch.cat([flat, torch.zeros(pad, device=flat.device, dtype=flat.dtype)], dim=0)

    blocks = flat.reshape(-1, block_size)
    max_abs = blocks.abs().amax(dim=1, keepdim=True)
    nonzero = max_abs > 0

    shared_exp = torch.zeros_like(max_abs)
    # ====== 核心数学修复：+1.0 防止最大值溢出被腰斩 ======
    shared_exp[nonzero] = torch.floor(torch.log2(max_abs[nonzero])) + 1.0
    # ====================================================

    qmax = float((1 << (mantissa_bits - 1)) - 1)
    scale = torch.pow(2.0, shared_exp - (mantissa_bits - 1))
    scale = torch.where(nonzero, scale, torch.ones_like(scale))

    q = torch.round(blocks / scale).clamp(-qmax - 1.0, qmax)
    deq = q * scale
    deq = torch.where(nonzero, deq, torch.zeros_like(deq))

    out = deq.reshape(-1)[:n]
    return out.reshape_as(x)
```

---

## 4. 实验数据追踪表 (W4A8, ScienceQA, SmolVLM2)

| 量化策略 (W4A8) | 算法演进状态 | 准确率 (Accuracy) | 结果分析 |
| :--- | :--- | :--- | :--- |
| **FP32 (基线)** | 无量化原盘 | **60.5%** | 理论天花板 |
| **RTN** | 原定硬件基准 (group=32) | 27.0% | 均匀量化导致细节特征丢失（大象踩死蚂蚁） |
| **APoT (旧)** | 全局量化 (Per-tensor) | 23.0% | 极值拖垮整层刻度，剪裁则切除核心脑白质 |
| **BFP (旧)** | 代码 Bug (漏+1且参数耦合) | 3.5% | 极值腰斩溢出，激活值被误压至 4-bit |
| **BFP (新)** | 修复指数与解耦 W4/A8 | **42.5%** | 浮点动态范围+局部隔离，完美抗击长尾极值 |
| **APoT (新)** | 进化为 Group-wise APoT | 36.5% | 局部隔离+非均匀网格，兼顾极值与细节 |
| **Hybrid (满血)** | Attention(APoT) + MLP(BFP) | **46.5%** | 结合分布异质性，实现精度与算力资源的极致平衡 |

---

## 5. 科学结论
通过本次软硬件联合验证，我们得出了明确的结论：在不进行重训练（PTQ）的前提下，传统的均匀量化（RTN）无法满足视觉大模型在 W4 极限环境下的特征表达。我们提出的 **基于算子特征感知的 Hybrid 异构量化架构**，不仅大幅降低了 FPGA 板卡的 DSP 计算开销，还在纯理论层面将准确率推高至 46.5%，为下一步的硬件 RTL 部署确立了极具工程指导意义的 Golden Reference（黄金参考基准）。