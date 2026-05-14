# PAPI: Exploiting Dynamic Parallelism in Large Language Model Decoding with a Processing-In-Memory-Enabled Computing System
PArallel Decoding with PIM

## abstract

## 1.Introduction
- (第四段) 作者 profiling 发现：在真实系统里，kernel 的性质会因为 decoding parallelism 的变化而动态改变。parallelism 会变的三个原因：
  - max decoding parallelism 受内存容量限制，而请求输出长度事先未知
  - 不同用户有不同 QoS/quality of service 要求，系统会调整 max decoding parallelism
  - 一些 batching/speculative decoding 方法本身就会在运行时动态调参
  $\Rightarrow$ 同一个 FC kernel 在某些时刻适合 GPU，在另一些时刻反而更适合 PIM
- (第6段) 关键技术：
  - dynamic parallelism-aware task scheduling，根据运行时并行度决定 kernel 去哪跑。
  - 异构体系结构，包含 GPU、PIM 和 host CPU。
  - hybrid PIM architecture，分别服务不同类型的 memory-bound kernel。
    - 一种偏性能优化 performance-optimized
    - 一种偏大容量优化 memory-capacity-optimized

## 2.Background 
(已融进自己的笔记)

## 3.Motivation
### 3.1.Analysis of LLM inference
用 roofline model 分析 batch size (batching) 与 speculation length () 如何影响 FC 和 attention kernel 的算术强度 arithmetic intensity.
硬件平台是 A100，峰值算力 312 TFLOPS (每秒真正做了多少 FLOPs)，峰值带宽 1935 GB/s。

- 第二段分析 Figure 2(a):
  - batching 能提高 FC 对权重的重用，从而提高 FC 的 arithmetic intensity
  - attention 在 batching 下几乎没有类似的数据复用，因此 arithmetic intensity 几乎不变。
- 第三段分析 Figure 2(b)
  - FC 的瓶颈会随着 speculation length 变化而切换
  - attention 更稳定地偏 memory-bound。
- $\Rightarrow$ 
  - FC 的“适合 GPU 还是适合 PIM”不是固定的，和 batch size、speculation length 都强相关；
  - attention 更稳定地适合 memory-centric 硬件。

### 3.2.Varying Parallelization Levels in LLM Inference
真实服务环境中，batch size 和 speculation length 都会在运行时显著变化，因此并行度是动态的。原因：
- Initial RLP = the RLP when batched execution begins. 为什么会变？
  - SLO limits:
    - 提高 RLP 可以提高吞吐，但会增大单请求时延
    - 高要求 user latency <u>Service Level Objective/SLOs</u> 会限制 intial RLP, 限制最大 batch size
  - 内存容量限制:
    - 输入输出越长，可容纳请求数越少 (Longer sequences need more memory capacity for KV cache), batch size 越小
  - dynamic batching 意味着 starts processing a batch once the batch is full or exceeds a time limit
    - 但当请求到达不频繁时，系统可能启动 with different batch size，因此初始 RLP 不固定。
- runtime RLP = batch 执行过程中并发请求数。为什么会变？
  - 如果采用 static batching，as 每个请求输出长度不同，随着某些请求先结束，RLP 会逐步下降。
  - 如果采用 mixed continuous batching，则新请求还能插入进来，因此 RLP 可能上上下下变化，以尽量提高硬件利用率。
- TLP. 为什么会变？
  - 有些工作会在每轮 decoding 时动态改变 speculation length
  - batching 和 speculative decoding 还需要联合优化，比如 batch 小时可以把 speculation length 拉高，以更好填满硬件。

### 3.3. Limitations of Existing Processing-In-Memory Architectures for LLM Inference
- (第1段) PIM 把处理逻辑放到内存里，给 memory-bound kernel 提供高带宽、降低数据搬移瓶颈
- (第2段) 以前的工作：已经结合 GPU 与 PIM，而不是单一 GPU，但仍有两个主要缺点:
  - 静态映射, i.e., FC and attention kernels each are always mapped to the same computing hardware
    > e.g. (A100+) AttAcc 永远把 attention 放 PIM、FC 放 GPU；
    > e.g. IANUS: 永远把 FC 放 PIM、attention 放 NPU；
    > e.g. SpecPIM: executes attention and FC kernels at the high-performance processor and PIM devices concurrently, 仍假设固定 batch size 和 speculation length。
    - 定量证明:
      - 低并行度 (e.g. batch size=1 且 speculation length=8，或 batch size=4 且 speculation length=2) 时，PIM 方案比 A100 更快 (更像 memory-bound)；
      - 高并行度 (e.g. batch size≥16) 时, A100 显著更快 (更像 compute-bound)，适合 GPU。既然 RLP 和 TLP 事先不可知，就必须动态决定 FC 在哪里跑。
  - 只提供一种 PIM: 即便 attention 和 FC 都可能在 GPU 上表现为 memory-bound，它们的 arithmetic intensity 仍有很大不同。
    > e.g. batch size=4、speculation length=8 时，FC 是 31.7 FLOPs/Byte，而 attention 只有 7.0 FLOPs/Byte。若计算吞吐相同，则 attention 需要大约 4.5× 更高的内存带宽。

### 3.4.Our Goal


## 4.PAPI: Overview
### 4.1.PAPI: Key Components
- Heterogeneous Architecture 包含三类部件：
  - host CPU: 负责下发指令给高性能处理器和 Attn-PIM
  - 高性能处理器: 包括 processing units/PUs（这里实验用了 GPU tensor cores）、FC-PIM memory units 和 hardware scheduler
  - Attn-PIM
- Hybrid PIM Units
  - FC-PIM 提供更高的计算能力，服务 FC kernel
    - FC kernel 可能是 compute-bound 或 memory-bound，因此由调度器动态决定
  - Attn-PIM 提供更大的内存容量，服务 attention kernel。
    - attention kernel 一直是 memory-bound，所以固定放到 Attn-PIM；
- Dynamic Parallelism-Aware Scheduling：
  - 判断 FC kernel 当前更像 memory-bound，则在 FC-PIM 上执行；若 compute-bound，则在高性能处理器的 PUs 上执行，FC-PIM 当作主存来存 weight，PUs 去读取并计算。
  - attention 不参与这种二选一，因为它一直在 Attn-PIM 上。

## 5.PAPI Dynamic Scheduling

### 5.1.Memory-Boundedness Identification of the FC Kernel
- (第一段) ... AI ≈ RLP × TLP
- (第二段) 验证这个近似的准确性: 作者在 GPT-3 66B 上用各种 RLP、TLP 配置比较真实 arithmetic intensity 和估计值，发现大多数情况下几乎吻合。
  - 只有在极大并行度时估计略高，但这不会影响调度判断，因为那时 FC 无论如何都会明显偏 compute-bound。

### 5.2.Runtime Scheduling Implementation
#### 5.2.1.Initial Scheduling
- 发生在服务启动前。
- 此时 RLP 设成 batch size，TLP 设成系统规定的 speculation length。
- 调度器计算 RLP×TLP，并与一个 memory-boundedness threshold α 比较。如果大于 α，就认为 FC 是 compute-bound，放到 PUs
  - 作者分别在 PIM 和 PU 上跑 FC kernel，看不同并行度下谁更快，再选一个最优阈值。

#### 5.2.2.Runtime Scheduling
- 把当前 batch 中所有请求的输出 token 收集成一个向量
- 统计这个向量里有多少个 <|eos|>。如果>0，说明有请求结束了，RLP 下降，相应 Attn-PIM 资源释放出来
- TLP 一般变动没那么频繁，所以系统用一个寄存器保存 TLP 值. 如果 host CPU 软件修改了 speculation length，就通知 PAPI sys 更新这个寄存器
- 计算新的 RLP×TLP，预测下一轮 FC 的 arithmetic intensity
- 把这个值与阈值 α 比较，决定是否需要把 FC 从 PU 切换到 FC-PIM，或者从 FC-PIM 切回 PU
 

## 6.PAPI Architecture
### 6.1.FC-PIM Design
- (第一段) 设计目标：FC-PIM 提供比传统近存计算更高的计算并行度，同时还不能违反功耗约束。作者 use 开源 HBM-PIM simulator which is 基于 Ramulator 2.0 (内存系统模拟器) to 评估不同 PIM 配置。
- (第二段) 分析传统 1P1B (one processing core per memory bank) PIM 的能耗构成:
  - DRAM Access: the energy consumption required to activate and precharge an HighBandwidthMemory DRAM row to read the weight data
    - 在没有 DRAM data reuse 时，占总能耗的 96.7%
  - Transfer: the energy consumption of transferring activation data from the buffer die, via the TSV, global controller, and bank group
controller, to the processing core.
  - Computation: computation energy in FloatingPointUnits of the processing core.
- (第三段) 如果同一行 DRAM 数据取一次后能被多次复用，那么 DRAM Access 的占比会大幅下降。
- (第四段) 实验：每 bank 配 1、2、4 个 FPU，并观察不同 data reuse level 下功耗。结果表明，data reuse 越高，功耗越低 
- (第五段) 分析面积约束: m(n×AFPU + Abank) ≤ AMax (m DRAM banks and each DRAM bank employs n FPUs)，并带入 CACTI-3DD 估计 (单 bank 面积 0.83 mm²，单 FPU 面积 0.1025 mm²，单 HBM die 面积上限 121 mm²)。对于 4P1B 配置，最多只能放不到 97 个 bank，因此最终设计为 96 banks，也就是 3 个 bank groups。 

### 6.2.Attn-PIM Design
- FC-PIM arithmetic intensity 更高，用 4 FPUs per bank，提高执行并行度；
  Attn-PIM 采用 1 FPU for every 2 banks = 1P2B。
  - 为什么 attention 不用 1P1B 呢？因为 attention 基本没有 data reuse，所以 1P1B 在功耗上会超出 HBM 限制，而 1P2B 能压到预算之内。
- FC 所需内存容量主要由模型大小 (W 是先训练好的) 决定，运行时基本不变. Attention 所需容量则随序列长度线性增长 (历史越长，要读的K V 越多)。因此，Attn-PIM 采用 disaggregated 设计，方便在系统中部署更多 device，以容纳更大的 KV cache。 
- PAPI 不是把一个 PIM 用到所有 kernel 上，而是分别优化“计算并行度”和“内存容量”，从而同时满足 FC 和 attention 的不同需求。

### 6.3.System Integration
不同子系统之间不必用同一种昂贵互连，应按通信需求分层设计。

- FC-PIM 和 PUs 之间要传大量权重数据，所以需要 NVLink 这样的高速互连；否则数据传输会反过来成为瓶颈。
- attention (跨设备) 主要拿当前较小的 query 去查本地 Attn-PIM 里存着的 KV cache，因此用 PCIe / CXL 就够了。
  - PCIe 每条总线最多支持 32 devices，CXL 可扩展到 4096 devices
  - attention 相关的大块数据（KV cache）尽量放在 Attn-PIM 那边本地

### 6.4.Data Partitioning Across PIM Devices
[补充] 一大块矩阵太大了，没法塞到一个 PIM 设备里，也没法让一个小计算单元独自高效处理，所以必须把矩阵切成很多块，分散到很多 HBM/PIM 设备和设备内部的小单元里并行处理。
- 层级结构：
  - 多个 HBM devices
    - 每个 HBM device 里有 pseudo-channels
      - 每个 pseudo-channel 里有 bank groups
        - 每个 bank group 里有 banks
          - bank 下面再有更小的乘法器/计算单元
- column-wise = 把矩阵竖着切
  row-wise = 把矩阵横着切
---

- 对于 attention kernel, 作者把不同 attention heads 分配到不同 Attn-PIM 设备.
  - K^T 在 pseudo-channel 和 bank-group 层是 column-wise partitioned; 
    在 bank 和 multiplier 层是 row-wise partitioned.
  - V 在 pseudo-channel 和 bank-group 层是 row-wise partitioned; 
    在 bank 和 multiplier 层是 column-wise partitioned.
- 对于 FC, 作者先把大 weight matrix 分成多个 2D blocks，each 映射到 HBM device 上。
  - 每个 2D block 在每个 HBM device 内的分法基本与 attention 中 KT 的分法相似。

### 6.5.Practicality and Architectural Scalability
- Complementary PIM Units for Diverse Workloads：
  - FC-PIM 和 Attn-PIM 共享相同的 bank-level computation fabric 和 memory hierarchy，差别主要在每个 bank 放多少 processing units, i.e., 是在统一 HBM-PIM substrate 上的两种不同配置。
    - Attn-PIM 少放 PUs，专门处理 memory-bound 工作；
    - FC-PIM 多放 PUs，以提升 FC 的计算吞吐。
  - Attn-PIM 这一类设计已经在 UPMEM、HBM-PIM 等原型或产品中出现过，因此工业实现有依据。
  - FC-PIM 增加每 bank 的计算资源，同时不突破 HBM 功耗限制。
  - 两类 PIM 都不需要修改 DRAM core array，而是在外围电路放计算逻辑，这样面积开销更低，也更容易兼容现有 HBM 工艺。作者认为这有利于未来大规模部署。

- Deployment of Emerging LLM Models
  - 作者认为 Mixture-of-Experts 模型只激活部分 expert，本身带有稀疏性，因此很适合硬件做资源利用优化。
  - FC-PIM 尤其适合这种稀疏计算，因为它可以把不同 expert 的权重切片放在同一个 DRAM bank 中，减少 FPU 闲置，并降低数据搬移、能耗和延迟。
    - 一个 expert 本质上就是一套大的权重矩阵，尤其常见在 FFN 里。这套权重太大，不可能整个塞进一个很小的 bank 局部计算单元里，所以通常会被切成很多小块
      - 以前：这个 bank 绑定单一 expert，没选中就全闲
      - 现在：这个 bank 覆盖多个 experts，命中任意一个就能工作
    - 所以 PAPI 不只是为 dense transformer 准备，也试图说明自己对未来模型有扩展性。

## 7.Evaluation
### 7.1.Evaluation Methodology
- Comparison Points and Simulation Methodology
  - 对比系统
    - A100+AttAcc：
      - 6 个 A100 GPU
        - Each GPU contains 5 HBM devices connected via NVLink，6 个 GPU ↔ 30 个 HBM devices，总约 80GB
      - AttAcc PIM (1 FPUnit per DRAM bank)，FC 全在 GPU 上，attention 在 PIM 上
        - 其他 HBM devices，包括 Attn-PIM，容量都是 16 GB      
      - 所有系统都配 90 个 HBM devices，其中 30 个给 FC kernels 的 weight parameters，60 个给 attention kernels
        - 外部公平条件统一：HBM3, 频率 = 333MHz, pin 带宽 = 5.2Gbps/pin, device 数量
    - A100+HBM-PIM：
      - 6 个 A100 GPU
      - 三星 HBM-PIM: 1 FPUnit per 2 DRAM banks
    - AttAcc-only：纯 PIM，FC 和 attention 都在 PIM 上执行。
    - PAPI：
      - 6 个 A100 GPU
        - 每 GPU 只算 60GB
        - 要容纳 GPT-3 175B 的模型参数需要大约 350 GB memory
      - FC-PIM: 4P1B; 更偏算力，容量缩小到 12GB
      - Attn-PIM: 1P2B; 更偏容量，保持 16GB
- Workloads: 评估 LLaMA-65B、GPT-3 66B、GPT-3 175B，数据类型 FP16，数据集用 creative-writing 和 general-qa tasks in the Dolly dataset，
  - 采用 static batching with varying initial RLP (batch size). $\Rightarrow$ 让系统在真实输入/输出长度变化下，经历动态并行度变化。

### 7.2.End-to-End Performance and Energy Efficiency
- Performance Speedup: 
  - PAPI 在 Dolly creative-writing dataset 对比 A100+AttAcc、A100+HBM-PIM、AttAcc-only 平均分别达到 1.8×、1.9×、11.1× 的加速。原因:
    - 调度动态，FC 能随时在 GPU 和 PIM 之间切换；
    - 硬件异构，PAPI 有 hybrid PIM，能更贴合 FC 和 attention 的不同需求。
  - 为什么 AttAcc-only 经常更差? 纯 PIM 方案的计算吞吐有限，而实验里的 FC 在很多配置下更偏计算密集，因此纯 PIM 顶不住
  - 为什么 A100+AttAcc 和 A100+HBM-PIM 差不多？二者主要只在 attention 的 PIM 实现上不同，而 attention 占整体运行时间相对较小，所以系统级差距不大。
  - PAPI 在 Dolly general-qa dataset 对比 A100+AttAcc、A100+HBM-PIM、AttAcc-only 的加速分别是 1.7×、1.7×、8.1×，略低于 creative-writing。因为：
    - creative-writing 输出更长，所以 decoding 占比更高，PAPI 的优势更容易放大；
    - 输出更长意味着运行过程中的 parallelism change 更显著，更利于动态调度发挥作用。
- Energy Efficiency
  - PAPI 相比 A100+AttAcc，在 creative-writing 和 general-qa 上平均提升能效 3.4× 和 3.1×。
    - 原因：A100+AttAcc 的 FC 全在高能耗 GPU 上执行，而 PAPI 会把部分 FC 移到 FC-PIM，降低数据搬移和功耗。
  - 相比 AttAcc-only，PAPI 的能效提升只有 1.15× 和 1.01×。
    - PAPI 毕竟有时会把 FC 放到 GPU 上，而 GPU 本身比纯 PIM 更耗能。所以 PAPI 的主要能效优势是相对“GPU主导系统”体现出来的，而不是相对“纯 PIM”特别夸张。

### 7.3.Sensitivity to Parallelization Levels
- RLP
  - 低 RLP 时，AttAcc-only 甚至可能比 A100+AttAcc 更好，因为这时 FC 更像 memory-bound；
  - 但随着 RLP 增大，AttAcc-only 会快速恶化，因为 FC 越来越算力密集，纯 PIM 跑不过 GPU。
  - PAPI 在所有 RLP 配置下都最好，因为它能随 RLP 变化调整 FC 的执行位置。
- TLP (固定 batch size=4，只改变 speculation length)
  - PAPI 相比 A100+AttAcc 和 AttAcc-only 平均分别有 1.5× 和 3.0× speedup。
  - PAPI 相比 A100+AttAcc 的优势会随着 TLP 增大而下降。
    - 原因: TLP 越大，FC 越像 compute-bound，PAPI 就会把更多 FC 放回 GPU；
    - 若 TLP 足够大，PAPI 的行为会越来越接近 A100+AttAcc

### 7.4.Performance Analysis of PAPI
- 比较两个 PIM-only 架构：AttAcc-only, 去掉 A100 后只保留 FC-PIM+Attn-PIM 的 PIM-only PAPI $\Rightarrow$ 隔离出 hybrid PIM 本身的价值
  - PIM-only PAPI 相比 AttAcc-only 平均有 2.3× speedup。并行度越高，FC 越需要更多计算能力，所以 PAPI with more PUs 的 speedup 越大
  - FC kernels are responsible for most of the execution time, so im- proving their performance has the largest impact on overall speedup.
- execution time breakdown for 两个 PIM-only 架构：
  - FC kernel 占总执行时间主导地位，所以给 FC 更高执行并行度最有价值。
  - PIM-only PAPI 在 FC 上实现了 2.9× speedup
  - attention 在 Attn-PIM 上反而比 AttAcc-only 慢 1.7×，因为 Attn-PIM 用了 1P2B，计算并行度更低，这是为了换更低 area/power overhead。
  - 通信占 decoding 总时间的 28.2%，说明未来如果互连再加强，PAPI 还可能更快。

## 8.Related Work
- 现有 PIM-LLM 方案
- 其他 LLM accelerator，比如多 FPGA 或 CPU-GPU 协同方案。这些方案虽然能加速推理，但没有显式处理“运行时并行度变化导致 kernel 瓶颈切换”的问题。
- pruning、quantization 等近似方法。作者强调，PAPI 不依赖近似，不牺牲模型质量，而是纯靠架构和调度优化来获得收益。

## 9.Conclusion


# PIM is All You Need
A CXL-Enabled GPU-Free System for Large Language Model Inference

## abstract
- LLM inference 是 autoregressive to 一个 token 一个 token 生成；operational intensity 低，更偏 memory-bound。
- 同时 LLM 参数量大: 每个 prompt 都有独立 KV cache, 长 context window 会让 KV cache 容量需求继续放大 => 限制 batch size。
- 现有 GPU/TPU 主要为 compute throughput 优化，但 LLM decoding 主要需要高 memory bandwidth，所以大量昂贵算力没有被充分利用。
- 作者提出 CENT: CXL-ENabled GPU-Free sysTem for LLM inference。
  - 用 CXL memory expansion 提供大容量；
  - 用 near-bank PIM 提供高内部带宽；
  - 用 PNM units 支持 PIM 不适合做的复杂操作；
  - 用 CXL switch 支持多 CXL devices 之间的 peer-to-peer + collective communication。
- 结果：相比 GPU baseline，在相似平均功耗和最大可支持 batch size 下，CENT achieves 2.3× higher throughput, 2.9× less energy, and 5.2× more tokens per dollar。

## 1.Introduction
- (第一段) LLM 服务成本很高，原因: 模型大, 推理 serving 需要大量高端 GPU server。因此需要 effective + 更便宜的 inference server。
- (第二段) Decoder-only LLM 带来的 larger parameter sizes: 
  - model parameters 大; decoding 缺少参数复用;
  - KV cache 随 context window length 和 batch size(用户数)增长
  $\Rightarrow$ multi-GPU systems 的计算资源未被充分利用，users pay for expensive computing resources for memory-bound LLM inference tasks.
- (第三段) PIM/PNM 的 tradeoff：
  - PIM 把 PU 放在 DRAM bank 附近，内部带宽高，但 DRAM process 下的 near-bank PU 会带来面积开销，降低 memory density；
    - DRAM process 指专门用来制造 DRAM 存储单元的半导体工艺, 目标: 在尽可能小的面积里存尽可能多的 bit, 会极度优化 density, 
      - 不太适合做复杂逻辑电路, 所以同样一个 compute unit 可能面积更大、效率更差，且会占掉本来可以用来放 memory cell 的面积
      - 一个 DRAM cell 通常可以粗略理解成：1 transistor + 1 capacitor/ 1T1C, 要在很小面积里放一个电容来存电荷
  - PNM 把计算放在 memory controller 附近，CMOS process 下更省面积、更灵活，但带宽不如 PIM。
    - CMOS process 指制造普通逻辑芯片的工艺, 晶体管速度快, 适合做逻辑计算/复杂电路

  $\Rightarrow$ 只用 PIM 或只用 PNM 都不够，需要 hierarchical PIM-PNM。
- CENT 的 key ideas：
  - 每个 CXL device 里有 16 个 GDDR6-PIM chips，每个 chip 有 2 个 GDDR6-PIM channels，并配有 PNM units；
    - intra-device communication between PIM chips and PNM units is supported through a shared buffer
  - 多个 CXL devices 通过 CXL switch 连接，由 host CPU 驱动；
    - inter-device communication is enabled by CXL transactions
  - PIM 做 GEMV/MAC 这类占绝大多数计算量的操作，PNM 做 Softmax、sqrt、division 等复杂操作。
- (第六段) 作者还强调 CENT 不是只做单 device，而是支持 LLM 并行映射：
  - Pipeline Parallel/PP: 每个 transformer block / pipeline stage 放到 device 内多个 memory channels，提升 throughput；
  - TP: 一个 transformer block 分到多个 CXL devices，降低 latency；
  - Hybrid TP-PP: 在 throughput 和 latency 之间折中。

## 2.Motivation
### High Memory Capacity Requirement
- LLM 参数从 billion 到 trillion 级别增长；context window 也从几千扩到 128K/1M。
- KV cache 是 per-user 的，不能像 weight 那样在 batch 内共享；context 越长，单个请求占用 memory 越大，系统能容纳的 batch size 越小。
- Figure 1 的核心意思：Llama2-70B 在 4×A100 80GB 上，batch size 变大时 throughput 会先提高，但一旦 memory requirement 超过 GPU capacity 就饱和；context 从 4K 增到 32K 后，可支持 batch size 从 128 降到 8。

### Low Operational Intensity
- LLM inference 有两个阶段：
  - prefill: prompt tokens 可以并行处理，主要是 GEMM，operational intensity 高；
  - decoding: output tokens 依次生成，主要是 GEMV，operational intensity 低。
- batching 可以把多个 GEMV 合成 GEMM，提高 operational intensity；但 attention 依赖每个 prompt 自己的 KV cache，所以提升不是线性的。
- Grouped-query attention 也能把部分 GEMV 变成 narrow GEMM，但仍低于 GPU 充分利用所需的 operational intensity。
- Figure 2(b): Llama2-70B 只有约 21% GPU compute utilization，而 BERT/ResNet 这类 GEMM-heavy 模型能更好利用 GPU。
- $\Rightarrow$ GPU 的问题不是跑不了 LLM，而是用昂贵 compute hardware 跑 memory-bound decoding 不划算。

### PIM Provides Higher Memory Bandwidth
- PIM 可以利用 DRAM internal bandwidth；这个 bandwidth 远高于 GPU 通过外部 HBM interface 能看到的 bandwidth。
- e.g. GDDR6-based AiM 有 16 TB/s internal bandwidth，而 A100 外部 HBM bandwidth 约 2 TB/s。
- 因为 LLM decoding low operational intensity，所以 PIM 这种 high bandwidth / lower compute 的设计更贴合 decoding。

### Low Memory Density of PIM
- PIM 的缺点是 near-bank PU 占用 DRAM die area，导致 memory density 下降。
- e.g. UPMEM DDR4 R-DIMM 容量降到普通 DIMM 的 25%~50%，AiM GDDR6 channel 降到 75%。
- 对 LLM 来说，这很麻烦：LLM 本来就需要大量 memory capacity，如果 PIM 容量太低，就必须靠更多 device 和 scalable interconnect 才能部署。

### Scalable Network of PIM
- 为了把 PIM 扩到 LLM 所需容量，系统需要：
  - scalable interconnect；
  - efficient collective communication primitives；
  - 适合 LLM 的 parallelization mapping。
- 作者选择 CXL 3.0：
  - 基于 PCIe physical layer，成本/生态更现实；
  - 支持通过 CXL switch 做 device-to-device communication；
  - CXL.mem latency 比 network-based RDMA 低很多；
  - 可扩展到 4096 nodes，比 NVLink 更 scalable。
- 虽然 NVLink 带宽更高，但作者认为 LLM inference 的跨 device 数据量较小，CXL bandwidth 不是主要瓶颈。

### Hierarchical PIM-PNM Architecture
- Transformer block 不只有 GEMV，还包括 RMSNorm、RoPE、SiLU、Softmax 等操作。
- 如果所有操作都用 general-purpose near-bank PU 做：
  - near-bank PU 面积开销大，memory density 更低；
  - general-purpose PU 对少量复杂操作来说 over-provisioned。
- 作者选择第二种路线：
  - PIM near-bank PU 做 MAC/GEMV，因为 MAC 占 transformer block 超过 99% arithmetic operations；
  - PNM units 做 Softmax、sqrt、division、复杂/模型相关操作。

## 3.Background
- Decoder-only LLM 先 prefill，再 decoding。
  - prefill: 并行处理 prompt tokens，填 KV cache；
  - decoding: 每轮基于 previous token 和 KV cache 生成下一个 token。
- 每个 decoder block 由 self-attention 和 FFN 组成，并配 residual connection 和 normalization。
- 以 Llama2 为例：
  - Q/K/V 由 input vector 乘对应 weight matrix 得到；
  - Q/K 经过 RoPE 编码位置信息；
  - K/V append 到 KV cache；
  - Q 与 K cache 得到 score，Softmax 后再乘 V cache；
  - attention output 经过 output projection、residual、RMSNorm；
  - FFN 里有两路 FC + SiLU + 最后一层 FC。

## 4.CENT Architecture
### 4.1.CXL-based Network Architecture
- CENT 系统结构：
  - host CPU；
  - CXL switch；
  - 32 个 CXL devices；
  - 每个 CXL device 有 CXL controller、PNM units、16 个 memory chips，每个 memory chip 有 2 个 GDDR6-PIM channels。
- CXL switch 到 host 用 x16 lanes，每个 CXL device 通过 x4 lanes 接 switch。
- Inter-device communication 通过 Shared Buffer 和 CXL port 完成，作者定义了：
  - SEND_CXL / RECV_CXL: device-to-device send/receive；
  - BCAST_CXL: broadcast，一个 device 写多个 device；
  - gather: 接收方执行多个 RECV_CXL，发送方各执行 SEND_CXL。
- 标准 CXL.mem 不直接支持 broadcast，作者用 reserved H-slot code 扩展，让 switch 把一个 flit 转发到多个目标 device。

### 4.2.Hierarchical PIM-PNM Architecture
- GDDR6-PIM channel:
  - 每个 CXL device 有 16 个 PIM controllers，每个 controller 管 2 个 PIM channels；
  - 每个 PIM channel 有 2KB Global Buffer、4 个 bank groups、16 个 banks；
  - 每个 bank 有 32MB capacity 和一个 near-bank PU。
- Near-bank PU:
  - 内部是 16 MAC reduction tree，处理 BF16；
  - 一个输入来自 local bank，另一个来自 Global Buffer 或 neighboring bank；
  - Global Buffer 可以把 256-bit data broadcast 给所有 PUs；
  - PU 频率 1GHz，单 PU 约 32 GFLOPS。
- PNM units:
  - 32 accumulators；
  - 32 reduction trees；
  - 32 exponent accelerators；
  - 8 个 BOOM-2wide RISC-V cores。
- PIM 和 PNM 之间通过 64KB Shared Buffer 通信。PIM 把 Shared Buffer 看成 256-bit registers，RISC-V core 把它看成 byte-addressable memory。

### 4.3.ISA Summary
- CENT ISA 分 arithmetic instructions 和 data movement instructions。
- Arithmetic:
  - MAC_ABK: all-bank MAC，是 GEMV/MAC 的核心；
  - EW_MUL: element-wise multiplication；
  - AF: activation function；
  - EXP/RED/ACC/RISCV: 给 PNM units 执行 exponent、reduction、accumulation 和 general RISC-V operation。
- Data movement:
  - CXLDevice ↔ CXLDevice: SEND_CXL、RECV_CXL、BCAST_CXL；
  - SharedBuffer ↔ DRAM Banks: WR_SBK、RD_SBK、WR_ABK；
  - GlobalBuffer ↔ DRAM Banks: COPY_BKGB、COPY_GBBK；
  - SharedBuffer ↔ PUs: WR_BIAS、RD_MAC；
  - SharedBuffer → GlobalBuffer: WR_GB。

## 5.Model Mapping
### 5.1.Pipeline-Parallel Mapping (PP)
- 目标：throughput critical serving。
- PP 把 transformer decoder blocks 分配到 pipeline stages，不同 prompts 同时在 pipeline 的不同 stage 上执行。
- CENT 会把多个 pipeline stages 放到一个 CXL device 里，每个 stage 用若干 PIM channels。
- stage 之间只传 embedding vector。Llama2-70B 里 embedding vector 约 16KB，所以跨 CXL device 的 PP 通信开销相对 PIM/PNM latency 很小。
- 作者不在单个 pipeline stage 内做 batching：
  - batching 需要更大的 Global Buffer / Shared Buffer 存多个 embedding；
  - PP 已经能利用 PIM compute resources；
  - 再 batching 反而增加 latency。

### 5.2.Tensor-Parallel Mapping (TP)
- 目标：latency critical serving。
- TP 用多个 CXL devices 的资源一起处理一个 decoder block，从而降低单 token latency。
- 具体映射：
  - FC layers 分布到多个 CXL devices，每个 device 做 weight matrix 的部分 rows；
  - FC 前，master CXL device broadcast 16KB embedding vector；
  - FC 后，各 device 把 partial result gather 回 master。
- attention layer 没有分布到所有 devices，而是放在 master CXL device 上。
  - 原因：如果把 attention 分布出去，需要频繁 AllReduce，CXL communication overhead 会很大。
- 对 Llama2-70B，TP 每个 transformer block 的 CXL transfer 只有约 135KB。

### 5.3.Hybrid Tensor-Pipeline Parallel Mapping
- TP 偏 latency，PP 偏 throughput；真实 serving 需要同时考虑 QoS。
- Hybrid TP-PP 把每个 transformer decoder 分配到多个连续 CXL devices。
  - e.g. 32 devices 中，每个 decoder 用 8 devices，则 TP=8、PP=4。
- embedding vectors 由每个 pipeline stage 的 master CXL device 做 multicast 和 gather。
- 这样可以一边用 TP 降低单 token latency，一边用 PP 并行处理多个 prompts 提高 throughput。

### 5.4.Transformer Block Mapping
- CENT 对 transformer block 做 fine-grained mapping，使整个 block 在 CXL device 内完成，避免 host 参与。
- PIM channels 负责：
  - FC layers 的 GEMV；
  - RMSNorm 的 vector dot product；
  - RMSNorm、SiLU、Softmax、RoPE 里的 element-wise multiplication。
- PNM 负责：
  - square root；
  - division；
  - Softmax 中复杂部分；
  - residual connection 的 vector addition；
  - RoPE 中 complex/real transform 等模型相关操作。
- 对 GDDR6-PIM channel 内的三个核心计算：
  - GEMV: matrix 按 rows 分到 16 banks，vector 进 Global Buffer，MAC_ABK 广播 vector segments 并从 banks 读 matrix rows 做 MAC；
  - Vector dot product: 两个 input vectors 放 neighboring banks，用 MAC_ABK 做 dot；
  - Element-wise multiplication: 输入放同一 bank group 的两个 banks，EW_MUL 读出后相乘并写回。

### 5.5.End-to-End Model Mapping
- CENT 支持完整 LLM inference。
- prefill 阶段：逐 token 处理 prompt，填 KV cache，映射方式和 decoding 类似。
- decoding 阶段：input embedding 和 transformer blocks 在 CXL devices 上执行；经过所有 blocks 后，top-k sampling 放到 host CPU 上做。

### 5.6.Programming Model
- 用户指定 CENT hardware configuration，比如使用多少 PIM channels、多少 pipeline stages。
- CENT library 提供 Python APIs，用于：
  - 分配 memory space；
  - 按 mapping strategy load model parameters；
  - 调用 GEMV、LayerNorm、RMSNorm、RoPE、SoftMax、GeLU、SiLU 等常用 LLM operations。
- CENT 用 in-house compiler 生成 arithmetic/data movement instructions。
- GEMV 编译逻辑大致是：
  - vector operand 放 Shared Buffer；
  - matrix operand 放 PIM channels；
  - WR_GB 把 vector 写到 Global Buffer；
  - WR_BIAS 初始化 accumulator；
  - MAC_ABK 做 multiply-accumulate；
  - RD_MAC 读回结果。

## 6.Methodology
- Baseline:
  - GPU system: 4×NVIDIA A100 80GB，NVLink 3.0；
  - CENT: 32 CXL devices，平均功耗与 GPU system 相近。
- System configuration:
  - CENT memory: 512GB GDDR6；
  - GPU memory: 320GB HBM2E；
  - CENT compute: 512 TFLOPS(PIM) + 96 TFLOPS(PNM)；
  - GPU compute: 1248 TFLOPS；
  - CENT internal bandwidth: 512 TB/s；
  - GPU external bandwidth: 8 TB/s。
- Workloads:
  - Llama2 7B、13B、70B；
  - 每个 query: prefill 512 tokens + decoding 3584 tokens = 4K context；
  - GPU 用 vLLM，batch size=128（throughput 饱和点）；
  - 不同模型用 1/2/4 GPUs 和 8/20/32 CXL devices 对齐参数规模。
- Simulation:
  - functional simulator 验证 instruction traces；
  - modify Ramulator2 建模 CXL device；
  - CXL 3.0 communication 用 analytical model；
  - power 用 activity-based model；
  - TCO 同时算 owned 和 rental 两类。

## 7.Results
### 7.1.CENT versus GPU Baseline
- Latency Critical (batch=1, TP mapping):
  - CENT end-to-end latency 相比 GPU 降低 4.6×。
  - 原因：decoding dominated by memory bandwidth，而 CENT 有更高 internal memory bandwidth。
- Throughput Critical:
  - GPU 用 batch=128；
  - CENT 用 PP，Llama2 7B/13B/70B 分别支持 batch=32/40/80；
  - CENT 平均 2.3× higher end-to-end throughput。
- 分阶段看：
  - prefill 阶段 GPU 更强，throughput 比 CENT 高 2.5×，因为 prefill 是 compute-intensive GEMM；
  - decoding 阶段 CENT 更强，throughput 比 GPU 高 2.5×，因为 decoding 是 memory-intensive GEMV；
  - prefill 只占 GPU end-to-end processing time 约 2%，所以整体结果主要由 decoding 决定。
- 长 context 下 CENT 优势更明显：
  - GPU 在 4K context 可 batch=128，但 32K context 只能 batch=16；
  - Llama2-70B 上，CENT 在 32K context 的 decoding throughput speedup 可到 3.3×。
- QoS:
  - 在类似 throughput 下，CENT query latency 比 GPU 低 3.4×~7.6×。

### 7.2.Power and Energy Consumption Analysis
- CENT 的 power 主要来自：
  - PIM operations: 54.5%；
  - DRAM activate/precharge: 30.2%。
- 单个 A100 的功耗约等于一个 CENT device 的 8×。
- GPU 在 prefill 和 decoding 都接近 TDP：
  - prefill 阶段 SM utilization 高，功耗高；
  - decoding 阶段 SM utilization 低，但 memory bandwidth 使用和更高 clock 仍让 board power 接近 TDP。
- Figure 15(c):
  - end-to-end: CENT 平均 2.9× more tokens/J；
  - prefill: GPU 2.4× more energy efficient，因为 compute-bound 且 SRAM reuse 好；
  - decoding: CENT 3.2× more energy efficient，因为 GPU 在 low operational intensity 下难以复用数据。

### 7.3.CENT versus PIM/PNM Baselines
- CENT vs CXL-PNM:
  - CXL-PNM 把 matrix/vector units 放在 CXL controller 旁边，靠近 LPDDR5X，但没有把计算放进 DRAM banks；
  - CENT 把 compute logic 放到 DRAM bank 附近，因此 memory bandwidth 和 compute throughput 更高；
  - 代价是 memory capacity 较低。
  - 在 OPT-66B 上，CENT 比 CXL-PNM throughput 高 4.5×。
- CENT vs GPU-PIM (AttAcc / NeuPIM):
  - AttAcc/NeuPIM 是 GPU + PIM heterogeneous system；
  - GPU 负责 prefill，PIM subsystem 负责剩余计算；
  - CENT 完全去掉 GPU。
- 作者认为 CENT 不用 GPU 做 prefill 的原因：
  - end-to-end performance 主要受 decoding 限制，prefill 平均只占 GPU inference time 约 2%；
  - CENT compute throughput 也不算极低，约为 GPU 的 49%；
  - 只为了 prefill 保留昂贵 GPU，TCO 不划算。
- 结果：
  - CENT tokens/$ 比 AttAcc 高 1.8×~3.7×；
  - 比 NeuPIM 高 1.8×~5.3×；
  - raw throughput 则取决于 input/output length 和 batch size，短序列下 GPU-PIM 可能借助 batching 提高 FC operational intensity，但长序列限制 batch size 后 CENT 更有优势。

### 7.4.Design Space Exploration
- CENT 可以通过增加 CXL devices 扩展。
- Llama2-70B 上，从 16 到 128 devices，throughput 从 0.68K tokens/s 增到 5.7K tokens/s。
- 扩展不是完全线性，会出现 plateau：
  - e.g. Llama2-70B 有 80 个 transformer blocks，40 devices 时可每 device 2 blocks；
  - 44 devices 时平均 1.8 blocks/device，若强行拆 block 会增加跨 device communication；
  - 因此作者保持与 40 devices 相同的 block 分布，让多出的 4 devices idle。
- 单 switch 下规模受两个因素限制：
  - CXL switch 的 lanes/ports 数；
  - server power supply。
- 更多 devices 可以靠 multi-socket CPUs 或两级 CXL switches / memory pool 扩展。

### 7.5.Generality
- CENT 面向 LLM 的通用结构，而不是只硬编码 Llama2。
- 不同 LLM 的差异主要在 activation functions 和 positional encodings。
- CENT 支持 GeLU、Swish 及 GLU variants，方法是把它们拆成 sigmoid/tanh 等基础非线性操作，再用 lookup table、PIM operations 和 RISC-V operations 支持。
- positional embedding 也支持 absolute 和 relative 两类。
- RISC-V cores 的意义：给未来 LLM 的新操作留 flexible support，而不是每次都重新设计 near-bank hardware。

## 8.Related Work
- ML accelerators / HW-SW co-design。
- CXL memory expansion。
- PIM / PNM architectures。
- Transformer accelerators。
- 作者强调 CENT 的区别：
  - 不是 GPU-centric accelerator；
  - 不是单纯 CXL memory expansion；
  - 不是只在 memory controller 附近做 PNM；
  - 而是 CXL-scale-out + PIM high internal bandwidth + PNM flexibility 的 GPU-free inference system。

## 9.Conclusion
- CENT 的核心判断：decoder-only LLM inference 同时 low operational intensity + high memory capacity requirement，因此继续堆 GPU 并不 cost-effective。
- CENT 用 CXL memory expansion 解决容量，用 PIM 解决 decoding bandwidth，用 PNM/RISC-V 解决复杂操作灵活性。
- 相比 GPU baseline，在最大支持 batch size 下，CENT achieves 2.3× higher throughput, 2.3×/2.9× less energy（正文不同位置表述略有差异）, and 5.2× more tokens per dollar。
- 更大的模型、更长的 context、更受 memory capacity 限制的 batch size，会进一步放大 CENT 这类 GPU-free PIM+CXL 架构的优势。

# PIMphony: Overcoming Bandwidth and Capacity Inefficiency in PIM-based Long-Context LLM Inference System

## abstract
- Long-context LLM 把 PIM 的问题放大了：context 变长以后，attention 读 KV cache 的 bandwidth pressure 和 capacity pressure 都更严重。
- 作者发现现有 PIM-based LLM inference 在 long-context 下有三个关键 inefficiency：
  - severe channel underutilization；
  - I/O bottleneck，WR-INP → MAC → RD-OUT 这种固定 pipeline 会让 MAC units 等数据；
  - static KV cache management 会按最大 context length 预留，造成大量 memory waste。
- 作者提出 PIMphony，一个 PIM orchestrator，不是单独换一个 PIM device，而是通过 compiler/runtime/hardware-aware scheduling 去提高多 PIM 系统利用率。
- 三个核心技术：
  - Token-Centric PIM Partitioning (TCP): 沿 token dimension 切分，让所有 PIM channels 都参与；
  - Dynamic PIM Command Scheduling (DCS): 根据 data dependency 动态发 PIM command，overlap I/O 和 MAC；
  - Dynamic PIM Access (DPA): 在 PIM module 内做类似 lightweight MMU 的动态地址转换和 lazy KV allocation。
- 结果：PIM-only systems 最高 11.3× throughput improvement，xPU+PIM systems 最高 8.4×。

## I.Introduction
- (第一段) Long-context LLM 的应用越来越多，比如 long document summarization、repo-level code analysis、CoT reasoning。这些任务需要几十万甚至 1M tokens 的 context window。
- (第二段) 长 context 带来的核心瓶颈是 memory bandwidth 和 capacity：
  - decoding 里 GEMV/attention 的 compute intensity 低；
  - KV cache 随 context length 增长，且会占据大量 memory。
- (第三段) 现有 PIM-LLM 工作（NeuPIMs、CENT 等）证明了 PIM 对 memory-bound kernel 有价值，但主要关注较短 context（比如 4K/32K），没有系统处理 128K/1M long-context 下 PIM 自身利用率下降的问题。
- (第四段) 作者 identify 三类 long-context PIM inefficiency：
  - workload imbalance 让 channels idle；
  - fixed write-compute-read pipeline 导致 I/O bottleneck；
  - static KV cache allocation 按最大长度预留，真实请求长度差异大时容量浪费。
- (第五段) PIMphony 的思路是 orchestration：
  - TCP 解决 channel utilization；
  - DCS 解决 command scheduling / I/O overlap；
  - DPA 解决 dynamic memory management。
- (第六段) 实现上，作者扩展 MLIR-based compiler 和 runtime，生成 token-level partitioning 和 dynamic KV allocation 所需的 PIM commands，并改 Ramulator-based simulator 去评估。

## II.Background/Motivation
### A.Long-Context LLM Inference/Decoding
- Long-context LLM 仍然是 Transformer decoder 架构，每层包含 MHA 和 FFN。
- attention 中 Q/K/V 生成后，K/V append 到 KV cache；后续 QKT 和 SV 都要访问这个 cache。
- GQA 会让多个 query heads 共享一组 K/V，从而减少 KV cache，但 attention 仍然随 context length 增长。
- 作者的 observation：
  - context 越长，attention compute intensity 越低；
  - KV cache memory footprint 随 context length 和 batch size 增长，导致 GPU/PIM 都受 capacity 限制。

### B.PIM Architecture and Instruction Execution
- DRAM-based PIM module 通常包括：
  - 每个 DRAM bank 附近的 MAC units；
  - shared Global Buffer (GBuf)，存输入；
  - Output Registers (OutRegs)，存结果；
  - PIM Controller / HUB，负责 decode 和 dispatch commands；
  - EPU/activation unit 处理 Softmax、activation 等辅助操作。
- PIM instruction 常见三类：
  - WR-INP: 从 GPR 复制 input 到 GBuf；
  - MAC: 对 DRAM row 做 dot-product；
  - RD-OUT: 从 OutReg 读输出到 GPR。
- 传统执行是静态 command stream，按 compiler 生成的固定顺序执行，缺少 runtime dependency-aware scheduling。

### C.Multi-Node DRAM-PIM System
- 为了容纳大模型和长 KV cache，需要多 PIM modules。
- 常见并行：
  - TP: 把 tensor shard 分到不同 PIM modules，最后同步 partial results；
  - PP: 把 transformer layers 分到不同 modules，micro-batches 在 pipeline stages 中流动。
- 问题是：跨 module 有并行，但 module 内部 channels 仍可能因为 partitioning 不当而空转。

### D.Challenges
- Channel Underutilization:
  - 现有方法多按 head/batch dimension 切 KV cache，即 Head-First Partitioning (HFP)；
  - long-context 下 batch size 变小，一个 request 的 KV cache 可能占满一个 channel；
  - head/batch tiles 不够多，很多 channels 没活干。
- I/O Bottleneck:
  - WR-INP → MAC → RD-OUT 固定 pipeline 中，2KB input buffer 和 4-byte per-bank output buffer 很容易成为瓶颈；
  - attention 的 feature dimension 小，I/O reuse 低，buffer turnover 更频繁；
  - static scheduling 无法有效 overlap data movement 和 computation。
- Static Memory Management:
  - 现有 PIM 为 variable context length 往往按 $T_{max}$ 预留 KV cache；
  - 真实 workload 长度差异很大，平均 capacity utilization 只有约 36.2%；
  - 根因是 PIM commands 里嵌入 fixed physical addresses，不能 runtime repurpose unused regions。

## III.PIMphony Overview
- PIMphony 是一个 PIM orchestrator，目标是让 long-context 下的 PIM MAC units、channels、memory capacity 都被更充分利用。
- TCP:
  - 不再主要依赖 batch/head dimension；
  - 把 token dimension 切到所有 channels；
  - 因为 long-context 的 token dimension 足够大，所以能稳定填满 channel parallelism。
- DCS:
  - 在 PIM controller 中加入 dependency-aware command scheduler；
  - 根据 real-time data dependency out-of-order issue commands；
  - 目标是把 WR-INP/RD-OUT 的 I/O latency 藏到 MAC 执行后面。
- DPA:
  - 在 PIM module 内加入 on-module dispatcher；
  - 支持 runtime virtual-to-physical translation；
  - 用 lazy allocation 替代按最大 context length 静态预留。
- 三者关系：
  - TCP 提高 channel utilization 后，系统可能更容易 I/O-bound；
  - DCS 进一步解除 I/O bottleneck；
  - DPA 提高 effective capacity，从而支持更大 batch / 更好 PP。

## IV.Token-Centric PIM Partitioning
### A.Channel Underutilization Issue
- Head-First Partitioning (HFP) 假设 head×batch 有足够 parallelism。
- 但 long-context 下 batch size 因 KV cache capacity 被压低；一个 request 可能占满 channel，导致 head/batch parallelism 不够。
- 在 TP 下，不同 requests token length 不同，短请求对应的 channels 先结束，长请求 channels 继续跑，整体被最慢 channel 卡住。
- 在 PP 下，每个 pipeline stage 只激活与当前 request/head 对应的一部分 channels，stage-level idle 和 pipeline bubbles 叠加。

### C.Intra-Module Token-Centric Partitioning
- TCP 把一个 attention head 的 token dimension 分布到所有 channels。
- QKT:
  - 每个 channel 处理一段 tokens；
  - 多个 channels 并行算 score 的不同 token segment。
- SV:
  - 每个 channel 对自己负责的 token segment 做 partial reduction；
  - partial results 经 PIM HUB/GPR 聚合成最终输出。
- TCP 的关键优势：
  - token length 在 long-context 下很大，比 batch/head 更稳定；
  - TP 中可缓解不同 request length 导致的 imbalance；
  - PP 中可让所有 channels 在每个 active stage 都参与。

## V.Dynamic PIM Command Scheduling
- 传统静态 schedule 的问题：
  - command 顺序固定，WR-INP、MAC、RD-OUT 很难充分 overlap；
  - 为避免 data hazard，静态调度通常保守，导致 MAC units 等 I/O。
- DCS 的做法：
  - PIM controller 维护 command queue、dependency table/status table；
  - 以 buffer entry / output register 为粒度检查 dependency；
  - 只要没有 RAW/WAW 等 hazard，就允许 out-of-order issue。
- 这样可以让后续 MAC 更早接上前面的 MAC，减少 I/O hand-off stalls。
- 作者给的例子里，DCS 把一个 execution 从 34 cycles 降到 22 cycles。
- 对 GQA 特别有用：
  - GQA 有 KV row reuse 的机会；
  - 但 row-reuse mapping 会增加 query/score 的 input transfer；
  - DCS 能 overlap 这些额外 I/O，让 row reuse 真正转化为性能收益。

## VI.Dynamic PIM Memory Management
### A.Conventional KV Cache Management in PIM
- 传统 PIM 的 instruction sequence 是预编译的，loop counts 和 operand addresses 固定。
- Attention 计算随当前 token length 变化，KV cache 地址也应该 runtime 变化，但传统 PIM 不支持。
- 所以 compiler 只能按最大 token length 生成 instructions 和分配 KV cache，导致大量 unused memory。

### B.Dynamic PIM Access (DPA) Instructions
- DPA 引入 compact dynamic instruction 表达：
  - Dyn-Loop: loop bound 由 request 当前 token length 决定，而不是 $T_{max}$；
  - Dyn-Modi: 在 loop 内按 stride 动态修改 operand fields，比如 MAC 的 row/col address。
- DPA 的结果：
  - instruction size 不再随 context length 线性增长；
  - 可以用 virtual address 表达动态/非连续 KV cache；
  - PIM 在 runtime 通过 VA2PA table 解析到 physical address。

### C.On-Module PIM Instruction Dispatcher
- DPA 需要 PIM HUB 里加 lightweight dispatcher，类似 pseudo-MMU。
- dispatcher 包含：
  - instruction buffer；
  - config buffer（request id、当前 token index 等）；
  - VA2PA table；
  - decoding unit。
- Host 只在新请求进入、需要额外 memory chunk、请求结束释放时和 PIM 交互；不是每个 decoding step 都交互，所以 overhead 小。
- Lazy allocation:
  - host 按 1MB chunks on demand 分配 KV cache；
  - VA2PA 支持 non-contiguous chunks；
  - fragmentation 只发生在每个 request 最后一个 chunk。

## VII.PIMphony Implementation
- PIMphony 用 MLIR 实现 compiler flow：
  - pattern matching 识别 Transformer decoder 中适合 PIM 的 kernels，比如 QKT、SV、FFN；
  - 生成 PIM-specific code；
  - 嵌入 token-centric partitioning 和 dynamic memory allocation metadata。
- 编译离线完成，不影响 inference latency。
- Runtime 基于 IREE/HAL 扩展，负责把 PIMphony instruction sequence 部署到多 PIM modules，并根据 context length 变化自适应。
- 系统集成：
  - 在 NeuPIMs 中，PIMphony offload bandwidth-bound attention 到 PIM，xPU/NPU 处理其他部分；
  - 在 CENT 中，所有 computation 都在 PIM/PIM-like modules 上完成。
- Hardware overhead:
  - I/O-aware buffering per bank 面积约 0.47% MAC area；
  - DCS 让 PIM HUB control blocks area 增加约 0.5%，power 增加约 1.3%；
  - DPA on-module dispatcher area overhead 约 4%，内部 buffers 小于 200KB。

## VIII.Evaluation
### A.Evaluation Settings
- Models:
  - LLM-7B / LLM-72B；
  - non-GQA 版本测试 32K context；
  - GQA-enabled 版本测试 128K context。
- Datasets:
  - LongBench；
  - LV-Eval。
- Baselines:
  - CENT: PIM-only system，16GB per module；
  - NeuPIMs: xPU+PIM heterogeneous system，32GB per module。
- System capacity:
  - 7B 用 128GB；
  - 72B 用 512GB。

### B.Performance Evaluation
- Overall throughput:
  - non-GQA models 上，PIMphony 有 2.1×~4.5× speedup；
  - GQA-enabled long-context models 上，最高 11.3× speedup；
  - xPU+PIM 中最高 8.4×。
- 为什么 72B 上收益更明显？
  - baseline inefficiency 更严重；
  - PIM utilization / memory management 问题更容易成为系统瓶颈。
- 三个技术的作用：
  - TCP 直接提高 channel utilization；
  - TCP 之后系统更容易 I/O-bound，DCS 再通过 overlap I/O/MAC 释放性能；
  - DPA 通过扩大 effective batch size 提高 NPU/xPU utilization，尤其对 xPU+PIM 系统重要。

### C.Ablation Study
- TP vs PP:
  - TCP 让 TP 中的 channel utilization 提高；
  - DCS 减少 attention 的 I/O idle time；
  - DPA 提高 effective batch size，让 PP 也更有优势。
- Energy:
  - baseline attention 中低 MAC utilization 导致 background energy 高，占约 71.5%；
  - PIMphony 加速 attention 后，background energy 占比降到约 13.0%；
  - attention energy 最多降低 3.46×。
- Scalability:
  - system capacity 从 128GB 到 1024GB 扩展时，PIMphony throughput 能持续增长；
  - fixed 512GB 下 context 扩到 1M tokens 时，CENT baseline 因 pipeline bubbles 只剩约 2% utilization，而 PIMphony 可达到 46.6× speedup；
  - NeuPIMs 更稳定，但 PIMphony 仍有 5.0× speedup。
- DPA capacity utilization:
  - static memory management 下 utilization 只有 31.0%~40.5%；
  - DPA lazy allocation 把平均 capacity utilization 提到 75.6%。

## IX.Related Work
- Multi-node LLM serving：TP/PP、continuous batching、paged attention 等主要解决 GPU serving，但 decoding 仍 memory-bound。
- DRAM-based PIM：UPMEM、HBM-PIM、AiM、AiMX 等提供 near-memory bandwidth。
- PIM for LLM inference：CENT、NeuPIMs、AttAcc、IANUS 等证明 PIM 有价值，但没有同时解决 long-context 下 channel utilization、I/O scheduling 和 dynamic KV allocation。
- PIM-based MMU：已有工作更多关注 DRAM ↔ local SRAM transfer，而 PIMphony 是在 PIM 内部做 VA2PA translation，用于 dynamic KV cache allocation。

## X.Conclusions
- PIMphony 的核心判断：long-context 下，瓶颈不只是“要不要用 PIM”，而是 PIM 自己会因为 partitioning、command scheduling、static KV allocation 而严重低效。
- TCP、DCS、DPA 分别从 parallelism、I/O pipeline、capacity management 三个层面解决 PIM inefficiency。
- 最终在 PIM-only 和 xPU+PIM 系统上分别达到最高 11.3× 和 8.4× throughput improvement。


# Adaptive Draft Sequence Length: Enhancing Speculative Decoding Throughput on PIM-Enabled Systems

## abstract
- Speculative decoding 用小 draft language model (DLM) 先生成多个 draft tokens，再用大 target language model (TLM) 并行验证，从而缓解 autoregressive decoding 一次只出一个 token 的瓶颈。
- PIM-enabled heterogeneous systems 适合 speculative decoding，因为：
  - DLM prediction 仍是 sequential GEMV，更 memory-intensive，适合 PIM；
  - TLM verification 同时验证多个 tokens，更 compute-intensive，适合 GPU/xPU。
- 现有系统通常用 fixed draft sequence length。问题是 batch size 大时，很多 draft tokens 会被 TLM reject，导致 DLM 生成和 TLM 验证都做了无效计算。
- 作者提出 SADDLE: a PIM-enabled heterogeneous System that leverages ADaptive Draft sequence LEngths。
- SADDLE 的两个主要机制：
  - asynchronous speculative decoding pipeline，decouple DLM prediction 和 TLM verification；
  - arithmetic intensity-aware operator scheduler，根据 runtime draft length / micro-batch size 动态决定 operator 去 GPU 还是 PIM。
- 结果：平均比 GPU-only speculative/autoregressive baseline 快 2.88×，比最强 GPU+PIM baseline 快 1.71×。

## I.Introduction
- (第一段) LLM decoding 是 autoregressive，生成 T 个 output tokens 需要 T-1 个 sequential decoding steps，因此 throughput 低、compute resource utilization 差。
- (第二段) Speculative decoding 的基本流程：
  - DLM 先预测接下来 d 个 tokens；
  - TLM 并行验证这 d 个 tokens；
  - 如果某个 token 被 reject，它及后续 draft tokens 都丢弃，并由 TLM 输出正确 token。
- (第三段) DLM 和 TLM 的计算特征不同：
  - DLM prediction 一次预测一个 draft token，主要是 memory-intensive GEMV；
  - TLM verification 一次验证多个 tokens，更像 compute-intensive GEMM。
  $\Rightarrow$ GPU/PIM heterogeneous system 很自然：compute-heavy 给 GPU，memory-heavy 给 PIM。
- (第四段) 现有 PIM speculative decoding 系统的问题：fixed draft length。
  - draft 太长：acceptance rate 下降，大量 tokens 被丢弃；
  - draft 太短：TLM verification 太频繁，token-level parallelism 不够。
- (第五段) adaptive draft length 看起来直接，但有三个挑战：
  - 每个 request 的最优 draft length 随模型、任务、iteration 变化；
  - batch 内不同 requests draft length 不同，会导致 DLM/TLM sequential pipeline bubbles；
  - draft length 改变 operator arithmetic intensity，让 offline static operator mapping 失效。
- (第六段) SADDLE 的三个创新：
  - runtime adaptive draft length adjustment；
  - asynchronous speculative decoding pipeline；
  - arithmetic intensity-aware operator scheduling。

## II.Background
### A.Transformer-based LLM Inference
- LLM inference 有 prefill 和 decoding。
- 每个 decoder layer 主要有 QKV generation、MHA、FFN。
- QKV generation 和 FFN 属于 FC operators；MHA 是 attention operator。
- MHA 需要访问历史 K/V，所以 KV cache 随已生成 token 增长。

### B.Parallel Optimization Techniques in LLM Inference
- Batching:
  - 把多个 requests 的 GEMV 合并成 GEMM，提高 request-level parallelism；
  - 但 attention 仍然频繁访问每个 request 的 KV cache，大 batch 下 memory traffic 可能成为瓶颈。
- Speculative Decoding:
  - DLM serially 生成 d 个 draft tokens；
  - TLM parallelly 验证；
  - accepted tokens 直接成为 output，rejected token 由 TLM 修正；
  - 理想情况下，一轮可以输出多个 tokens。

### C.PIM-Enabled Heterogeneous Systems for LLM Inference
- PIM 把 PEs 放在 memory 内部/附近，适合 memory-bound attention/GEMV。
- AttAcc: attention 在 HBM-PIM，其他 operator 在 GPU。
- NeuPIM / IANUS: NPU + PIM heterogeneous system。
- SpecPIM: 面向 speculative decoding，在执行前根据 fixed batch size / fixed draft length 做 offline operator mapping。
- PAPI: 能随 batch size 等动态 remap，但它主要面向单 TLM decoding，不处理 DLM/TLM speculative pipeline 的 idle time。

## III.Motivation
### A.Bottleneck Analysis
- 作者用 SpecPIM 做分析，硬件是 4×A100，每个 GPU 配 5 个 HBM-PIM devices。
- Model pair:
  - TLM: OPT-66B；
  - DLM: OPT-1.3B。
- batch size 增大时，SpecPIM throughput 先上升再下降；batch size=64 时甚至低于普通 autoregressive baseline。
- 根因是 fixed draft length:
  - Figure 3(b): batch=64 时 draft length=4 最优，能比 baseline 快 1.41×；
  - draft length=8 时，因为 rejected tokens 太多，throughput 反而低于 baseline。
- token acceptance rate 随 draft length 增大下降；不同任务（creative_writing、summarization、open_qa）的 acceptance pattern 也不同。
- 不同 speculative decoding iteration 的 optimal draft length 也不同：
  - e.g. creative_writing 中，最优 draft length 会从前几轮的 5 降到后面的 1。
  $\Rightarrow$ 一个固定 d 不可能同时适配所有 request、task、iteration。

### B.Our Proposal: Adaptive Draft Sequence Length
- Adaptive draft length 可以同时减少两类浪费：
  - fixed d 太大时，少生成/少验证无效 tokens；
  - fixed d 太小时，允许简单请求多生成 tokens，提高 TLM verification parallelism。
- 但 naive adaptive 会引入同步问题：
  - batch 内短 draft 的 request 必须等长 draft request 生成完，TLM 才能一起验证；
  - 等待时间形成 pipeline bubbles。
- 另一个问题是 arithmetic intensity 变动：
  - draft length 和 effective batch size 改变后，DLM FC / TLM attention 的 FLOPs/Byte 会变；
  - 某个 operator 在某时刻适合 GPU，另一个时刻可能适合 PIM。
- Figure 6 的结论：static mapping 会失效，频繁重新跑 offline remapping 又太贵。

## IV.The SADDLE Architecture
### A.Architecture Overview
- SADDLE 包含：
  - host；
  - SADDLE Manager；
  - 多个 SADDLE PIM devices，通过 CXL 或 NVLink 等高速互连连接。
- SADDLE Manager 负责：
  - adaptive draft length；
  - DLM/TLM pipeline scheduling；
  - runtime operator scheduling。
- 每个 SADDLE PIM device 包含：
  - centralized processor（实验中是 GPU，也可以是 TPU 等 xPU）；
  - router；
  - 多个 HBM-PIM chips。
- PIM chip 基于 HBM-PIM：
  - 每个 bank 配一个 PE；
  - PE 有 FP16 multipliers/adders，处理 matrix-vector；
  - buffer die 上集成 SFU，处理 residual add、softmax、layernorm、activation 等非矩阵操作。

### B.Execution Flow and Data Mapping
- SADDLE 用 pipeline parallelism，把 PIM devices 分成 S groups，对应 S 个 pipeline stages。
- batch 被拆成多个 micro-batches，让各 pipeline stages 尽量满。
- Weight matrix partitioning:
  - QKV generation 用 row-wise / head-wise partition，让每个 head 的 weights 连续，便于直接进入 MHA；
  - projection 和 FFN 中部分层用 column-wise partition，产生 partial output 后 AllReduce；
  - 第一层/第二层 FC 根据数据流选择 row/column partition，减少通信。
- KV cache mapping:
  - 每个 attention head 分配到特定 HBM stack；
  - $K^T$ 在 pCH/BG 层 column-wise，在 bank 层 row-wise；
  - V 在 pCH/BG 层 row-wise，在 bank 层 column-wise；
  - 目标是 balance load、提高 row buffer reuse 和 internal memory bandwidth。

### C.Adaptive Draft Length Adjustment
- 每个 request 的 draft length 通过 cumulative acceptance probability 决定。
- DLM 在第 t 个 draft step 采样 token $x_t$，其概率为 $p_t$。
- 累计概率 $H_t = \prod_{i=1}^{t} p_i$。
- 当 $H_t$ 低于阈值 $\tau$ 时，说明继续生成的 tokens 很可能被 TLM reject，于是提前停止 drafting。
- 硬件实现：
  - Controller 中有 softmax unit、multipliers、comparators；
  - monitoring/decision overhead 很低，后面实验显示只占 end-to-end latency 的 0.83%。
- 为避免一个 request draft 太长拖住整个 batch，SADDLE 用 cross-micro-batch Shared Pool：
  - draft tokens 先进入 Shared Pool；
  - 当 pool 满或 GPU idle 时，统一送 TLM verification；
  - Manager greedy 优先验证更可能 pass 的 tokens。

### D.Prediction-Verification Decoupled Asynchronous Pipeline
- 仅有 Shared Pool 仍可能让 prediction 和 verification 串行：TLM 在验证时，DLM 可能停下来等。
- SADDLE 用 Eager Pool 解耦：
  - Shared Pool 满后触发 TLM verification；
  - TLM 验证期间，DLM 继续为高置信 request 生成新 draft tokens；
  - 这些新 tokens 暂存在 Eager Pool。
- verification 完成后：
  - 如果该 request 之前的 draft tokens 全部 accepted，则 Eager Pool 里的新 tokens 移到 Shared Pool；
  - 如果有 rejected，则 Eager Pool 里该 request 的 tokens 丢弃，从 TLM-corrected token 重新开始。
- 这个机制基于 optimistic assumption：正在验证的 tokens 会被接受。即使错了，也只是丢弃 Eager Pool 中对应 tokens。
- Shared Pool 和 Eager Pool 都很小，论文配置约 1KB，迁移是 on-chip memory 操作，overhead negligible。

### E.Arithmetic Intensity-Aware Operator Scheduling
- dynamic draft length 和 cross-micro-batch verification 会让每个 operator 的 arithmetic intensity 在 runtime 变化。
- 初始 mapping：
  - DLM attention：一次生成一个 token，low arithmetic intensity，放 PIM；
  - TLM FC：Shared Pool 聚合多个 tokens 后 compute-bound，放 GPU/xPU。
- 动态 mapping 的对象：
  - DLM FC；
  - TLM attention。
- Scheduler 做法：
  - DLM 每次 prediction 后，统计还有多少 request 继续生成，得到 effective micro-batch size；
  - 用 effective batch size 估计 DLM FC arithmetic intensity，和预先 profile 的 PIM/GPU threshold 比较；
  - TLM verification 前，统计 Shared Pool 中每个 request 的 draft token 数，估计 TLM attention arithmetic intensity；
  - 决定 operator 放 PIM 还是 GPU。
- 这相当于把 PAPI 那类“根据 runtime parallelism 判断 GPU/PIM”的思想，扩展到 speculative decoding 的 DLM/TLM 双模型场景。

## V.Evaluation
### A.Experimental Setup
- TLM/DLM pairs:
  - OPT-66B + OPT-1.3B；
  - OPT-175B + OPT-6.7B；
  - Llama-3.1-70B-Instruct + Llama-3.2-1B-Instruct。
- Dataset:
  - Dolly instruction-following dataset。
- Batch size:
  - 16 到 128；
  - 大多数实验 max sequence length=1024；
  - OPT-175B 因 memory 限制 max sequence length=512。
- Baselines:
  - GPU-AD: GPU 上普通 autoregressive decoding；
  - GPU-SD: GPU 上 speculative decoding；
  - PIM-AD: GPU+PIM 上普通 autoregressive decoding；
  - PIM-SD: GPU+PIM 上 speculative decoding（SpecPIM-like）。
- Hardware:
  - GPU baselines: 8×A100 80GB，总 640GB；
  - PIM baselines/SADDLE: 8 个 SADDLE PIM devices，每个 device 5 个 HBM stacks，总容量对齐 640GB；
  - simulator 基于 Ramulator2 和 ATTACC 修改。

### B.Overall Performance
- Throughput:
  - SADDLE 平均比 GPU-AD 快 3.36×；
  - 比 GPU-SD 快 2.88×；
  - 比 PIM-AD 快 1.94×；
  - 比 PIM-SD 快 1.71×。
- 小 batch（16/32）时，普通 speculative decoding 也有收益；
- batch 变大后，fixed-length speculative decoding 的优势下降甚至反转，因为 invalid draft tokens 消耗大量资源；
- SADDLE 在大 batch 下仍保持优势，因为 adaptive draft length 减少无效 token，并且 asynchronous pipeline 减少等待。
- Energy efficiency:
  - SADDLE 平均比 GPU-AD、GPU-SD、PIM-AD、PIM-SD 分别高 6.81×、5.96×、2.32×、1.45×。
- 原因：
  - PIM 减少 memory-bound operators 的 GPU global memory traffic；
  - adaptive drafting 减少 invalid tokens；
  - operator scheduling 提高 GPU/PIM utilization。

### C.Ablation Analysis
- SADDLE-d：只用 adaptive draft length。
  - 反而比 PIM-SD 平均低 1.22×，说明 adaptive 本身不够；因为会引入 pipeline bubbles 和 operator intensity fluctuation。
- SADDLE-p：adaptive + Shared Pool cross-micro-batch verification。
  - 比 SADDLE-d 快 1.52×；
  - 比 PIM-SD 快 1.25×。
- 加 Eager Pool：
  - 进一步 1.24× speedup；
  - 因为 DLM prediction 和 TLM verification 可以 overlap。
- SADDLE-s / dynamic operator mapping：
  - 再带来 1.13× improvement。
- 结论：三个机制是互补的，adaptive draft length 必须配合 pipeline decoupling 和 dynamic scheduling 才真正有效。

### D.Sensitivity Analysis
- Sequence length:
  - sequence 越长，attention 执行时间越高，PIM offloading 更有价值；
  - SADDLE 相对 GPU-AD 的优势随 sequence length 增长而扩大。
- Batch size:
  - 小 batch 下 invalid draft computation 不严重，SADDLE 接近 PIM-SD；
  - batch 增大后，SADDLE 的 adaptive + async pipeline 优势更明显。

### E.Performance Discussion
- Latency breakdown:
  - monitoring 和 decision-making 只占 SADDLE end-to-end latency 的 0.83%；
  - adaptive draft length 把 prediction/verification latency 分别降低 1.18× 和 1.23×；
  - decoupled asynchronous pipeline 带来 1.73× end-to-end latency reduction。
- Communication cost:
  - TLM verification 中 cross-pCH communication 只占 13.54%；
  - cross-stack communication 约 7.51%，不是主要瓶颈。
- Hardware utilization:
  - 相比 PIM-AD 和 PIM-SD，SADDLE 分别提高 GPU utilization 1.13× / 1.37×；
  - 提高 PIM utilization 1.84× / 1.18×。
- Operator scheduling:
  - 没有 arithmetic intensity-aware scheduling 时，operator 分布约 9.51% 在 PIM、90.49% 在 GPU；
  - 开启后变为 14.89% 在 PIM、85.11% 在 GPU；
  - 带来约 1.21× throughput improvement。

### F.Area Overhead
- SADDLE 增加约 16.24 mm² per DRAM die 和 1.62 mm² per buffer die。
- 每个 pCH 有 16 PEs 和 4 accumulators。
- PE area 中：
  - arithmetic units 约 57%；
  - on-chip buffers 约 16%；
  - control logic 约 27%。
- 相对 121 mm² HBM3 die，PIM logic 约 13.4% DRAM-die area overhead，作者认为与已有 HBM-PIM accelerator comparable，不会显著损害 memory capacity。

## VI.Related Work
- ASIC-based LLM accelerators：A3、SpAtten、Sanger、ELSA、DOTA 等多通过 sparsity/approximation 降低 attention cost，但主要针对 attention，难以在大 batch 下提供完整 end-to-end speedup。
- PIM-based LLM accelerators：
  - TransPIM、CENT 等强调 memory-centric / GPU-free PIM；
  - AttAcc、IANUS、SpecPIM、PAPI 等强调 GPU+PIM heterogeneous scheduling；
  - SADDLE 的区别是针对 speculative decoding，联合 adaptive draft length、batching 和 PIM/GPU runtime scheduling。
- Adaptive draft sequence length：
  - Disco、OPT-Tree 等关注单请求 latency；
  - SADDLE 关注 large-batch speculative decoding throughput，并用硬件友好的 adaptive mechanism + async pipeline 解决 batch 内 draft length 不一致导致的 stalls。

## VII.Conclusion
- Fixed draft length 在 PIM-enabled speculative decoding 中会造成两类问题：
  - draft 太长导致 rejected tokens 过多；
  - draft 太短导致 TLM parallelism 不足。
- Adaptive draft length 必须配合系统级支持，否则会引入 pipeline bubbles 和 dynamic operator mismatch。
- SADDLE 用 runtime draft-length tuning、prediction-verification decoupled pipeline、arithmetic intensity-aware scheduling 三个机制，把 adaptive drafting 真正落到 PIM+GPU heterogeneous system 上。
- 实验结果：平均比 GPU-only 系统快 2.88×，比现有 PIM-enabled speculative decoding 系统快 1.71×。


# LIA

LIA 提出一种面向单 GPU 大模型推理的加速框架：利用 AMX 加速的 CPU 与 GPU 协同计算，把每个 decoder layer 拆成 6 个子层，并用向量策略决定哪些子层在 CPU、哪些在 GPU 执行，通过对每个子层的 load/compute/store 代价建模来自动选出在给定 batch/序列长度下的最优划分，从而在显存不足时减少 PCIe 数据搬运并提升吞吐/降低延迟；同时在吞吐优先的大 batch 场景引入 CXL 内存扩容，采用“参数放 CXL、KV cache 留 DDR”避免 KV 相关子层被 CXL 带宽/延迟拖慢
未来工作：
（1）扩展 LIA 的协同计算框架到 多 GPU （多 GPU 后，不只是 CPU-GPU 传输，GPU-GPU 之间的通信也会变瓶颈）
（2） 从“跑前选一次策略”变成“运行时根据实时负载自动换策略”

# Exploiting Intel Advanced Matrix Extensions (AMX) for Large Language Model Inference

针对“参数/KV/激活放 CPU 内存、GPU 通过 PCIe 按层搬运”时推理被 PCIe 传输主导的问题，利用 SPR 上的 AMX 让 CPU 具备可观矩阵计算能力，并提出基于层算术强度（ARI）的自适应分区：当层 ARI 较低、传输开销占比高时优先在 CPU 侧计算以避免 PCIe 往返，高 ARI/算力密集部分交给 GPU，同时对 prefill 的注意力打分层做局部性例外处理
未来工作
(1) 把ARI 阈值驱动的层级分区这种思想应用到一个更完整的系统里
