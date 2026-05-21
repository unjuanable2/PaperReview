# uGEMM: Unary Computing for GEMM Applications

## abstract
- GEMM/general matrix multiplication 在 signal processing、computer vision、machine learning 里都很核心。
- 传统 binary GEMM 用空间上的多 bit 表示和并行计算，随着并行度升高会遇到 area / power / energy scalability 问题，尤其是 wire congestion。
- Unary computing 把数值放到 temporal domain，用 serial bit stream 表示，计算逻辑极简单，但过去缺少高效的 unary GEMM 架构。
- 作者提出 uGEMM：
  - 设计 universally compatible arithmetic units，可以同时支持不同 unary input coding；
  - 兼顾 input-insensitivity 和 high output accuracy；
  - 支持 early termination，用 accuracy-aware metric 做 dynamic energy-accuracy scaling；
  - 开源 UnarySim，方便后续 unary computing 研究。

## Introduction
- (第1段) GEMM 是很多应用，尤其 deep learning 的基础操作。Binary GEMM 可以通过硬件获得效率，也可以通过软件获得灵活性，但 binary 表示本质上是 spatial，多个 parallel bits 会带来面积、功耗、能耗和布线扩展性问题。
- (第2段) Unary computing 用 serial bit streams 处理数据，把计算从 spatial domain 转到 temporal domain。好处是硬件逻辑极小、功耗低；代价是 latency 长。已有 unary computing 包括：
  - rate-coded stochastic computing；
  - temporal-coded race logic。
  但这些方案往往是 application-specific。Stochastic computing 支持 arithmetic 和 relational operations，但容易因为 correlation 出错；race logic 准确但操作类型有限，不能直接支持 GEMM。
- (第3段) 理想 unary GEMM 有三个挑战：
  - 哪些 unary computing units 能组合成 GEMM？
  - 如何在提高能效时保证可靠 accuracy？
  - 如何模拟和探索 unary design space？
  作者用 uGEMM 架构和 UnarySim simulator 回答这些问题，希望让 unary computing 更适合资源受限系统。
- (第4段) 针对第一个挑战，作者设计新的 unary arithmetic units 和 compatible microarchitectures。
  - 这些 units 支持 arbitrary input bit streams，即 input-insensitive；
  - 不需要 unary-binary conversion，能够 fully streaming；
  - 因此多个 uGEMM block 可以连续 stream。
- (第5段) uGEMM 是由这些 units 组成的 MAC array。第二个挑战通过 stability metric 解决：它监控 output accuracy 随时间的收敛过程，使系统能够 reliable early terminate，在满足 accuracy 约束时减少能耗。第三个挑战通过开源 UnarySim 解决，可以做 microarchitecture / architecture / application level evaluation。

## Background
### Rate-Coding-Based Unary Computing
- Stochastic computing 使用 rate coding，数值由 bit stream 里 1 的频率表示。每个 bit 通常通过 source data 和 random number generator/RNG 比较得到。
- 根据 polarity，bit stream 可以是：
  - unipolar: value = P(S=1)，范围 [0,1]；
  - bipolar: value = 2P(S=1)-1，范围 [-1,1]。
- Stochastic computing 通过统计方式操作 bit streams：
  - MUX + stochastic select stream 可以做 scaled addition；
  - OR gate 可以做 unipolar nonscaled addition；
  - AND gate 可以做 unipolar multiplication；
  - XNOR gate 可以做 bipolar multiplication。
- 问题是 correlation。AND gate 只有在两个输入 bit stream zero correlation 时才准确表示乘法；如果 paired ones 过多，结果会偏向 minimum function。已有解决办法通常需要更贵的 RNG 或更长 latency。

### Temporal-Coding-Based Unary Computing
- Race logic 使用 temporal coding，数值编码在 signal transition/edge 的时间位置上，bit stream 通常是一串 1 后接一串 0，或者反过来。bit 由 source data 和 counter output 比较产生。
- 在 race logic 里，AND / OR 不再表示 stochastic multiplication / nonscaled addition，而是准确表示 minimum / maximum。
- Race logic 的优点是 deterministic edge，所以 min/max 很准；但在本文之前，temporal computing 还没有可用的 multiplier 和 adder，因此无法直接支持 GEMM。

## uGEMM Microarchitecture and Architecture
### uMUL
- 作者提出 unipolar 和 bipolar uMUL。
- 核心观察：naive AND gate multiplication 只有当 input0 为 1 时才会产生有效 output bit，所以可以把 input0 看成 input1 bitstream generator 的 enable signal。
- 具体做法：
  - input0=1 时，input1 generator 的 RNG 才更新并生成新 bit；
  - 这样等价于 conditional bit stream generation；
  - 可以从源头消除 correlation problem，只额外需要一个 enable signal。
- 对 bipolar multiplication，作者把 XNOR 分解成两个 AND gates，然后每个 AND gate 都用 conditional bit stream generation 来保持准确性。

### uSADD
- Scaled addition 计算所有输入的 average，输出 scaling 来自平均操作本身。
- unipolar 和 bipolar 的 uSADD microarchitecture 相同。
- 作者的思路是：input average 等于 input-one counts 的均值。
  - 先用 parallel counter 收集所有输入 bit 中 1 的数量；
  - accumulator 累加这些 count；
  - output 取 accumulator carry bit。
- 当累计到 N 个 ones 时才输出一个 1，正好实现 output scaling。由于只依赖 one count，不依赖 bit streams 之间的具体 alignment，因此消除了 correlation 影响。

### uNSADD
- Nonscaled addition 计算 clipped sum；当 sum 超出合法 unary range 时做 clipping。
- uNSADD 前半部分也用 parallel counter 和 accumulator，但不只取 carry bit，而是使用完整 accumulation output。
- 然后用 offset 减去 accumulation result，恢复期望 output-one count。
  - unipolar offset = 0；
  - bipolar offset = (N-1)/2。
- 最后根据 anticipated one count 和 actual one count 比较来生成输出：如果期望 ones 多于当前 actual ones，就持续输出 1，直到二者相等。
- 与 uSADD 类似，它基于 count 而不是 input correlation 工作。作者强调这是第一个支持 bipolar nonscaled addition 的 unary design。

### Integrated uGEMM Architecture
- 因为 uMUL、uSADD、uNSADD 都 input-insensitive 且 fully streaming，uGEMM 可以直接由它们组成。
- 作者用 `O = A * B + C` 举例：
  - A, B, C 进入 m-by-n PE array；
  - 第 (i,j) 个 PE 读取 A 的第 i 行、B 的第 j 列，做 k-element MAC；
  - 再加上 C 的第 (i,j) 个元素得到输出。
- uGEMM 继承三个关键特性：
  - unary logic 简单，因此 parallelism 高；
  - 通过减弱/消除 correlation problem 获得 input-insensitivity 和 high output accuracy；
  - fully streaming，因此可以可靠 early terminate。

## Energy-Accuracy Scaling Evaluation
- 作者把 GEMM shape 固定为 `m=k=n=16`，比较 uGEMM 与 Gaines、Sim、Jenson、Najafi 等 unary schemes。
- 重要公平点：
  - uGEMM 是唯一支持 bipolar nonscaled addition 的方案；
  - Najafi 的设计不支持 temporal coding。
- 硬件结果：
  - uGEMM 面积和功耗较小，优于 Gaines、Sim、Najafi；
  - Jenson 的设计面积/功耗更低，因为用了 resource sharing，但 latency 极长，最终 energy 更高。
- Early termination 的意义：partial bitstream 表示一个不太准确的值，但可以减少 latency 和 energy。
- 作者提出 stability metric 衡量 bit stream 多早进入稳定状态。
  - 对长度 L 的 bitstream，`V_L` 是最终 expected value，`V_l` 是前 l bits 的 partial value；
  - 若从第 l bit 开始，bias `|V_l - V_L|` 一直小于用户设定阈值 `VTHD`，则可认为已经稳定；
  - stability 越高，说明越早可以 terminate。
- 通常 rate coding 的 stability 高于 temporal coding。给定 `VTHD` 后，最大不稳定位置称为 stable point，此后 output accuracy 统计上不会低于阈值。
- 实验设置 `VTHD=0.05`，比较 progressive accuracy 和 stable point。
  - uGEMM 在 final accuracy 上优于所有 baseline；
  - 对不同 inputs 的差距也小，说明 input-insensitive；
  - stable point 更靠近 y-axis，说明同样 error budget 下可以更早 early terminate。
- 结论：uGEMM 不只是硬件开销低，还能通过 stability 指标可靠地做 energy-accuracy scaling，因此更适合 ultra-low-power architecture。

## UnarySim: A Unary Computing Simulator
- 背景问题：已有工具可以 synthesis unary circuits，但缺少公开的 toolchain 来快速做 general unary architectures / applications 的 design space exploration 和 simulation。
- 作者开源 UnarySim，用来标准化地比较不同 unary designs 的 tradeoff。
- UnarySim 是 cycle-accurate simulator，包含两部分：
  - software simulation；
  - hardware implementation examples。
- Software simulation 基于 PyTorch。
  - 好处是天然支持 deep learning applications；
  - high-level architecture design 和 low-level performance simulation 解耦；
  - 可扩展性更好。
- UnarySim 的关键 components 分三类：
  - stream components：生成 rate-coded 或 temporal-coded bit streams，并控制 pairwise correlation；
  - kernel components：实现 unary arithmetic / nonlinear / logical operations，包括本文提出的 uGEMM units；
  - metric components：包括 correlation、accuracy、stability 等，可以 pin 到系统中任意节点监控状态。
- 调优完成后，metric components 可以移除以加速模拟。
- 所有 software components 都是 PyTorch modules，并提供 test examples，降低上手门槛。
- Hardware implementation 部分提供本文和 prior works 中若干 component-level examples。
- 限制：当前 metric components 只在 software 中模拟，还没有对应硬件实现。未来工作包括扩展 components 和自动把 arbitrary algorithms 映射到 unary hardware。

## Conclusion
- Unary computing 因为 ultra-low power，在 signal processing、computer vision、machine learning 等领域越来越受关注。
- 本文面向更通用的 GEMM 操作提出 uGEMM。
- uGEMM 相比 prior art 的优势：
  - input-insensitivity；
  - output-stability；
  - hardware efficiency。
- 作者用 stability metric 展示了 accuracy-aware energy-accuracy scaling，可以通过 early termination 进一步节能。
- UnarySim 则为 unary computing 提供了更低门槛的研究和设计空间探索工具。


# Zero Correlation Error: A Metric for Finite-Length Bitstream Independence in Stochastic Computing

## abstract
- Stochastic computing/SC 用概率式 unary bitstream 表示数据，能用很简单的电路实现复杂操作。
- 但 SC 与传统 binary computing 不同，必须小心处理 bitstreams 之间的 correlation，否则结果会严重不准确。
- 很多 SC circuits 假设输入 bitstreams 独立，因此需要准确衡量 finite-length bitstreams 的 independence。
- 作者提出 Zero Correlation Error/ZCE，用来量化两个 finite-length bitstreams 有多独立。
- ZCE 解决了 SC 社区常用 SCC/stochastic cross correlation 在衡量 independence 时的根本限制，并在 functional unit level 和 application level 展示用途。

## 1 Introduction
- (第1段) SC 是重新受到关注的 computing paradigm，应用包括 image processing、error correction codes、neural networks。它用 bit-serial unary bitstreams，而不是 bit-parallel binary registers。数值由 bitstream 中 bit=1 的概率表示，因此乘法可以简化成一个 AND gate。
- (第2段) SC 的问题是数据值必须谨慎处理。很多关键操作，如 multiply 和 scaled add，要求输入 bitstreams statistically independent。实际 SC bitstreams 有限长且不是真随机，因此直接套用概率论里的 independence 定义并不简单。当前社区常用 SCC 来衡量 cross-correlation。
- (第3段) SCC 通过 bitstreams 里 0/1 的 alignment 来描述 similarity。对需要 independent inputs 的 functional unit，理想情况是 SCC=0。但 finite-length bitstreams 下 SCC=0 不一定可达。直觉上可能会选择 `|SCC|` 最接近 0 的 alignment，但 Figure 1 的 counterexample 说明这不总是 error 最小的 alignment。
- (第4段) Figure 1 中，`P_X=3/16`、`P_Y=14/16` 做 AND multiplication 时有多种 alignment。按 SCC 选出来的 alignment error 为 `-10/256`，而真正 error 最小的 alignment error 为 `6/256`。所以 SCC 可以描述 correlation，但在 SCC 无法达到 0 的情况下，不一定能判断最独立 alignment。
- (第5段) 其他已有 error analysis 多关注与 correlation 无关的误差，或者混合多个 error source，因此不能 isolate input independence 带来的误差。
- 贡献：
  - 证明 bitstream independence 与 SCC 之间存在 disparity；
  - 详细分析 SCC，发现对 64-bit bitstreams，SCC 平均约 20% 情况不能正确识别最独立 alignment；
  - 提出 ZCE，并从 probability theory 推导；
  - 展示 ZCE 在 design space exploration、profiling、optimal bitstreams selection 中的用途。

## 2 Background and Motivation
### 2.1 Stochastic Computing
- SC 把数据编码成 unary bitstreams，数值由 bit=1 的概率表示。
- 例如长度 16 的 bitstream 里有 14 个 1，就表示 `14/16`，不关心这些 1 具体出现在哪些位置。
- SC 通过 probability transformation 和简单 logic gate 做计算。优势是 arithmetic unit 很小，例如 multiplication 只需要 AND gate。

### 2.2 Independence of Bitstreams
- 概率论中，若 `P(X∩Y)=P(X)P(Y)`，则两个事件 independent。
- finite-length bitstreams 下，`P_X P_Y` 需要比 bitstream 本身更细的精度，不一定能由 AND result 精确表示。
- 因此作者把 AND-gate result `P_{X∧Y}` 称为 finite-length joint probability。
- 如果 `P_{X∧Y}=P_XP_Y`，则两个 finite-length bitstreams 在这个有限长度下已经“尽可能 independent”。
- 因为 `P_{X∧Y}` 取决于 0/1 alignment，所以衡量 independence 必须分析 alignments/cross-correlation。

### 2.3 SCC and Limitations
- SCC 是当前衡量 SC bitstreams cross-correlation 的 de facto metric。
- aligned 0s/1s 越多，bitstreams 越 positively correlated；misaligned 越多，越 negatively correlated。
- 作者的关键洞察：finite-length bitstreams 下，SCC 和 independence closely related，但并不等价。
- SCC 的公式会把 maximally positive correlation 映射到 +1，把 maximally negative correlation 映射到 -1，并在二者之间线性插值。
- SCC 对判断“更正相关”或“更负相关”很有效。
- 但当 functional unit 偏好 independent inputs 时，用 `|SCC|` 接近 0 作为 accuracy proxy 会失效，因为 positive side 和 negative side 归一化斜率不同。
- 作者枚举 finite bitstream length 下所有 representable value pairs，比较 SCC 推荐的 most uncorrelated alignment 和真正 minimizing `|P_{X∧Y}-P_XP_Y|` 的 most independent alignment。
  - Figure 2 显示，SCC 平均约 18% 情况不能识别 maximally independent alignment；
  - Figure 3 显示错误并非均匀分布，small value + large value 这类组合更容易出问题。
- 因此需要一个专门衡量 finite-length bitstream independence 的 metric。

## 3 Zero Correlation Error
- ZCE 的两个目标：
  - alignment 越 independent，metric 越接近 0；
  - 对 finite bitstream length 下所有 most independent alignments，都输出 0。
- 这样 ZCE 不只能评价 bitstream independence，还能作为 accuracy proxy，并方便 SC circuit analysis / design exploration。

### 3.1 SCC vs. ZCE
- 作者用 `P_X * P_Y = 21/32 * 22/32` 的所有 possible alignments 作图，比较 SCC、true quantization error、ZCE。
- SCC 会把两个端点 A 和 D 分别 normalize 到 -1 和 +1，用于表示最大负相关和最大正相关。
- 这个 normalize 让 crossing 0 的两侧斜率不同，导致 `|SCC|` 最接近 0 的点不一定是 absolute error 最小的点。
- 在例子中：
  - true error 最小的是 B；
  - `|SCC|` 最小的是 C。
- ZCE 的视觉目标：
  - 保留 quantization error 曲线的斜率；
  - 把接近 0 的 crossing “加厚”，使 finite-length 下最小正误差和最小负误差若同等接近，都可视作 ZCE=0。
- ZCE 保留 error direction：
  - positive ZCE 表示偏 positive correlation；
  - negative ZCE 表示偏 negative correlation。
- Takeaway：SCC 仍适合判断 maximum positive / negative correlation；ZCE 则用于判断 maximum independence。

### 3.2 Derivation of ZCE
- 推导从 quantization error 开始。
- 对 finite-length bitstreams，理想 product `P_XP_Y` 可能需要 `L^2` 精度，而 AND result `P_{X∧Y}` 只有 `L` 精度，因此即使是最独立 alignment 也会有 quantization error。
- 作者定义 `Δ0` 为 product quantized 到 `1/L` 粒度后与真实 product 之间的 error。
- 用 alignment counts 表示时，若 `(a,b,c,d)` 分别是 `(1,1),(1,0),(0,1),(0,0)` 的数量，则 `a+b` 是 X 中 1 的数量，`a+c` 是 Y 中 1 的数量。`Δ0` 与具体 alignment 无关，只由输入值决定。
- 对当前要评价的两个 bitstreams，actual error 定义为：
  - `Δ = P_{X∧Y} - P_XP_Y`
  - 等价于 `a/L - ((a+b)(a+c))/L^2`。
- ZCE 的核心是从 actual error 中减去 finite precision unavoidable 的部分：
  - `ZCE = Δ / |Δ| * (|Δ| - |Δ0|)`
- 因此，若当前 alignment 已达到 finite-length 下最独立状态，ZCE=0；若还可进一步接近 independence，ZCE 的绝对值会反映距离。

## 4 Evaluation
### 4.1 Use Case: Finding Optimal Alignment
- 问题：如何对齐 input bitstreams，让 circuit error 最小？
- 对偏好 independent inputs 的 functional units，比如 AND-gate multiplication 和 MUX-based scaled addition，ZCE 可以直接指导 alignment selection。

#### Multiplication
- 作者枚举所有可能输入值，用 SCC 和 ZCE 分别推荐 AND-gate multiply 的 optimal alignment。
- ZCE 因为定义直接对应 finite-length independence，所以推荐 alignment 的 AND result 都等价于 quantized product，是能达到的最优结果。
- SCC 推荐 alignment 的 average error 会高于 quantization error；bitstream length 越长，每个 bit 对总值影响越小，所以误差差距逐渐减小。

#### Scaled Addition
- 作者用 MUX 实现 scaled addition，以 `P_Z=(P_X+P_Y)/4` 为例。
- MUX select bitstream 固定为 `1/4`，SCC/ZCE 分别为两个输入 stream 相对 select stream 选择 alignment。
- ZCE 的 error 低于 SCC，但不一定达到仅有 quantization error 的最优。
- 原因是 2-to-1 MUX 逻辑 `Z = !S X + S Y` 有两个 AND-like components，正负误差可能互相抵消。ZCE 对每个 component 单独找最 independent alignment，不主动利用这种 cancellation。

#### 2D Convolution
- 作者在 application level 做 2D convolution benchmark。
- 输入是 256x256 grayscale images，卷积核是 3x3 Gaussian blur。
- 每个 multiplication 用 SCC 或 ZCE 决定 pixel value 与 weight value 的 best alignment，然后用 parallel adder 累加 9 个 products。
- 与 floating-point baseline 比较 RMSE。结果显示 ZCE 在所有 bitstream lengths 下都比 SCC 产生更低 error。
- Takeaway：ZCE 在 functional unit 和 application level 都能更稳定地推荐 lower-error alignments。

### 4.2 Use Case: Profiling Bitstreams
- 问题：如何快速判断 bitstreams 是否已经达到最独立 alignment？
- 用 ZCE 很简单：检查 ZCE 是否等于 0。
- 用 SCC 则更麻烦，因为很多输入值组合根本不存在 SCC=0 的 alignment，所以通常还要计算第二个 alignment，看哪个 `|SCC|` 更接近 0；即便如此也不一定真最独立。
- 实验中，作者对所有 possible input value pairs 生成 random alignments，比较 SCC/ZCE 判断 optimal alignment 的 operation count 和 wall-clock time。
- SCC 大多数时候需要算两次；且 bitstream length 越长，不存在 SCC=0 的 value pairs 越多：
  - length 64 时约 91%；
  - length 1024 时约 99%。
- 尽管 ZCE 单次计算更复杂，整体 wall-clock 仍约快 1.45x。
- Takeaway：ZCE 把 optimal independence 统一定义为 ZCE=0，因此 profiling/debugging 更直接、更快。

### 4.3 Use Case: Design Space Exploration
- 问题：如何比较不同 random number generator/RNG 设计，选择最适合 SC circuit 的方案？
- RNG 会影响不同 bitstreams 之间的 cross-correlation，进而影响计算 accuracy。

#### Multiplication
- 作者比较不同 LFSR feedback taps 形成的 LFSR pairs。
- 对 6-bit LFSR，有 6 种设计，两两组合得到 21 个 configurations；8-bit LFSR 同理。
- 对每个 configuration，枚举所有 value pairs，计算 average SCC、average ZCE 和 true computation error。
- 在 AND-gate multiply 中，ZCE 给出的 configuration ranking 与 true error ranking 完全一致；SCC ranking 有偏差。
- Kendall's tau：
  - 6-bit LFSR: ZCE=1，SCC=0.82；
  - 8-bit LFSR: ZCE=1，SCC=0.83。

#### 2D Convolution
- Application-level 同样用 3x3 Gaussian blur benchmark。
- 输入 pixel values 共用一个 RNG，weight values 共用另一个 RNG；每种 bitstream length 下比较 21 个 LFSR-pair configurations。
- ZCE ranking 不再 100% 等同 true error ranking，因为 application 中可能存在正负误差抵消。
- 但 across different bitstream lengths，ZCE 的 Kendall's tau 始终高于 SCC。
- Takeaway：ZCE 比 SCC 更适合 pruning design space 和比较 SC number generator 的实现/参数。

## 5 Related Work
- SCC 用于衡量 SC bitstreams 的 cross-correlation，能判断 positive / negative correlation 程度；ZCE 则专门用于判断 independence。
- Probabilistic transfer matrices/PTM 是分析 SC circuit correlation-induced error 的代数框架：用户给出 input combinations 的概率向量，再与 logic function matrix 相乘得到 expected output。
- 也有工作用 mixed integer programming 自动合成 number sequences，以最大化 SC circuit accuracy。
- ZCE 与这些工作正交：它不是 synthesis tool，而是 evaluation metric。
- 另一类相关工作关注 autocorrelation，即一个 bitstream 与其 delayed version 的 correlation。
  - SBoNG 设计低 autocorrelation RNG；
  - SANG 生成具有指定 autocorrelation property 的 bitstreams。
- 这些工作主要研究 number generation；ZCE 提供的是衡量 bitstream independence 的方法。

## 6 Conclusion
- 本文提出 ZCE，一个用于 finite-length SC bitstreams 的 independence metric。
- 作者通过分析 SCC，指出 SCC 在衡量 bitstream independence 时存在根本限制。
- ZCE 的优势：
  - 更好的 alignment recommendation，降低 functional unit 和 application error；
  - 更快判断 bitstreams 是否 optimally independent；
  - 更准确比较 RNG variants 和设计参数。


# Streaming Accuracy: Characterizing Early Termination in Stochastic Computing

## abstract
- SC 可以用很小面积实现复杂计算，但代价是 accuracy 损失和更高 latency。
- Early termination 是降低 SC latency 和 area/energy cost 的常用技术。
- 因此需要衡量一个 bitstream 是否适合 early termination。
- 作者提出 Streaming Accuracy，衡量一个 bitstream 距离“最适合 early termination 的形式”有多远。
- 作者说明该 metric 克服 prior work 局限，刻画 stochastic circuits 的 early termination design space，并提出能生成 optimal streaming accuracy bitstreams 的硬件 bitstream generator。

## I. Introduction
- (第1段) SC 用 bit-serial unary bitstreams 表示数据，而不是 bit-parallel binary registers。数值由 bit=1 的概率表示，例如 `0010` 和 `1000` 在 unipolar format 中都表示 `1/4`。这种表示让 functional unit 极小，例如 multiplication 可以用一个 AND gate 实现，因此适合 low power / low area 计算。
- (第2段) SC 的代价是 latency。为了达到接近 binary counterpart 的 accuracy，SC 往往需要指数级更高 latency。Early termination 可以缩短计算，是让 SC 更可行的重要技术。因此设计者需要知道提前停止会带来多少 error。
- (第3段) 现有两个相关指标：
  - progressive precision/PP；
  - normalized stability。
  它们各有适用场景，但不能回答“任意 termination point 下哪个 bitstream 更好”。
  - PP 只看 powers-of-2 的 partial lengths；
  - normalized stability 需要用户给定 error threshold，而这个 threshold 不容易设。
- (第4段) 作者提出 streaming accuracy，用来衡量 bitstream 距离 most early-terminable form 有多近。Streaming accuracy=1 表示任意 early termination point 下都达到可能的最高 accuracy。
- (第5段) Streaming accuracy 可以帮助设计者分析 SC circuit 中哪些 component 提升或破坏 early termination 能力，也可以在没有 application-specific error tolerance 的情况下做 design space exploration。
- (第6段) 作者还提出 streaming-accurate/SA generator，硬件上总能生成 streaming accuracy=1 的 bitstreams。实验中 SA generator 在 early termination 下比 Halton 和 LFSR generators error 更低。
- 贡献：
  - 提出 streaming accuracy metric；
  - 分析不同 functional units 对 streaming accuracy 的 retention；
  - 分析 early termination 与 bitstream independence 的关系；
  - 定义 streaming-accurate bitstream，并设计对应 generator。

## II. Background and Related Work
### A. Early Termination Metrics
#### Progressive Precision
- Progressive precision/PP 衡量 stochastic computation 是否能用速度换 accuracy。
- 对长度 L 的 bitstream X，如果所有 `l=2^i` 的 initial partial bitstreams 的 bit-error 都不超过 k，则称 X 是 k-PP。
- bit-error 定义为 `l * |P_{X_l} - P_X^*|`。
- 例子中，`X=0111111111110000`、目标值 `10/16` 时，长度 2、4、8、16 的 partial bit-error 分别为 0.25、0.5、2、1，因此 X 是 2-PP。
- 限制：PP 只检查 powers-of-2 partial lengths，而 streaming accuracy 会检查任意 partial length。

#### Normalized Stability
- Normalized stability 把 bitstream 进入 stable state 的时间归一化。
- Stable state 表示 bitstream value fluctuation 低于用户指定 error threshold。
- 例如 threshold=10% 时，表示 0.5 的 `1010101010101010` 最稳定，8 bits 后就保证在 threshold 内；`0000111100001111` 要 12 bits 后才稳定，因此 normalized stability 较低。
- 限制：normalized stability 需要用户提前给 error threshold；没有应用知识时很难设。

### B. SC Bitstream Generator
- 典型长度 L 的 SC bitstream 由 comparator + RNG 生成：
  - 每个 cycle 产生一个 random number；
  - 如果 random number 小于目标数值，则输出 1，否则输出 0。
- 例如长度 8、目标值 `4/8`，RNG 输出 0 到 7，与 4 比较，统计上生成 4 个 1。
- 常见 RNG 包括 LFSR、Halton、Sobol 等 low-discrepancy sequences。

## III. Streaming Accuracy
### Limitations of Prior Studies
- Figure 1 用 normalized stability 和 progressive precision 选择表示 `1/32` 的 bitstreams，展示它们无法唯一识别真正最适合 early termination 的 bitstream。
- 对 normalized stability，threshold=5% 和 10% 时都有多个 bitstreams 达到最高 score，但它们的 early termination accuracy 并不一样。
- 对 progressive precision，也有多个 bitstreams 得到相同最小 k 值，但其中很多在非 powers-of-2 partial lengths 上明显劣于最佳 bitstream。
- PP 的根本限制是只看 powers-of-2 lengths，因此不能保证任意 termination point。
- Normalized stability 的根本限制是 threshold-dependent；threshold 越小，认为安全的 termination cycle 越靠后，而且 threshold=0 时不可用。
- 因此这两个指标适合各自场景，但不适合作为 generalized early termination metric。

### Streaming Accuracy Definition
- 作者希望 metric 能反映任意 early termination point 下的 accuracy。
- 对长度 L、表示值 `P_X` 的 bitstream，它有 L 个 partial values `P_{X_i}`。
- 作者定义 raw streaming accuracy：
  - `Phi = 1 - (sum_i |P_{X_i} - P_X|) / L`
- 对同一个 value，early termination 最好的 bitstream 会有最大的 `Phi`。
- Figure 2 展示表示 `5/16` 的所有 partial error。最优 bitstream 需要在每个 partial length 上都让 error 尽可能小。
- 作者观察到：从长度 i 到 i+1，最优 partial bitstream 的 1 的数量至多增加 1，因此可以 greedily 构造。
- 对 `5/16`，一个 optimal early-terminable bitstream 是 `0100100100010010`。
- 作者称这种 bitstream 为 streaming-accurate bitstream，它在所有 partial lengths 上都达到最低可能 error，记为 `Phi_best`。
- Value-dependent streaming accuracy 定义为：
  - `Streaming accuracy = Phi / Phi_best`
  - 值为 1 表示 optimal early-terminable form。
- 因为 `Phi` 和 `Phi_best` 依赖 bitstream 表示的 value，不方便跨不同 values 聚合，作者又定义 value-independent 版本：
  - `(Phi - Phi_worst) / (Phi_best - Phi_worst)`
- `Phi_worst` 是同一 value 下最不适合 early termination 的 bitstream。
  - value < 0.5 时，最坏是所有 1 聚在开头；
  - value > 0.5 时，最坏是所有 1 聚在结尾。
- Value-independent 版本中，0 表示最差 early-terminable form，1 表示最佳。

### Evaluation
- 作者比较 streaming accuracy、progressive precision、normalized stability 选择 bitstream 的效果。
- 对 length=32 的所有可表示 values，每个 value 最多 sample 10k 个 bitstreams。
- 各 metric 选出它认为最适合 early termination 的 bitstreams；若有多个并列，则对它们 accuracy 求平均。
- accuracy 计算为 `1 - |P_X - P_{S_i}|`，再对所有 values 平均。
- Figure 3 结果：
  - streaming accuracy 选出的 bitstreams 在各 partial lengths 上都保持最高 average accuracy；
  - PP 在 powers-of-2 points 处 accuracy 高，但其他 points 下降；
  - normalized stability 在较早 termination points 上 accuracy 低，只有接近 stable cycle 后才提升。
- 有趣现象：normalized stability 的 T=10% 在 cycle 13 前反而比 T=5% accuracy 高，说明 threshold 选择会引入反直觉结果。

## IV. Characterizing Early Termination
- Streaming accuracy 可用于刻画 SC circuits。
- 作者做两类 characterization：
  - different functional units 是否 retain streaming accuracy；
  - streaming accuracy 和 bitstream independence 的关系。

### A. Streaming Accuracy Retention
- 设计 early-terminable SC circuit 时，设计者需要知道某个 component 会提高、降低还是保持 output streaming accuracy。
- 作者对多个 arithmetic functional units 做分析，bitstream length=64。
- 输入是覆盖所有 possible `(X,Y)` value combinations 的 100K random bitstream samples。
- `SA_out/SA_in` 表示 output streaming accuracy 与 input average streaming accuracy 的比值。
- 结果：
  - multiplication 和 scaled addition 的不同实现基本能 retain，甚至略微提高 streaming accuracy；
  - division implementations 会降低 streaming accuracy，因为为了准确除法，它们需要重排 bits。
- 结论：functional unit 设计不只要看 arithmetic accuracy，也可以看 streaming accuracy retention，这打开了新的 design space。

### B. Relation to Bitstream Independence
- Bitstream independence 对 SC 很重要。例如 AND multiplication 在 inputs 不独立时 accuracy 会下降。
- 因此不能只优化 streaming accuracy，也要考虑 independence。
- 作者用 ZCE 衡量 independence，ZCE 越接近 0 表示两个 bitstreams 越 independent。
- 实验：
  - 先把 X 设置为每个可表示 value 的 streaming-accurate version；
  - 对每个 X，遍历 Y 的所有 values 和所有 bitstream forms；
  - 画出 Y 的 streaming accuracy 与 ZCE(X,Y) 的关系。
- 结果：streaming accuracy 和 independence 不是 orthogonal。
  - 越 independent 的 bitstream pair，Y 的 streaming accuracy 通常越低；
  - 原因是强制两个 bitstreams 都 high streaming accuracy 会限制 0/1 alignment 的自由度，使它们更相似，从而更难 independent。
- 结论：未来 SC circuit 需要 joint optimization of streaming accuracy and bitstream independence。

## V. Streaming-Accurate Bitstream Generator
- Section III 说明：给定 bitstream length 和 value，总存在 most early-terminable 的 SA bitstream。
- SA bitstream 的形式是把 0 和 1 尽可能均匀分布。
  - `3/7` 的 SA bitstream 可以是 `0101010`；
  - `4/7` 的 SA bitstream 可以是 `1010101`。
- 对偶数 length，可能存在多个 SA bitstreams，例如 `2/8` 的 `01000010` 和 `00100010` 都可满足。
- 作者提出硬件 SA generator，总能生成 SA bitstream，适合支持 early termination 的 SC systems。
- 硬件结构本质上是带 preset 的 n-bit adder，`n=log2(L)`。
- 对长度 L、目标值 `k/L`：
  - 初始 sum 设为 `L/2`；
  - 每个 cycle 加 k；
  - 若 adder overflow，则当前 bit 输出 1，否则输出 0。
- 对非 `2^n` length，使用 `ceil(log2 L)` 位 adder，再用 comparator 检查 `sum >= L` 来模拟 arbitrary overflow；每次 overflow 后从 sum 中减 L。

### A. Area and Power Evaluation
- 作者用 Verilog RTL 实现 SA generator，并用 Synopsys Design Compiler + NanGate FreePDK45 standard-cell library 综合。
- 比较对象包括：
  - LFSR-based generator；
  - Halton base-2；
  - Halton base-3；
  - Sobol；
  - SA generator。
- length=64、500MHz 下：
  - SA generator area 接近 LFSR 和 base-2 Halton，属于低 area 组；
  - base-3 Halton 面积明显更高，因为 binary-coded base-b counter 的开销随 b 增大。

### B. Accuracy Evaluation
#### Functional Unit Evaluation
- 作者在 functional unit level 测试 uMUL-ST multiplier。
- uMUL-ST 内部 generator 分别设为 LFSR、Halton、Sobol、SA。
- 对 10k random `(X,Y)` input value pairs，统计不同 early termination points 下的 average accuracy。
- 结果：
  - LFSR generator accuracy 最低；
  - SA generator accuracy 最高；
  - SA 不仅更快收敛到高 accuracy，即使不 early terminate，final accuracy 平均也更高。

#### Application Evaluation
- Application level 使用 2D convolution SC system。
- 输入为 6 张 256x256 grayscale images，卷积核是 3x3 Gaussian blur。
- 所有 multiplications 用 uMUL-ST，9 个 products 用 parallel adder 累加。
- 相对 floating-point baseline 计算 RMSE，并对 6 张图取 geomean。
- 结果：
  - SA generator 在所有 early termination points 下 RMSE 最低；
  - LFSR generator RMSE 最高。
- 这说明 SA generator 对实际 SC system 的 efficient early termination 有直接收益。

## VI. Conclusion
- 本文提出 streaming accuracy，用于衡量 bitstream 与 streaming-accurate form 的接近程度。
- 相比 progressive precision 和 normalized stability，它能在 arbitrary termination points 上更一般地评价 early termination。
- 作者还设计了保证生成 SA bitstreams 的硬件 generator。
- 实验显示，SA generator 相比 LFSR、Halton、Sobol 等 generator，在 functional unit 和 application level early termination 下都有更低 error。
- 对 SC 这种 area / accuracy / latency tradeoff 很强的计算范式，streaming accuracy 提供了更系统的 early termination 分析与优化工具。
