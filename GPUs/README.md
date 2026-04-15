
# Efficient GPU Synchronization without Scopes: Saying No to Complex Consistency Models

## ABSTRACT

这一部分先把整篇论文的核心观点一次性讲清楚。

第1段主要讲研究背景：GPU 现在越来越通用，不再只跑那种规则、几乎不共享数据的程序，而开始出现更复杂的共享模式和细粒度同步。问题是，传统 GPU coherence protocol 很简单，但同步操作非常重。以前的一些工作想通过 scoped synchronization 来解决这个问题，也就是给同步加“作用域”，让同步只在某一层缓存范围内生效。这样确实能提高效率，但代价是内存一致性模型会变复杂，从常见的 DRF 变成更复杂的 HRF。

第2段给出论文主张：作者不是去接受这种更复杂的模型，而是尝试把 DeNovo coherence protocol 用到 GPU 上，并比较：

1. conventional GPU coherence + DRF
2. conventional GPU coherence + HRF
3. DeNovo + DRF
4. DeNovo + HRF

作者想说明一个关键观点：HRF 的复杂性并不是获得高性能的必要条件，也不是充分条件。也就是说：

* 你把模型搞复杂了，不一定就最好
* 不搞复杂，也可能做得很好

第3段给出结论总览：DeNovo + DRF 在性能、能耗、硬件开销、模型复杂度之间是一个很好的平衡点。

第4段给出具体结果：

* 对于 global scope 的细粒度同步 benchmark，相比 GPU+HRF，DeNovo+DRF 平均执行时间低 28%，能耗低 51%
* 对于 mostly local scope 的细粒度同步 benchmark，GPU+HRF 略好一点
* 但这种略好的收益，要靠更复杂的 HRF 模型来换
* 而且只要给 DeNovo+DRF 加一个 modest enhancement，这个优势就基本没了
* 如果你真的接受 HRF 的复杂度，那么 DeNovo+HRF 是最强配置

摘要其实已经把论文主线全说完了：

1. scoped synchronization 能提高性能，但会让模型复杂
2. DeNovo 通过 ownership 达到类似甚至更好的效果
3. 所以不一定要用 scopes 才能高效同步

# 1. INTRODUCTION

## 第1段：GPU 正在变得更通用

作者先说 GPU 最初是为 data-parallel execution 优化的，但现在越来越 general purpose。工业界也在推进 CPU 和 GPU 共享统一地址空间，这样 CPU 和 GPU 可以访问同一份数据，而不必显式 copy。

这一段的重点是：统一地址空间虽然让编程更方便，但也带来了 coherence、synchronization、memory consistency 这些问题。以前 CPU 和 GPU 是分开的，这些问题没那么尖锐；现在它们共享内存，就必须认真处理。

## 第2段：传统 GPU coherence 的设计前提

这一段解释为什么早期 GPU coherence 很简单。

因为以前 GPU 跑的主要是：

* 数据并行
* 流式
* Compute Units 之间几乎不共享
* 数据复用不强
* 同步大多是粗粒度，比如 kernel boundary

所以 GPU 使用的软件驱动 coherence 很粗暴：

* acquire 时 invalidate 整个 cache
* release 前把所有 dirty data flush / writethrough 到更低层共享缓存
* 细粒度同步一般很少，所以直接在更下层共享缓存做，绕过 private cache

这一段很重要，因为它解释了传统 GPU 协议为什么“简单但笨重”：

* 简单：没有 CPU 那种复杂的 invalidation、ownership、directory
* 笨重：每次同步代价很大

## 第3段：传统 GPU 协议和 CPU 协议的差别

作者强调，和 CPU 常见的 multicore coherence 比，传统 GPU coherence 非常简单，不需要：

* writer-initiated invalidations
* ownership requests
* downgrade requests
* protocol state bits
* directories

而且虽然 GPU memory consistency model 以前定义得不太清晰，但这种实现大体上还是 compatible with DRF 的。

这一段的作用是做对比：传统 GPU 的优点是简单。

## 第4段：问题出现了

随着 GPGPU 应用扩展，越来越多应用开始有：

* 更一般的 sharing pattern
* fine-grained synchronization

此时，传统 GPU coherence 的几个重成本就暴露了：

* full cache invalidates
* dirty data flushes
* remote execution of synchronization

也就是，原来同步少，还能忍；现在同步细了、频繁了，成本就爆炸。

## 第5段：已有解决方案是 scopes

这一段说已有工作提出：给同步访问加 scope，表明这个同步只涉及哪一层内存层级。

比如 local scope 表示：

* 只同步同一个 CU 内 thread blocks 访问的数据
* 因为这些 block 共享 L1，所以同步可以在 L1 做
* 不需要 invalidate 或 flush 到更低层

这段本质上在说：scopes 的核心价值是“缩小同步生效范围”，从而减少同步成本。

## 第6段：scopes 的代价是 memory model 复杂化

虽然 scopes 很高效，但编程复杂度更高。

作者这里讲得很关键：一旦引入 locally scoped synchronization，普通的 DRF 模型就不够了。因为这会产生 synchronization races，从而让程序即使“看起来同步了”，也可能出现违反 sequential consistency 的非直观行为。

为了解决这个问题，之前工作提出了 HRF。HSA 和 OpenCL 2.0 也采用了类似思路。

这段要记住一句中心思想：
scopes 不是白来的，它不是单纯的“优化技巧”，而是会改变程序员必须理解的 memory model。

## 第7段：作者的立场

作者说，虽然 HRF 定义得很严谨，但它掩盖不了一个事实：scope 本质上是硬件启发的机制，它把 memory hierarchy 暴露给程序员了。

作者的态度很明确：

* 过去大家已经因为硬件导向的 memory model 吃过很多苦
* DRF 虽然已经不算特别简单了，但好歹是软件导向的
* GPU consistency model 不应该比 CPU 的还复杂

这段其实是论文的“价值立场”：
不是单纯追求性能，而是追求“性能和编程模型复杂度的平衡”。

## 第8段：论文提出的问题

作者明确提出本文核心问题：

能不能设计一种 GPU coherence protocol：

* 保持接近传统 GPU protocol 的简单性
* 获得 scoped synchronization 的性能收益
* 同时保持 memory model 不比 DRF 更复杂

这句就是整篇论文的 research question。

## 第9段：答案是 DeNovo

作者说可以，办法是考虑 DeNovo 这个最近提出的 hardware-software hybrid protocol。

DeNovo 的特点：

* 不需要 writer-initiated invalidations
* 不需要 directory
* 但写入时会获取 ownership
* 相比 GPU+HRF，额外开销主要是 cache 里每 word 1 bit
* 以前 DeNovo 还用 software regions 做 selective invalidation，但本文 baseline 不用 region，尽量压低开销

这段说明作者选 DeNovo 的原因：
它不像 MESI 那么重，但又比传统 GPU 协议更聪明。

## 第10段：贡献1

作者说第一个贡献是：证明 DeNovo（不带 regions）可以作为 GPU coherence protocol。

核心原因是：

* 它在 write 时获取 ownership
* 所以可以跨同步边界复用写过的数据和同步变量
* 不必借助 scopes 的复杂性

## 第11段：贡献2

第二个贡献是比较多种组合：

* DeNovo + DRF
* GPU + DRF
* GPU + HRF

结果表明：

* GPU+DRF 在 fine-grained synchronization 下表现差，这个不意外
* DeNovo+DRF 在性能、能耗、复杂度、实现开销之间是 sweet spot

作者随后细分 benchmark 类别：

1. 无 fine-grained synchronization：DeNovo ≈ GPU
2. global-scope fine-grained synchronization：DeNovo 显著优于 GPU+HRF
3. mostly local-scope synchronization：GPU+HRF 略优于 DeNovo

但作者强调，这个 local-scope 的小优势必须用更复杂的模型换。

## 第12段：贡献3

第三个贡献是改进版 DeNovo+DRF：

* 加 selective invalidation
* 避免在 acquire 时 invalidating 有效的 read-only 区域

这个优化只需要软件传递 read-only region 信息，不增加额外状态位。

结果是：增强后的 DeNovo+DRF 平均能达到 GPU+HRF 一样的性能和能耗。

## 第13段：贡献4

第四个贡献是：如果你能接受 HRF 的复杂度，那也可以把 HRF 和 DeNovo 结合。

结果：

* DeNovo+HRF 是总体最强的
* 在所有情况里都不差于 GPU+HRF
* 在 global synchronization 下明显更好

## 第14段：为什么不跟 MESI 比

作者专门解释：虽然 MESI 这种传统硬件协议也能在 DRF 下支持细粒度同步，但不拿它当重点比较对象，因为：

* 复杂度高
* 需要 writer-initiated invalidations
* 有 directory 开销
* 有很多 transient states
* 之前研究已表明它不适合传统 GPU 应用

此外 DeNovo 在 CPU 上已被证明性能与 MESI 相当甚至更好，但复杂度更低。

## 第15段：引言总结

最后一句是引言的总结：
这是第一篇展示 GPU 可以在 modest hardware overhead 下，用 DRF 而不是 HRF，也高效支持 fine-grained synchronization 的工作。

一句话总结 introduction：
作者反对“GPU 必须用更复杂的一致性模型才能高效同步”这个趋势，提出 DeNovo+DRF 作为更好的平衡点。

# 2. BACKGROUND: MEMORY CONSISTENCY MODELS

这一节是理论基础，解释 DRF 和 HRF。

## 第1段：本文研究用到两种模型

作者说，根据 coherence protocol 是否用 scoped synchronization，本文采用：

* DRF
* HRF

也就是说：

* 没有 scopes → DRF
* 有 scopes → HRF

## 第2段：DRF 的定义

DRF 保障的是：
对于 data-race-free 的程序，系统提供 sequential consistency。

作者解释 data-race-free 的定义：

* 程序中的内存访问分成 data access 和 synchronization access
* 对所有 SC 执行，如果任意一对冲突的数据访问，都被 happens-before 排序约束住，那么程序就是 data-race-free

然后解释 happens-before：
它是 program order 和 synchronization order 的传递闭包。
其中 synchronization order 指：

* 一个 synchronization write（release）
* 发生在一个 synchronization read（acquire）之前
* 那么它们之间建立顺序

这一段要记的核心：
DRF 的本质是“所有冲突数据访问都必须被同步顺序约束住”。

## 第3段：HRF 的定义

HRF 和 DRF 类似，但多了一个 scope attribute。

关键变化是：

* 每个 synchronization access 都有 scope
* HRF 的 synchronization order 只排序“scope 相同”的同步访问

然后作者说 HRF 有两种：

1. HRF-Direct：要求所有参与同步的线程使用相同 scope
2. HRF-Indirect：在 HRF-Direct 基础上，额外支持不同 scope 之间的传递同步

这一段要记住：
HRF 的难点就在于 scope 会影响同步顺序是否成立。

## 第4段：synchronization races

这里提出一个非常关键的概念：synchronization races。

意思是：
如果两个冲突的 synchronization accesses 作用域不同，HRF 不会自动给它们排序。
这种未被排序的同步冲突就叫 synchronization race。

而这种 race：

* 模型不允许你把它当作合法同步来推理
* 也不能靠它给 data access 建立顺序

这正是 scopes 让模型变复杂的根源：
不是所有“你以为在同步的同步”，都真的能同步。

## 第5段：程序顺序约束

最后一段说，无论 DRF 还是 HRF，实现时通常都要满足 program order requirement：

如果 X 在程序顺序上早于 Y，且满足以下任意条件之一，则 X 必须先 complete，Y 才能进行：

1. X 是 acquire，Y 是 data access
2. X 是 data access，Y 是 release
3. X 和 Y 都是 synchronization

然后作者说，对于 cache system，access 什么时候算 complete，要由底层 coherence protocol 来定义。下一节就讲这个。

这一节的核心任务是让你理解：

* DRF：简单，scope 不参与
* HRF：同步顺序取决于 scope，复杂
* coherence protocol 不只是保证数据对，还要配合 consistency model 定义“完成”的含义

# 3. A CLASSIFICATION OF COHERENCE PROTOCOLS

这一节是全文最重要的概念框架之一。作者不是直接讲某个协议，而是先搭一个分类框架。

## 第1段：coherence protocol 的终极目标

作者先说 coherence protocol 的终极目标是：
确保 read 从 cache 中读到正确值。

对于 DRF / HRF，正确值就是：
按 happens-before 排序后，最后一个冲突 write 写入的值。

## 第2段：把 coherence 拆成两个任务

作者说根据 DeNovo 之前的观察，可以把 coherence protocol 的任务拆成两件事：

1. No stale data
   load hit in private cache 不能读到 stale data

2. Locatable up-to-date data
   如果 private cache miss，系统必须知道去哪里找到最新数据

这个拆法特别重要，因为后面所有协议的比较都围绕这两件事展开。

## 第3段：三类协议的分类

表1把协议分成三类：

* Conventional hardware protocol
* Software protocol
* Hybrid protocol

分类标准是：

1. invalidation tracking 谁来做
2. 谁负责维护最新副本位置
3. 能否支持不同 scopes

作者给出的例子是：

* conventional HW：writer 负责 invalidation 和 ownership
* software：reader 负责 invalidation，writer 通过 writethrough 维持新值
* hybrid：reader 负责 invalidation，但也有 ownership

这一段其实就是在说：
GPU、MESI、DeNovo 三者最大的不同，不是“有没有 coherence”，而是“谁负责防 stale、谁负责找最新值”。

## 第4段：这个 taxonomy 的意义

作者承认这个分类不一定穷尽所有协议，但它覆盖了：

* CPU 常见协议
* GPU 常见协议
* 新的软硬件混合协议

并且之后假设一个两级 cache hierarchy：

* private L1
* shared last-level L2

在 GPU 中，一个 CU 内的 thread blocks 共享该 CU 的 L1。

这个假设是全文性能分析和协议描述的基础。

## Conventional Hardware Protocols used in CPUs

### 第1段：CPU 常用的 MESI 类协议

CPU 一般用纯硬件 coherence，比如 MESI，依赖：

* writer-initiated invalidation
* ownership tracking

通常还要 directory 记录：

* sharers list
* 或当前 owner

### 第2段：写入流程

如果某 core 想写一个自己不拥有的 line：

* 向 directory 请求 ownership
* directory 给 sharers 或前 owner 发 invalidation
* 最后 writer 拿到写权限

这里数据访问和同步访问一般被统一对待。

### 第3段：program order requirement 如何满足

在这种协议里：

* 一个 write 何时 complete？当它的 invalidations 到达所有 sharers / previous owner
* 一个 read 何时 complete？当它返回值，且这个值 globally visible

### 第4段：作者对这类协议的态度

作者说这类协议理论上也可以和 HRF 结合，用 scopes。
但：

* 加 scopes 后是否还有明显收益，不清楚
* 而且这类协议本来就不适合 GPU
* 所以这里只是为了完整性提一下

这一小节的核心是：
CPU 协议能做很多事，但代价太大，不是 GPU 的理想方向。

## Software Protocols used in GPUs

### 第1段：GPU 的基本机制

GPU 用简单、主要软件驱动的 coherence，不做：

* writer-initiated invalidations
* ownership tracking

先看没有 scopes 的版本。

### 第2段：GPU 如何避免 stale data

GPU 使用 reader-initiated invalidation：

* acquire 时 invalidate 整个 cache
* 这样以后再读就不会命中 stale value

GPU 又通过 writethrough 到共享 L2 保证最新值能找到：

* write 被缓冲在 store buffer 里
* 到 release 或 buffer 满时，再写到 L2
* 所以 read miss 总能从 L2 找到 up-to-date copy

### 第3段：同步操作为什么必须去 L2

因为 GPU 没有 writer-initiated invalidation，也没有 ownership tracking，所以 synchronization access 必须在所有参与者都共享的层级做，也就是 L2。

因此：

* release 要等前面的 writethrough 到达 L2 才算 complete
* synchronization access 在 L2 执行完才算 complete

### 第4段：这种协议的优缺点

优点：

* 简单
* 不需要复杂状态位
* 没有 invalidation / ack 协议流量

缺点：

* synchronization 很贵
* acquire 要 invalidate 整个 cache
* release 要等所有写刷到 L2
* 同步操作本身要去较远的 L2

### 第5段：加 scopes 后怎么改

两级层次下有两个 scope：

* local = L1
* global = L2

如果是 local scope synchronization：

* acquire 不用 invalidate L1
* release 不用等 writethrough 到 L2
* 同步直接在 L1 做

global scope 则和原来差不多。

### 第6段：代价

虽然 scopes 能降开销，但它把 memory hierarchy 暴露给程序员了。

这一小节你可以记成：
传统 GPU 协议的高性能来自“简单”，而 local scopes 的高性能来自“少做事”，但代价是程序员要懂作用域。

## DeNovo: A Hybrid Hardware-Software Protocol

这是本节最关键的一部分。

### 第1段：DeNovo 的基本思想

DeNovo 是 hybrid protocol：

* 用 reader-initiated invalidations
* 但用 hardware-tracked ownership

因为没有 writer-initiated invalidations，所以不需要 directory 去跟踪 sharers list。

### 第2段：ownership 怎么存

DeNovo 用 shared L2 的 data banks 来记录 ownership：

* 要么 L2 自己有最新数据，表示没有 L1 拥有
* 要么 L2 记录当前哪个 core 拥有它

DeNovo 把 L2 叫 registry，把获取 ownership 叫 registration。

这段一定要记住 registry / registration 这两个词。

### 第3段：状态

DeNovo 有三个状态：

* Registered
* Valid
* Invalid

对应 MSI 的：

* Modified
* Shared
* Invalid

但和 MSI 不同的是：

* DeNovo 就这三个状态
* 没有 transient states
* 因为它利用了 data-race-freedom，不做 writer-initiated invalidations

同时，DeNovo 的 coherence state 是 word granularity，不是 line granularity。

这点非常关键：
DeNovo 把 coherence 跟踪粒度做得更细，所以能更精准地处理同步和数据复用。

### 第4段：DeNovo 的 invalidation 行为

和 GPU 协议一样，DeNovo 也会在 acquire 时 invalidation cache。
但不同的是，它可以 selective invalidation。

baseline DeNovo 利用了一个性质：

* registered state 的数据一定是 up-to-date
* 所以 acquire 时不必 invalidating 它

之前的 DeNovo 工作还考虑过 software regions、touched bits 等更高级优化。
本文只加一个简单优化：

* read-only region 不 invalidate

而且作者强调，read-only region 是程序级属性，是 hardware-oblivious 的；
相比之下 scope annotation 是 hardware- and schedule-specific，更难。

这段是全文一个核心论据：
如果你只是想少 invalidation，不一定非得用 scopes，也可以靠 ownership + 简单 region 信息做到。

### 第5段：同步变量怎么处理

对 synchronization accesses，作者用的是 DeNovoSync0。

其特点是：

* 同步读和同步写都要 register
* 只要该位置在 L1 里不是 registered state，就按 miss 处理，必须发 registration request

这样带来的好处是：
同步变量如果有 temporal locality，就可能 hit in L1；
而传统 GPU 协议所有同步都得去 L2，没有这种 hit。

### 第6段：racy synchronization 的处理

这一段比较细，但考试可能会考。

DeNovoSync0 对有竞争的同步 registration 是这样处理的：

* registry 按到达顺序立刻处理
* 如果某个字已经被别的 L1 registered，新请求会被 forward 到那个 L1
* 如果 forward 请求比原 owner 的 registration acknowledgment 到得还早，就先排在 owner 的 MSHR 里
* 在高竞争场景下，不同 core 的同步请求会形成 distributed queue

还有一个 locality 优化：

* 同一个 CU 内多个同步请求会在 MSHR 里 coalesce
* 并且本地请求优先于远端排队请求

作者解释这种 distributed queue 的效果：

* 如果很多竞争同步都会失败，比如抢锁失败，那么这种串行化 / throttling 是好事
* 但如果很多同步实际上会成功，可能增加关键路径延迟
* 不过作者说这种情况通常少见

这段的本质是：DeNovo 的同步不只是“能在 L1 做”，还通过 registration 队列来控制 contention 行为。

### 第7段：DeNovoSync 的进一步优化

DeNovoSync 还可以在 read-read contention 太多时加 backoff，但本文为了简单不研究。

### 第8段：program order requirement 在 DeNovo 下怎么定义

DeNovo 认为：

* data write complete：当它完成 registration
* synchronization read/write complete：当它完成 registration
* data read complete：当它返回值

这个定义与 GPU 协议不同，是因为 ownership 改变了“什么时候算真的完成”。

### 第9段：DeNovo 如何支持 scopes

DeNovo 之前没评估过 scoped synchronization，但可以自然扩展：

* local acquire / release 不 invalidate cache
* 不 flush store buffer
* local synchronization operation 可以延迟获得 ownership

这一句说明：
DeNovo 并不是不能跟 HRF 结合，它也能用 scopes，只是作者的重点是说明“不用 scopes 也能做得很好”。

# 4. QUALITATIVE ANALYSIS OF THE PROTOCOLS

这一节不做定量实验，而是先讲“道理上谁强在哪里”。

## 4.1 Qualitative Performance Analysis

### 第1段：比较对象

作者要比较四个协议：

* GPU coherence + DRF，记作 GPU-D
* GPU coherence + HRF，记作 GPU-H
* DeNovo + DRF，记作 DeNovo-D
* DeNovo + HRF，记作 DeNovo-H

并用表2从几个关键能力上比较它们。

### 第2段：比较维度

表2比较的特性包括：

* reuse written data across synchronization points
* reuse valid data across synchronization points
* avoid bursty traffic
* no invalidations / acknowledgments
* decoupled coherence and transfer granularity
* reuse synchronization variables
* efficient support for dynamic sharing

这些维度其实就是作者对 emerging GPU workloads 关心的能力总结。

## GPU coherence, DRF consistency (GPU-D)

### 第1段：优点

GPU-D 不需要 invalidation message 和 acknowledgment message，因为它：

* 在每个同步点 self-invalidate 全部 valid data
* 把 dirty data 全部 writethrough 到 backing LLC

所以简单。

### 第2段：缺点1，不支持跨同步点复用

因为不获取 ownership：

* synchronization 必须在 LLC 做
* release 时必须 flush dirty data
* acquire 时必须 invalidate 全 cache

所以 GPU-D 无法在同步点之间复用数据。

### 第3段：缺点2，bursty traffic

release 时或 kernel boundary 时，store buffer 被集中 flush，会造成 bursty writethrough traffic。

### 第4段：缺点3，coarse granularity

GPU 协议按较粗粒度传输数据以利用 spatial locality。
但对于细粒度同步或 strided access，这可能不是最优。

### 第5段：缺点4，dynamic sharing 支持差

对于 work stealing 这种 dynamic sharing，必须在 LLC 做同步，否则可能读到 stale data。

一句话总结 GPU-D：
简单，但同步特别重，不能复用，适合老式 GPU workload，不适合新 workload。

## GPU coherence, HRF consistency (GPU-H)

### 第1段：HRF 带来的改进

从 DRF 改成 HRF 后，GPU 协议的很多低效能去掉，同时还保持了没有 invalidation/ack message 的优点。

### 第2段：local scope 的好处

locally scoped synchronization：

* 在 L1 做
* 不需要 bursty writebacks
* 不需要 self-invalidation
* 不需要 flush

因此：

* fine-grained synchronization 更高效
* data 可以跨 local synchronization point 复用

### 第3段：局限

但 scopes 不能很好支持 dynamic sharing，因为程序员必须保守地用 global scope，避免 stale data。

这段非常重要：
GPU-H 的优化收益，主要来自“local scope 可以省很多动作”；
可一旦 sharing pattern 动态化，就不得不退回 global scope。

## DeNovo coherence, DRF consistency (DeNovo-D)

### 第1段：相对 GPU-D 的优势

DeNovo-D 利用 ownership，不靠 scopes，也拿到了 GPU-H 的不少好处。

### 第2段：written data reuse

因为 acquire 时 registered data 不会被 invalidate，所以写过的数据能跨同步边界复用。

如果再加 read-only optimization，那么 read-only data 也能复用。

### 第3段：synchronization variable reuse

同步变量如果有 temporal locality，可以在同一 thread block 内 hit，也可以在同一 CU 的不同 thread block 间 hit。

### 第4段：避免 bursty traffic

因为写数据是获取 ownership，而不是 release 时统一 writethrough，所以不会出现 GPU-D 那种 bursty writebacks。

### 第5段：dynamic sharing 好

与 GPU-H 不同，ownership 机制天然支持 dynamic sharing，比如 work stealing。

### 第6段：更细的 granularity

DeNovo 把 coherence granularity 和 transfer granularity 分离，因此传的更可能是“有用数据”，而不是整块粗粒度数据。

### 第7段：代价

DeNovo 也不是没缺点：

* miss latency 可能更高
* 如果数据在 remote L1，可能多一跳
* 高 contention 同步可能因为 registration queue 被串行化

但作者说实验中收益占主导。

这小节是全文中最重要的 qualitative 结论之一：
DeNovo-D 的核心价值是“不用 scopes，也能靠 ownership 隐式获得很多 locality 和 reuse”。

## DeNovo coherence, HRF consistency (DeNovo-H)

这一段很短，意思是：
DeNovo-H 同时拥有：

* DeNovo-D 的 ownership 优势
* GPU-H 的 local scope 优势

也就是所有好处都叠加。

## 4.2 Protocol Implementation Overheads

这一小节分析硬件实现开销。

## GPU-D

L1 和 L2 只需要每 line 1 个 valid bit。
另外需要：

* flash invalidate entire cache on acquire
* store buffer buffering until release

所以 GPU-D 最省状态位。

## GPU-H

相比 GPU-D，GPU-H 额外需要：

* L1 每个 word 1 bit，记录 partial cache block writes
* L1 overhead 大约比 GPU-D 多 3%
* L2 仍然只要每 line 1 valid bit
* 还要支持 global acquire / release 时 flash invalidate

所以 GPU-H 比 GPU-D 稍贵。

## DeNovo-D / DeNovo-H

因为 DeNovo 做的是 word-granularity coherence，所以需要 per-word state bits。

L1：

* 3 个 coherence states，需要 2 bits / word
* 相比 GPU-H，L1 overhead 再多约 3%

L2：

* 每 line 1 valid + 1 dirty
* 每 word 1 bit
* 相比 GPU-H，L2 overhead 多约 3%

这说明 DeNovo 不是“零代价”，但作者想强调这个代价是 modest。

## DeNovo-D + RO

逻辑上 read-only optimization 需要每 word 再多 1 bit。
但作者复用了 DeNovo coherence bits 中未用完的状态，所以不额外增加硬件位数。

额外代价主要是：
软件要把 read-only region 信息传给硬件。

这一节要记住：
DeNovo 的代价主要是 per-word state；RO 优化主要增加软件信息传递，而不是硬件状态开销。

# 5. METHODOLOGY

这一节讲实验怎么做。

## 第1段：整体思路

作者说这项工作受之前 DeNovo 系列工作影响，沿用了已有基础设施，并扩展它来支持 GPU synchronization operations，基于 DeNovoSync0 协议。

也就是说，这不是凭空搭的新 simulator，而是在已有 DeNovo infrastructure 上扩展。

## 5.1 Baseline Heterogeneous Architecture

### 第1段：系统结构

作者模拟的是 tightly coupled CPU-GPU architecture：

* unified shared memory address space
* coherent caches
* CPU cores 和 GPU CUs 通过 interconnect 连接

每个 network node 上有：

* 一个 CPU core 或一个 GPU CU
* 一个 L1 cache
* 一个 shared L2 cache bank

GPU node 还带 scratchpad。

### 第2段：Figure 1 的含义

图1展示这个 baseline system。
你可以把它理解成：

* 每个 CPU core / GPU CU 都有自己的私有 L1
* 所有节点共享分布式 L2 banks
* 通过片上网络互连

这一结构接近 prior work 的系统模型。

### 第3段：哪些东西会变化

作者说不同 configuration 下会变的是：

* coherence protocol
* consistency model
* write policy

体系结构骨架不变。

## 5.2 Simulation Environment and Parameters

### 第1段：仿真平台

模拟器由三部分组成：

* Simics：建模 CPU
* Wisconsin GEMS：memory timing simulator
* GPGPU-Sim v3.2.1：建模 GPU

GPU 类似 NVIDIA GTX 480。
互连网络用 Garnet 模拟 4x4 mesh。

### 第2段：软件和参数

GPU kernels 使用 CUDA 3.1，因为这是 GPGPU-Sim 完整支持的最新版。

表3给出了关键参数：

* CPU：2 GHz，1 core
* GPU：700 MHz，15 CUs
* L1：32 KB，8 banks，8-way
* L2：4 MB，16 banks，NUCA
* store buffer：256 entries
* L1 hit latency：1 cycle
* remote L1：35-83 cycles
* L2 hit：29-61 cycles
* memory：197-261 cycles

这些参数要知道大概，不一定逐个背，但最好知道：
实验平台是 1 CPU core + 15 GPU CUs 的 heterogeneous system。

### 第3段：能耗模型

GPU CU 的能耗用 GPUWattch。
NoC 能耗用 McPAT。
CPU core 和 CPU L1 energy 不建模，因为 CPU 只做功能模拟，不是研究重点。

### 第4段：annotation API

作者提供了 API 支持手工插入 annotation：

* region info 给 DeNovo-D+RO
* synchronization instruction 和其 scope 给 HRF

这说明 HRF 和 RO 都需要程序员/软件层额外提供信息。

## 5.3 Configurations

这一小节列出所有实验配置。

### 第1段：共同假设

所有 configuration：

* CPU 总是使用 DeNovo coherence
* 都有 256-entry coalescing store buffer
* 假设支持在 L1 和 L2 用 atomics 做 synchronization
* 不研究 relaxed atomics，因为语义还在争议中

### GPU-D (GD)

基线 DRF memory model，无 scopes。
GPU coherence。
所有 synchronization accesses 都在 L2 做。

### GPU-H (GH)

GPU coherence + HRF-Indirect。
local scope synchronization 在 L1 做。
global scope synchronization 在 L2 做。

### DeNovo-D (DD)

DeNovoSync0 coherence，不带 regions。
DRF memory model。
所有 synchronization accesses 都在 L1 做，但前提是先 registration。

### DeNovo-D with read-only optimization (DD+RO)

在 DD 上增加 selective invalidation，acquire 时不 invalidating 有效 read-only data。

### DeNovo-H (DH)

DeNovo-D + HRF-Indirect。
像 GH 一样，local scope synchronization 总在 L1 做，不需要 invalidation 或 flush。

这部分最值得你背的就是五个简称：

* GD
* GH
* DD
* DD+RO
* DH

## 5.4 Benchmarks

### 第1段：为什么 benchmark 难选

作者说评估很难，因为真正用 fine-grained synchronization 的 GPU application benchmark 很少。

所以他们结合：

1. 没有 intra-kernel synchronization 的 application benchmarks
2. 需要 global scope synchronization 的 microbenchmarks
3. mostly local scope synchronization 的 benchmark

所有代码都用 15 个 GPU CUs 和 1 个 CPU core。

## 5.4.1 Applications without Intra-Kernel Synchronization

### 第1段：选了哪些

作者用了 10 个来自 Rodinia 和 Parboil 的应用：

* BP
* PF
* LUD
* NW
* SGEMM
* ST
* HS
* NN
* SRAD
* LavaMD

### 第2段：这些应用的意义

这些应用都：

* 不在 kernel 内做 synchronization
* 也没有专门写成跨 kernel 复用数据

所以它们主要是用来验证：
DeNovo 至少不会把今天主流 workload 弄得更差。

## 5.4.2 (Micro)Benchmarks with Intra-Kernel Synchronization

### 第1段：为什么用 microbenchmarks

因为真实 GPU 应用里 fine-grained synchronization 很少，所以作者用 Stuart and Owens 的同步原语 microbenchmarks：

* mutex locks
* semaphores
* barriers

还用了 UTS，这是 HRF 论文里唯一一个用了 fine-grained synchronization 的 benchmark。

### 第2段：这些 benchmark 的多样性

这些 microbenchmarks 覆盖了：

* centralized 和 decentralized algorithm
* 不同 stall cycles
* 不同 scalability
* 每线程不同工作量

目的就是尽量覆盖 synchronization use case。

### 第3段：作者对原 benchmark 的修改

这段很重要，因为作者不是直接拿来跑，而是改造过。

mutex benchmark 被改成两个版本：

* local synchronization 版本：每个 CU 访问独有数据
* global synchronization 版本：所有 thread blocks 访问同一批数据

barrier benchmark 被改成 tree barrier：

* 每个 CU 内 thread blocks 先 local barrier
* 然后每个 CU 派一个 thread block 参加 global barrier
* global barrier 后再交换数据

还加了一个版本：
每个 CU 在 join global barrier 前先 local exchange data。

### 第4段：对 semaphore 的修改

semaphore 改成 reader-writer 形式的 local synchronization：

* 每个 CU 有 1 writer TB
* 2 reader TB
* 每个 reader 读一半数据
* writer 把 reader 数据右移
* 为了防 stale data，writer 要获得整个 semaphore

### 第5段：working set

除 TBEX_LG 和 TB_LG 外，其余 microbenchmark 的 working set 都能放进 L1。
TBEX_LG 和 TB_LG 因为反复跨 CU 交换数据，working set 更大。

这一节要记住的核心：
作者特意构造了三类 workload，用来区分 no sync / global sync / local sync 三种场景。

# 6. RESULTS

这一节是实验结果。

Figures 2、3、4 分别对应：

* 图2：没有 fine-grained synchronization 的应用
* 图3：只用 global scope fine-grained synchronization 的 microbenchmark
* 图4：mostly local 或 hybrid synchronization 的 benchmark

每张图都有三部分：

* (a) execution time
* (b) dynamic energy
* (c) network traffic

网络流量又分：

* data reads
* data registrations
* writebacks / writethroughs
* atomics

作者还说明：

* 图2和图3只画 GPU-D 与 DeNovo-D，因为这两类场景 HRF 不起作用
* 用 G* 表示 GPU-D 与 GPU-H 一样
* 用 D* 表示 DeNovo-D 与 DeNovo-H 一样
* 图4 才画全部五种配置

## 第1段：总结果总结

相对于 best GPU coherence protocol（也就是 GPU-H）：

* 在 no fine-grained synchronization 应用上，DeNovo-D 可比
* 在只 global synchronization benchmark 上，DeNovo-D 更好，平均执行时间低 28%，能耗低 51%
* 在 mostly local synchronization benchmark 上，GPU-H 更好，平均时间低 6%，能耗低 4%

但作者马上强调：

* 这点优势靠更复杂的 HRF 换来
* 给 DeNovo-D 加 read-only optimization 后，这优势基本消失
* 若 DeNovo 也用 HRF，则 DeNovo-H 是最优

这一段就是整个结果部分的总括。

## 6.1 GPU-D vs. GPU-H

### 第1段：GPU-H 显著优于 GPU-D

图4表明，只要能用 locally scoped synchronization，GPU-H 相比 GPU-D 有大改进：

* execution time 平均下降 46%
* energy 平均下降 42%

### 第2段：原因1，同步延迟更低

locally scoped acquire 在 L1 执行，而不是去 L2，所以 atomic traffic 平均下降 94%。

### 第3段：原因2，不 invalidate / 不 flush

local acquire 不 invalidate cache，local release 不 flush store buffer。
于是数据可以跨 local synchronization boundary 复用。

结果就是：

* L1 hit 更多
* execution time 更低
* energy 更低
* network traffic 更低

### 第4段：量化改善

平均来看，GPU-H 的：

* L1 / L2 / network energy 下降 71%
* 非 atomic 网络流量下降 78%

这一小节的结论：
GPU-H 的强项非常明确，就是“如果大部分同步真的可以 local scope 化，那收益很大”。

## 6.2 DeNovo-D vs. GPU Coherence

### 6.2.1 Traditional GPU Applications

#### 第1段：总体差不多

对 10 个没有 fine-grained synchronization 的应用，图2显示：

* DeNovo* 执行时间和能耗平均只增加 0.5%
* 网络流量平均减少 5%

也就是说，DeNovo 并不会伤害今天典型 workload。

#### 第2段：LavaMD 的特殊收益

LavaMD 上 DeNovo* 明显减少网络流量。
原因是 LavaMD 会让 GPU* 的 store buffer 溢出，导致对同一地址的多次写无法 coalesce，只能分别 writethrough 到 L2。

而 DeNovo* 一旦拿到 word ownership，后续写都 hit，不再占用 store buffer。

这一段体现了 ownership 的一个额外好处：
可以缓解 store buffer 压力。

#### 第3段：为什么有时 DeNovo 会稍差

作者也诚实讲了两种 DeNovo 可能多花成本的情况：

1. 如果一个 word 在最后一次写前被 eviction，那么多次写可能会触发多次 ownership request
   而 GPU* 可能在 store buffer 里把这些写合成一次 writethrough

2. 如果一个 word 被 remote core registered，DeNovo* 的 read 或 registration miss 可能要多一跳
   而 GPU* 总是去 L2，路径更固定

#### 第4段：但这些额外成本不大

作者说在这些应用里，上述 overhead 很小，不影响性能。

而且第二种 remote L1 miss，如果有 direct cache-to-cache transfer，还能进一步缓解。

本小节结论：
对传统 GPU workload，DeNovo 至少不吃亏。

### 6.2.2 Global Synchronization Benchmarks

#### 第1段：HRF 在这里没用

图3展示只用 global synchronization 的四个 benchmark。
因为没有 local scope synchronization，所以 HRF 不起作用。

#### 第2段：DeNovo* 为什么更好

DeNovo* 对 written data 和 global synchronization variable 都获取 ownership，这带来三个关键收益：

1. synchronization variable reuse
   一旦某个 CU 拿到同步变量 ownership，该 CU 内所有 thread blocks 之后的访问都能 hit，直到 ownership 被别人拿走或被 eviction。

2. data reuse across synchronization
   owned data 在 acquire 时不会被 invalidate，因此同一个 CU 的线程块可以跨同步点复用数据。

3. release 更省流量
   release 不再是把 dirty data writethrough 到 L2，而是通过获取 ownership 来管理数据，所以 traffic 更少。

#### 第3段：整体结果

尽管 ownership 也可能有开销，但在 global synchronization benchmark 中，reuse 收益更大。

平均结果：

* execution time 降低 28%
* energy 降低 51%
* network traffic 降低 81%

这个结果是全文最“亮眼”的之一：
作者用它来证明，global synchronization 下根本不需要 scopes，DeNovo+DRF 反而更强。

### 6.2.3 Local Synchronization Benchmarks

#### 第1段：重点对比对象

对于 mostly local synchronization benchmark，重点比：

* DeNovo-D
* GPU-H

因为 GPU-H 是 GPU 协议里最好的。

#### 第2段：两者都比 GPU-D 好，但方式不同

GPU-H：

* 靠 local scope 让数据跨 local sync 重用
* local sync 在 L1 执行
* 但前提是程序必须显式标注 local scope

DeNovo-D：

* 不靠 scopes
* 通过 ownership 隐式复用
* 所有 synchronization operations 都获取 ownership
* 所以即使 global sync 也可能在本地执行

#### 第3段：在 hybrid benchmark 中 DeNovo 也受益

像 TB_LG 和 TBEX_LG 这种同时有 local 和 global synchronization 的 benchmark，
DeNovo-D 因为 atomics 也拿 ownership，所以在 global + local 混合情况下也能提高 locality 和 reuse。

#### 第4段：GPU-H 的 store buffer 问题

因为 GPU-H 不获取 ownership，所以一旦遇到 globally scoped release，还是得把 dirty data flush/downgrade 到 L2。
如果 store buffer 太小，就会限制写 coalescing。

TB_LG 和 TBEX_LG 就出现了这种情况。

#### 第5段：DeNovo-D 为什么在这类 benchmark 上流量更低

DeNovo-D 也会偶尔遇到 full store buffer，但它 flush 的代价更低：

* 每个 dirty cache line 只需要向 L2 发 ownership request
* 一旦获得 ownership，之后写都 hit，不再占用 store buffer

因此 DeNovo-D 在这些 benchmark 中能利用更多 reuse，降低 network traffic 和 energy。

#### 第6段：DeNovo-D 的弱点

但是 DeNovo 只能复用 owned data。
也就是说：

* read-only data 不能跨同步边界复用
* 某些 mostly-read 的 local synchronization benchmark 中，这会拖后腿

作者举出：

* SS_L 会受影响，因为读者先进 critical section，read-write data 会一直被 invalidated，直到 writer 进入并拿到 ownership
* UTS 上 DeNovo-D 也略差，因为它用 global synchronization，必须频繁 invalidate cache 和 flush store buffer

#### 第7段：平均结果

因此平均来看，GPU-H 相比 DeNovo-D：

* execution time 低 6%
* energy 低 4%
* 最大优势分别是 13% 和 10%

但作者再次强调：
这个优势换来的是更复杂的 memory model。

本小节的核心理解：
GPU-H 在 mostly local sync 场景下略好，不是因为它整体机制更先进，而是因为它能保留 read-only / valid data，不用每次 acquire 都把非 owned 的数据扔掉。

## 6.3 DeNovo-D with Selective (RO) Invalidations

### 第1段：为什么加 RO

作者说 DeNovo-D 在 local synchronization benchmark 上输给 GPU-H 的一个关键原因，就是 acquire 时会 invalidating read-only data。

### 第2段：加 RO 后效果

如果给 DeNovo-D 加 read-only region enhancement，那么 GPU-H 在平均 performance 和 energy 上的优势就消失了。

某些个别 case 里，GPU-H 还是稍微好一点，但最多：

* execution time 只好 7%
* energy 只好 4%

### 第3段：为什么 RO 比 HRF 更容易接受

作者强调：
DD+RO 也需要程序信息，但 unlike HRF，这种信息是 hardware agnostic 的。

意思是：

* RO 信息只是在说“这段数据是只读”
* 它不依赖具体 cache hierarchy、具体 scope、具体调度
* 因此比 scope annotation 更自然，也更不易出错

这一节的结论很关键：
DeNovo-D 不是天生输给 GPU-H，只是 baseline 里少了一个 read-only 优化；一旦补上，平均就打平了，而且不需要更复杂的 HRF。

## 6.4 Applying HRF to DeNovo

### 第1段：DeNovo-H 得到了全部好处

DeNovo-H 同时有：

* ownership for data access
* ownership for global synchronization
* local scoped synchronization 的收益

所以：

* owned data 能跨 global sync 复用
* 所有数据都能跨 local sync 复用
* local synchronization 总在本地执行
* global synchronization 在获得同步变量 ownership 后也能本地执行

### 第2段：相比 DeNovo-D 的额外收益

虽然 DeNovo-D 已经能通过 ownership 利用很多 locality，但 DeNovo-H 进一步通过显式 local scope：

* 保留 read-only data
* 保留多次 read 后才写的数据
* 对 local write 和 local synchronization operation，可以延迟获取 ownership

结果就是：
相对 DeNovo-D，DeNovo-H 在所有 local-scope 应用上都降低了 execution time、energy、network traffic。

### 第3段：相比 DeNovo-D+RO 的关系

尽管 DeNovo-D+RO 已能保留 read-only data，但 DeNovo-H 还有上面那些额外优势，所以在少数场景里仍稍微更强。

### 第4段：相比 GPU-H 的关系

相比 GPU-H，DeNovo-H 还能利用更多 locality，因为：

* owned data 可以跨任意 synchronization scope 复用
* synchronization variable registration 让 global synchronization request 也可能本地执行

### 第5段：最终评价

作者说 DeNovo-H 是所有配置中最强的，因为它把：

* ownership 的优势
* scoped synchronization 的优势

全结合起来了。

但问题也很明显：

* 它显著增加 memory model complexity
* 相比 DeNovo-D+RO，并没有显著好很多

所以作者还是更偏向 DeNovo-D+RO 或 baseline DeNovo-D 这条路线。

这一节的含义是：
如果纯看性能，DH 最强；
如果看“性能 + 简洁模型”的综合平衡，DD 或 DD+RO 更有吸引力。

# 7. RELATED WORK

## 7.1 Consistency

### 第1段：以前关于 GPU consistency 的工作

以前有工作研究过 GPU 上的 TSO 和 relaxed memory model，发现它们在 MOESI coherence + writeback caches 的系统里，对 SC 没有显著性能优势。

### 第2段：本文与这些工作的区别

但作者指出，那些工作：

* 没有测 coherence overhead
* 没有比较 alternative coherence protocol

而本文重点恰恰是：
“协议设计 + 一致性模型复杂度 + 性能”一起看。

## 7.2 Coherence Protocols

这一小节把 DeNovo-D 和一些相关 GPU coherence 工作比较。

表5是重点。

### HSC

HSC 是 hierarchical、ownership-based 的 CPU-GPU coherence protocol。
它通过 coarse-grained hardware regions 减少 MOESI 的网络流量。

优点：

* 也有 ownership 的很多优势

缺点：

* coarse regions 会限制 data layout 和 communication pattern
* coherence protocol 比 DeNovo 复杂得多

### Stash / TemporalCoherence / FusionCoherence

#### Stash

Stash 也是 DeNovo 的扩展，但重点是把 scratchpad 等 specialized private memories 纳入 unified address space。
它不支持 fine-grained synchronization，也没有和传统 GPU coherence 做比较。

#### TemporalCoherence / FusionCoherence

这两个是 timestamp-based protocol，用 self-invalidation 和 self-downgrade，所以有些好处和 DeNovo-D 类似。

但它们没有研究 fine-grained synchronization，也没有分析对 consistency model 的影响。

### QuickRelease

QuickRelease 减少 conventional GPU coherence 中 synchronization overhead，并允许数据跨同步点复用。

但它需要：

* broadcast invalidations
* 以确保不会读到 stale data

而且对 dynamic sharing 支持不好。

### RemoteScopes

RemoteScopes 在 QuickRelease 基础上改进 dynamic sharing 支持。
思路是：

* 通常动态共享数据先按 local scope 同步
* 一旦数据被更大范围共享，就“promote” scope

这确实改进了 dynamic sharing 性能，但因为它不获取 ownership，所以仍然需要很重的机制来避免 stale data：

* acquire 时 flush entire cache
* broadcast invalidations + acknowledgments

### Related work 总结

作者总结说：
以前每种协议都只拿到了 DeNovo-D 某些好处，但没有一种同时具备 DeNovo-D 的全部好处。
而且以前工作都没有深入讨论 ownership 对 consistency model 的影响。

这节的作用是把本文定位清楚：
不是第一篇做 GPU coherence 优化的，但它是第一批把“ownership + simpler model”明确绑定起来讨论的工作。

# 8. CONCLUSION

## 第1段：重申背景

GPGPU 应用开始有更一般的 sharing pattern 和 fine-grained synchronization。
传统 GPU coherence 对此支持不好。
已有工作用 HRF + scopes 解决，但模型更复杂。

## 第2段：本文方案

本文选择另一条路：
把 DeNovo 这种 software-driven, hardware coherence protocol 扩展到 GPU。

DeNovo 结合了：

* ownership-based protocol 的优点
* GPU-style protocol 的优点

所以它能高效支持 fine-grained synchronization。

## 第3段：一致性模型意义

更重要的是，DeNovo coherence protocol 支持简单的 SC-for-DRF memory consistency model。

与 HRF 不同，SC-for-DRF：

* 不暴露 memory hierarchy
* 不要求程序员仔细给每个 synchronization access 标 scope

这句话其实是全文最核心的价值判断。

## 第4段：实验结论再总结

对于 10 个 CPU-GPU application：

* 它们不使用 fine-grained synchronization 或 dynamic sharing
* DeNovo 性能与 conventional GPU coherence 相当

对于 global fine-grained synchronization：

* DeNovo 显著优于 conventional GPU coherence

对于 local fine-grained synchronization：

* GPU+HRF 略优于 baseline DeNovo
* 但代价是更复杂的一致性模型

给 DeNovo 加 selective invalidation for read-only regions 后：

* 平均性能和能耗与 GPU+HRF 一样

如果接受 HRF 复杂度，那么 DeNovo+HRF 又比 conventional GPU+HRF 更好。

## 第5段：最终结论

作者最后的结论是：
DeNovo with DRF 在 performance、energy、hardware overhead、memory model complexity 之间提供了 sweet spot。
也就是说，HRF 的复杂度并不是 GPU 高效细粒度同步所必需的。

这句就是你考试时最应该写出来的论文结论。

## 第6段：未来工作

未来工作有两个方向：

1. 等更多 full-sized 应用出现后，验证 DeNovo+DRF 在真实 fine-grained synchronization 应用里是否也有类似收益
2. 继续给 DeNovo+DRF 加其他优化，比如 direct cache-to-cache transfers

这说明论文作者也承认：当前 fine-grained sync 的真实 GPU 应用 benchmark 还不够多，实验更多依赖 microbenchmark。

# 9. REFERENCES

这一节是参考文献表，不属于作者展开论证的新内容，但你记笔记时可以知道论文主要建立在以下脉络上：

## 参考文献脉络

### 1. GPU / HSA 一致性与作用域

包括：

* HSA Platform System Architecture
* HRF
* HRF-Relaxed
* OpenCL 2.0
* GPU shared memory consistency model 的相关论文

这些文献对应本文在 memory model 方面的背景。

### 2. GPU synchronization primitives

包括 Stuart and Owens 关于 GPU synchronization primitives 的工作。
这些文献是本文 microbenchmark 的来源之一。

### 3. DeNovo 系列工作

包括：

* DeNovo
* DeNovoND
* DeNovoSync

这些是本文协议设计基础。

### 4. GPU coherence 相关工作

包括：

* QuickRelease
* TemporalCoherence
* HSC
* Fusion
* RemoteScopes
* Stash

这些是本文 related work 直接比较对象。

### 5. 模拟器与 benchmark 工具

包括：

* GEMS
* GPGPU-Sim
* Garnet
* GPUWattch
* McPAT
* Rodinia
* Parboil

这些是本文方法学基础。

# 这篇论文你考试时最应该记住的主线

## 1. 问题是什么

传统 GPU coherence 很简单，但 fine-grained synchronization 很低效。

## 2. 别人怎么做

别人用 scoped synchronization，提高 local sync 性能，但把 memory model 从 DRF 变成更复杂的 HRF。

## 3. 作者怎么做

作者把 DeNovo 用到 GPU：

* 不用 writer-initiated invalidation
* 不用 directory
* 但写时拿 ownership
* 从而不靠 scopes 也能获得很多 locality / reuse

## 4. 核心发现

### 对 no synchronization 应用

DeNovo 和 GPU 差不多

### 对 global synchronization

DeNovo+DRF 明显优于 GPU+HRF

### 对 mostly local synchronization

GPU+HRF 略优于 baseline DeNovo-D
但只要给 DeNovo-D 加 read-only optimization，平均就打平

### 如果允许 HRF

DeNovo+HRF 最强

## 5. 最核心观点

HRF 的复杂度不是 GPU 高效同步的必要条件。
DeNovo+DRF 是性能、能耗、开销、模型复杂度之间的 sweet spot。

# 一句话总总结

这篇论文真正想反驳的是：
为了让 GPU 支持高效细粒度同步，不一定必须把 memory consistency model 变复杂。

作者给出的答案是：
通过 DeNovo 的 ownership 机制，可以在保持 DRF 简洁性的同时，拿到接近甚至超过 scoped synchronization 的性能。

如果你要，我下一条可以继续给你整理一版：
“适合考前背诵的超精简版”
也就是把上面这份长笔记压成 1～2 页中文记忆提纲。
