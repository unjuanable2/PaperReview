# [wiscECE757[10]] The Bunker Cache for Spatio-Value Approximation (MICRO’16)
用于空间值近似的掩体缓存

## 摘要，介绍
现代处理器里，把数据搬来搬去（尤其是访问片外内存） 的代价仍然远高于计算本身；为了减少片外访问，芯片上用很大面积做缓存。但传统缓存把数据当“比特流”对待，并不利用很多应用本身对“近似”是容忍的这个事实。
论文提出：如果应用结果允许少量误差，我们能不能在缓存层就“用近似换效率”？spatio-value similarity空间-数值相似性：在内存中相隔固定步长（stride）的数据块，往往数值上也“差不多”。
基于这个规律，他们提出 Bunker Cache：只用“地址”就把可能相似的数据块映射到同一个缓存位置，让“错拿一个相近块”在可接受质量损失下，换来更少miss、更少片外访问、更少存储需求。
摘要里给了总体收益区间：性能 1.08×–1.19×、动态能耗节省 1.18×–1.39×（还提到泄漏节省）。


## 什么是 spatio-value similarity？为什么会出现？
数据本质上是对现实世界信息的某种表示。现实世界的信息在空间和时间上都具有高度冗余性，并且信息的变化是平滑且稀疏的。
> e.g. 图像仅仅是场景的二维投影，场景中的颜色通常是聚集的，并以平滑的梯度变化。
> e.g. 声波可以表示为一组稀疏的频率，也可以表示为时域中具有平滑周期性重复的连续信号。
> e.g. 稠密流体中的相邻粒子受到相似的外力作用，并倾向于向相同的速度收敛。
现实世界信息的这种固有特性被抽象化为一维地址空间中的比特。

尽管传统架构忽略了大部分此类信息，但我们观察到数据存储方式仍然存在规律性。用 spatio-value similarity来描述这种规律性：内存中数据值在规则间隔处的相似性
> e.g. 一个三维数据集，每个维度上的一行元素是如何映射到内存的: x 维连续、y/z 维就会以固定“跨度”间隔（or stride）出现。
这些“在数据结构空间相邻”的元素，虽然在一维内存里不连续，但会以规则 stride 出现，从而形成“按 stride 看也相似”的现象。

做了一个很直观的验证方法：强行让相隔某个 stride 处的地址共享同一个存储位置
如果它们之间存在值相似性，那么对应用程序输出质量的影响应该很小甚至没有影响。
扫不同 stride，看输出质量（用 信噪比SNR）怎么变。结果出现“山峰/山谷”并且周期性，峰值位置往往对应数据结构一行（row）的宽度。例如 2dconv 在 stride=1(A[r][c] 和 A[r][c+1])、240(A[r][c] 和 A[r+1][c])、480(A[r][c] 和 A[r+k][c]) 这些位置质量更高，240 对应图像按行存储时每行宽度（以 64B block 计）。

## Bunker Cache 的核心思路
将近似相似的数据块映射到同一个缓存位置, 多个“可能相似”的地址 → 同一个 cache 位置（many-to-one）。
实现：完全包含在缓存索引函数中：空间值相似性映射函数，这使得我们的技术更容易在通用系统中应用
constructive aliasing：主动制造别名来生成近似值，不需要读取/比较数据值来判断相似，只靠地址规律就决定“谁跟谁共用”。
假设应用程序的空间值相似度由程序员或用户指定
这比需要取回数据做相似检测/去重的方案（例如一些 dedup/approx-dedup）更轻。

## Mapping Function（地址 → bunk-address）
### 两个参数：STRIDE 与 RADIX
* STRIDE：认为“相隔 STRIDE 的块”是可接受相似的距离。
* RADIX：映射“有多激进”。radix=2 表示 2-to-1 映射（两个地址合并到一个 bunkaddress）。radix 越大，越省空间/越多“miss→hit”，但质量风险越大。
再派生一个：
* WINDOW = RADIX × STRIDE：把地址空间按 WINDOW 划窗；每个窗里每隔 STRIDE 的块被合并。
### 具体公式（论文 Figure 4）
mapping function 输入物理地址 addr，输出 bunkaddress：bunk@ = (addr / WINDOW) * STRIDE + (addr % WINDOW) % STRIDE
直觉解释：
* (addr / WINDOW) 找到你在第几个窗口（粗粒度分组）
* * STRIDE 把窗口编号缩放回 stride 粒度
* ((addr % WINDOW) % STRIDE) 决定你在该窗口里属于哪一“列”（即哪些地址会被合并）
例子：STRIDE=3、RADIX=2 → WINDOW=6。地址 0..11 会被映射到 bunk@ 0..5，其中每隔 3 的地址对（如 0 和 3，1 和 4…）会合并。
### 成本与可扩展性
STRIDE/RADIX/WINDOW 在运行中大多是静态的，因此除法可用倒数预计算，mapping path 里主要是少量乘加，延迟/面积开销低，适合放在缓存关键路径。另外它可以扩展到 n 维：如果 x/y/z 都有相似性，就按不同 stride 迭代调用 mapping。

## “近似数据”和“精确数据”怎么共存？
只有标注为 approximate 的数据才参与这种 many-to-one 映射；精确数据应保持普通缓存语义。论文给的做法是：在 bunkaddress 里放一位标记 approximate/precise；精确访问直接用物理地址当 bunkaddress（等价于 bypass mapping）。关闭近似时把 RADIX 设为 1 即可旁路并可做 power-gate。
这些“哪些数据可近似”的信息，论文假设通过程序注解 + ISA 扩展传下来。

## Cache 操作与一致性：为什么需要 directory？
由于 many-to-one，一个 bunkaddress 对应多个物理地址。但一致性状态（coherence）和 dirty 位必须按真实物理地址分别追踪，否则会把不同地址的写入/状态混在一起。作者因此要求使用单独的 directory 来按物理地址跟踪一致性/脏位，而 Bunker Cache 的 tag/data array 则按 bunkaddress 索引。
**查找（lookup）**
* 物理地址 → mapping → bunkaddress → 查 Bunker Cache
* 同时物理地址直接查 directory（directory 不走 mapping），所以 directory 的“有效容量”跟基线一样。

**替换（eviction）时的反向映射**
替换某个 bunkaddress 的块时，需要找出哪些物理地址（最多 RADIX 个）映射到它，以便写回/失效等操作。论文说这一步在非关键路径，通过“反向 mapping”（把 STRIDE/WINDOW 交换）做 one-to-many 查 directory。

**写入与（可容忍的）不一致**
因为多个地址共享同一物理 cache line，某个地址的 writeback 可能改变其他地址“看到”的值，甚至造成私有缓存副本之间不一致。论文的立场是：这些地址本来就预计相似，且应用对近似有韧性，因此这种影响可容忍，并在实验中质量没有“严重恶化”。

## 两个增强点：省泄漏 + 动态控质
### Drowsy blocks（省泄漏）
many-to-one 让“有效 footprint”大致按 RADIX 被压缩，所以会出现更多“当前没人映射到的 cache block”。作者建议对这些 block 进入 drowsy 状态省泄漏；需要在 tag 里加一个引用计数（大约 log2(MAX_RADIX)+1 位）记录有多少 directory entry 映射到该 block，计数为 0 就可 drowsy。
### Dynamic Quality Control（动态调 RADIX）
把运行划分成 epoch（按 approximate LLC access 计，实验用 100,000）。每个 epoch 抽样若干次访问（实验 100 次），对抽到的地址 A：
* 无论 hit/miss，都去片外取回 A 的真实数据；
* 还取回 A + WINDOW/2；
* 计算两者 SNR 作为“当前这个 WINDOW 下相似性是否靠谱”的质量探针，累积成 SNRe；
epoch 结束后与目标 SNRt 比较，更新 RADIX：
* 若 SNRe > (1+α)SNRt → 质量“富余”，RADIX 乘 β（更激进换效率）
* 若 SNRe < (1-α)SNRt → 质量不够，RADIX 除 β（更保守）
* α、β 在实验中用 0.5 与 4，并且 RADIX 改变时 flush 掉近似数据。

## 实验里最重要的“现象解释”与“收益来源”
### 为什么 y-similarity 往往比 x-similarity 更好？
论文指出：mapping 在 cache block 粒度（64B） 做，而 x 维是连续维，很多“真正相似的邻居”已经落在同一个 64B block 里；stride=1 的 x-sim 映射反而可能把元素映射到更远的位置（例如像素 [i][j] 被等价到 [i][j+16]），更伤质量。y 维不是连续维，配合合适 stride 更容易做到 [i][j] ↔ [i+1][j] 这种“真正邻居”的替换，所以质量更好。
### “性能/能耗为什么提升？”
根因是 LLC miss 显著减少：RADIX 越大，同一 cache line 覆盖的“地址集合”越大，越容易把原本的 miss 变成 hit，从而减少片外访问（延迟与能耗都省）。论文强调这甚至对 compulsory miss 也有帮助，因为你可能“拿到的是相似块”，不必真的去内存拿精确块。

## 论文给出的关键量化结果（你读结论时最常被问的）
* 在他们的评测设置下，随 RADIX 增大，平均 speedup 和动态能耗节省上升：
    * RADIX=2：平均 1.08× speedup、1.18× energy savings
    * RADIX=64：平均 1.19× speedup、1.39× energy savings
* 某些 benchmark 在 RADIX=4 就能有较高 speedup：debayer、histeq、jpeg 分别 1.29×、1.58×、1.10×，且对应动态能耗节省 1.61×、1.72×、1.28×（作者还解释了为什么 jpeg 虽 footprint 大但更 compute-heavy，所以 speedup 不如 histeq）。
* 输出质量方面（以 y-sim 为主），平均 SNR 随 RADIX 增大下降：例如 RADIX=2/8/64 时 SNR 约 18.8/12.5/8.2 dB（并给了对应的 NRMSE 量级）。
* 在缓存压缩维度，Bunker Cache 的“压缩率”随 RADIX 增大上升，并与 BΔI、uniDoppelganger 做了比较，强调互相正交可叠加。

 
