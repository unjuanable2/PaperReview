# [wiscECE757[7]] SLE/Speculative Lock Elision: Enabling Highly Concurrent Multithreaded Execution
推测性锁省略：实现高度并发的多线程执行

Summary: 推测性锁省略/SLE (一种新型架构技术):
很多时候临界区其实不会真的冲突，于是硬件“投机地”让多个线程同时执行临界区；一旦用缓存一致性机制检测到真正冲突, 再回滚并按传统加锁执行。
 硬件动态地识别同步操作，预测它们是不必要的，将其移除（好像无同步一样）。
成功的推测性省略无需获取锁即可验证并提交。
锁不一定要“被获取（acquired）”，它更多只需要“被观察（observed）”

## Introduction/动机/背景：
**为什么锁会拖慢多线程？临界区串行化**
In multithreaded programs, 传统锁：一个线程拿锁→执行code in 临界区→释放；其他线程等待，serialize access, 使临界区 原子执行/atomic
E.g. a store instruction to a shared object
但现实中很多“加锁”是保守的/ 很多accesses do not require serialization：
E.g. Most dynamic executions don’t perform the store operation => not require lock
E.g. 多线程更新different fields of a shared object（例如哈希表不同 key），逻辑上互不冲突，但锁把它们硬串行了。

**为什么当前处理器如此保守地用锁？**
In 传统 speculative execution in out-of-order processors, 不能充分利用并行性 是因为 缺乏dynamically detect such false inter-thread dependences. 

**为什么不从软件/程序方面提高并行？**
程序员需要一定的专业知识才能写出高性能的多线程程序，使用finer granularity lockng 来提高性能/增加了编程复杂性
理想情况下，程序员可以继续写“明显正确+简单+保守”的锁，硬件在运行时自动移除所有此类保守的同步机制（并保证正确）

=》 提出 SLE, 有以下特征/目标：
- Enable highly concurrent multithreaded execution, 且 不需要真正修改锁变量来判断正确性
- Simplifies correct multithreaded code development
- 易于实施：可以完全在微架构中实现，无需指令集支持和系统级修改; 对程序员透明，程序员不需要学习新的编程方法

 
## SLE的概念：
**lock-enforced control dependence**
锁本质是个控制变量，其值决定了线程的控制流，是a memory location checked and operated upon by the threads；线程对它的读写把“控制依赖”体现成了“数据依赖”（线程真正冲突的不是“锁变量”，而是临界区里共享数据），于是线程被迫等待。

**如何消除这些false dependence due to locks？/ 原子性atomicity的条件:**
SLE 的核心不是“不要锁”，而是：即使不拿锁，也要让临界区看起来像原子执行。论文给出两条必须满足的条件：
1. 临界区里读到的数据，在临界区完成前不能被其他线程改写。
2. 临界区里写的数据，在临界区完成前不能被其他线程读或写。
做到了这两点，就不会出现“别人看到我写了一半”的中间态，整个临界区对外仍像原子发生。

**SLE 的算法**
当看到“像 lock-acquire 的操作”时：
1. 预测：这次临界区可以保持原子性，于是省略elide 加锁lock-acquire operation
2. 临界区内：	Execute speculatively, buffer results.
3. 如果发现原子性无法保证（发生冲突）: trgger misspeculation，回滚，改为真正去拿锁再执行。
4. 如果走到 lock-release 位置仍未触发冲突: 说明原子性没破坏, 省略elide 解锁lock-release operation，提交投机状态commit speculative state, exit speculative critical section。
重启阈值restart threshold: 发生冲突后可以“再赌几次”，超过阈值就不赌了，直接拿锁，保证前进性。

上述算法要求处理器识别锁获取和释放操作。但现实里：
* 加锁通常是用一些“底层同步原语/指令”实现的（比如 test-and-set、CAS、LL/SC 之类）。这些底层同步指令不一定只用来实现加锁，它们也可能被用来实现别的并发逻辑（比如无锁队列、引用计数、一次性初始化等）。
* 解锁往往只是“普通的 store 写内存”，从指令层面根本看不出它是“解锁”。
所以处理器并没有“精确语义信息”能100%确定：某次同步原语一定是 lock-acquire、某次普通写一定是 lock-release，最多只能“猜”它们像是在做锁操作。

**为什么（即使without precise 语义的semantic information）仍然可以移除预测的锁获取和释放操作？静默写对/silent store-pair**
锁的 acquire 通常把 lock 从 FREE 写成 HELD；release 又把 HELD 写回 FREE。如果临界区最终像原子执行，那么 acquire 的那次写 + release 的那次写 合起来对最终体系结构状态是“抵消”的/ 作为一对它们是“静默的”。
因此 SLE 不一定非要“懂语义上这是锁”，它可以只观察到一种模式：
* 某个地址被写成 A（像 acquire），
* 之后不久又被写回原值 B（像 release），
* 期间的内存访问能保持原子性，且没有其他线程修改这个 lock 地址 → 那这两次写都可以被省略。

修改后SLE的完整算法 （硬件无需了解内存访问是否针对锁变量的语义信息，仅跟踪值的变化并观察来自其他线程的请求。）
1. 如果对某个地址进行候选加载（ldl_l）后，紧接着对同一地址进行存储（获取锁的 stl_c），则预测很快就会发生另一次存储（释放锁），将内存位置的值恢复到此存储（获取锁的 stl_c）之前的值。（预测1: 预测接下来会执行另一次存储操作，并撤销本次存储操作所做的更改。该预测无需实际执行存储操作即可完成，但需要监控存储操作的内存位置。如果预测正确，则省略这两次存储操作。）
2. 预测 memory operations in critical sections will occur atomically, 省略elide 加锁lock-acquire operation.
3. 临界区内：Execute speculatively, buffer results. （预测2: 在两个省略的存储操作所界定的窗口内的所有内存操作都是原子性的。）
4. 如果发现原子性无法保证（发生冲突）: trgger misspeculation，回滚，改为真正去拿锁再执行。
5. 如果走到 lock-release 位置仍未触发冲突: 说明原子性没破坏, 省略elide 解锁lock-release operation，提交投机状态commit speculative state, exit speculative critical section。

如果线程无法记录两个存储之间的访问，或者硬件无法提供原子性，则会触发错误推测，并从指令 6 重新开始执行。重新开始时，如果已达到重新开始阈值，则执行将以非推测的方式进行，并获取锁。

**为什么SLE能正常工作？**
上述两次预测并不依赖于程序的语义

任何部分更新都不会对其他线程可见。这样做保证了临界区语义。

架构状态保持不变。无论是否使用存储省略，第二次省略存储操作结束时的架构状态都相同。

如果另一个线程通过写入操作显式地获取了锁，则会触发错误推测，因为所有推测线程都会自动观察到该写入操作。

Nested locks. While it is possible to apply the elision algorithm to multiple nested locks, we apply it to only one level (can be any level and not necessarily the outermost) and any lock operations within this level are treated as speculative memory operations.

Memory consistency. No memory ordering problems exist because speculative memory operations under SLE have the appearance of atomicity. Regardless of the memory consistency model, it is always correct for a thread to insert an atomic set of memory operations into the global order of memory operations.

## 实施SLE
### 启动推测/发起投机
使用过滤器来检测推测的候选对象（例如，Idl_l/stl_c这些锁对），由程序计数器索引（同一个 lock-acquire 序列每次执行，PC 都差不多，所以硬件用 PC 来识别“这是同一把锁的那段代码”）。

每个锁对都被分配一个置信度指标（过去执行历史里：这段锁“投机省略”成功的概率高不高？）。

如果处理器预测锁将被持有，则它假定由于自身无法省略锁，该锁必须由另一个处理器获取。
=》 当硬件预测“现在锁大概率被别人拿着（held）” ，i.e. 有另一个处理器/线程没能省略这把锁
=》 在这种情况下，处理器不会启动推测。

这是一种保守的方法，但有助于防止在极端情况下性能下降。更精确的置信度估计是未来研究的一个重要方向。

### Buffering speculative state
为了从 SLE 误判中恢复，必须先缓冲寄存器register state和内存状态，直到 SLE 得到验证。

**寄存器投机状态Speculative register state**
1. 重排序缓冲区/Reorder buffer/ROB: 把临界区里的指令都留在 ROB 里，不真正“提交到架构状态”
    - 优势：可以利用已用于分支预测错误的恢复机制。
    - 劣势：ROB 的大小限制了临界区的大小（以动态指令的数量计）。
2. 寄存器检查点/Register checkpoint：在开始 SLE 投机的那一刻，把“寄存器的正确状态”拍一张快照。然后临界区执行时，指令可以放心地更新寄存器文件，甚至可以“投机退休”（speculatively retire），从 ROB 里移走。因为：如果失败，直接用 checkpoint 恢复寄存器就行
   - 可以是依赖关系映射（物理寄存器的释放方式可能存在某些限制），也可以是架构寄存器状态本身
   - 优势：使用检查点消除了临界区大小的限制

**内存投机状态Speculative memory state：扩展 write-buffer 来“憋住”投机写**
现代 CPU 可以投机执行 load，但通常不会投机“提交 store 到内存系统”。原因：load 读错了还能回滚不太难；但 store 一旦真写进 cache/内存、别人看见了，就麻烦了。
=》SLE 的做法：增强 CPU 和 L1 cache 之间本来就有的 write-buffer/写缓冲，让临界区里的写先进入 write-buffer
- 在 “lock elision 被验证成功” 之前，这些写不允许下沉到更低层内存层次（L1/更下层）
- 如果 misspeculation：直接把 write-buffer 里这些投机写 作废invalidate，就等于“这些写从没发生过”

此外/额外收益：因为成功的 SLE 要求临界区对外看起来原子完成，那么临界区内部对同一地址/同一 cache line 的多次写，外界反正只会看到“最终结果”，所以 write-buffer 里可以把多次写合并/merge，减少真正要提交的写流量，而且这件事不依赖具体内存一致性模型（因为外界只看到一个原子结果）。

### 误投条件及其检测
不用新协议，直接利用来检测跨线程冲突。

导致misspeculation的原因1: 原子性违例/atomicity violation。如何检测到？可直接利用现有 缓存一致性/coherence 机制
一些处理器里，LSQ（Load/Store Queue）会在缓存收到外部 invalidation 时被 snoop（监听/检查），用来支持更激进的内存一致性实现。
- 如果 SLE 用 ROB 方法，那么无需额外机制来跟踪“投机读取过的内存位置被外部写入”，因为 LSQ 已经在 snoop 这些外部 invalidation。
- 但如果用register checkpoint，load 可能会投机退休并离开 ROB，所以单靠 LSQ 不能检测这类 load 的违例。对策是记录投机临界区中读/写过哪些 cache block，给每个 cache block 增加一个 access bit，投机期间访问过就置位；外部请求/失效时查这个 bit：
	- 如果另一个核对我读过的块发来 invalidation（别人写了） → misspeculation
	- 如果别人请求我已经独占（exclusive）且我写过/将写的块 → misspeculation

该方案与缓存层级数无关，因为所有 cache 都保持一致性，任何更新都会被现有协议自动传播到所有一致性 cache。

在投机失败或投机成功提交（这次 SLE 结束了，这些标记就没意义了）时，要把 cache 里所有块的 access bit 清零；可以用 flash invalidation 之类技术实现。
对激进乱序处理器（乱序 CPU 会让指令提前/延后执行，所以硬件很难“精确知道”某个 load 在程序语义上是不是已经进入临界区）：当一个candidate store（即预测是 lock-acquire 的那次 store）被 decode 后（硬件就把它当成“临界区开始点/投机起点”的信号），处理器之后发出的任何 load （可能实际上不属于临界区）都把对应 cache line 的 access bit 置位（但保守地标记总是正确的）。

流水线里可以同时存在多个候选 store（多个预测的 lock-acquire）；只要 core 里还有任何一个候选 store，load 就继续保守标记 access bit。


导致misspeculation的原因2: Violations due to resource constraints
1. Finite cache size. checkpoint 方案下，load 可能退休离开 ROB/LSQ，所以要靠 cache 里的 access bit 来记“读过哪些 cache line”, 但 cache 容量有限、能打标记的块有限
2. Finite write-buffer size. The number of unique cache lines modified exceeds the write-buffer size.
3. Finite ROB size. If the checkpoint approach is used, the ROB size is not a problem.
4. Uncached accesses or events未缓存的访问或事件 (e.g., some system calls). 有些操作不走正常 cache/一致性跟踪路径（或会引入不可控副作用），硬件没法像普通 load/store 那样监控冲突或回滚; 遇到这种“无法追踪”的访问/事件，就必须退出投机。

对于条件 1、2 和 3，并非总是需要重启（传统做法）：资源不够 → 判定 misspeculation → 回滚 → 真的加锁 → 从头再跑临界区。
资源快不够了（比如 write-buffer 快满） 时，开始真的去写锁变量，把锁拿到手，其他线程就会被锁挡住，后续就不需要再投机监控那么多了。
但要确保：写锁成功前，如果一直没有发生冲突/原子性没破坏，那你前面投机执行的那段就等价于“一直持有锁执行”，所以可以 直接提交并继续，不用回滚重跑。


### Committing speculative memory state
要解决的问题：临界区里的 写store 先放在 write-buffer，别真的写进 cache/内存，失败就丢掉。如果投机成功，怎么把这些写“提交”出去，并且对外看起来像“一瞬间全生效”（原子性要求）？

cache 有两部分：状态（state） 和 数据（data）。只要数据不投机地改变，状态可以投机地变化。
The cache coherence protocol determines state transitions of cache block state.
当一个投机 store 进入 write-buffer 时：
1. speculative data is buffered in the write-buffer投机数据缓存在 write-buffer, 先不改
2. 同时发 exclusive request/ 独占请求 去拿权限（只动 state）引发一致性协议已有的状态转换，把对应 cache block 拉到本地 cache 的 exclusive state。=> write-buffer 里每个投机条目，都应该已经对应一个 在 cache 里处于 Exclusive 状态的 block；否则早就因为冲突/权限拿不到触发 misspeculation 了
(直觉：还没“真正写值”，但你先把“写这块内存所需的独占权”拿到了)
当临界区结束时，只要所有相关块都已 exclusive，就可以把 write-buffer 标记为“最新架构态”(提交的那一刻翻1-bit)，之后 write-buffer 可慢慢 drain 到 cache。

write-buffer 需要额外功能：能够对来自其他线程的请求提供数据source data
如果另一个核心此时来读你已经投机写过的地址：
因为那条 cache line 已经在 Exclusive 状态，别的核心会发 coherence 请求; 你的 cache 里 data 可能还是旧值（因为新值在 write-buffer）
所以必须让 write-buffer 能“转发/提供”新值给请求者


## 评估方法 & 结果
### 微基准microbenchmark
微基准: N个线程各加自己的计数器，所有计数器都被同一把锁保护（传统锁会严重争用）；SLE 发现其实不冲突，于是几乎完美扩展

### 基准测试
论文展示了在多种系统配置（CMP/SMP/DSM）下：
* 大量 lock acquire/release 对被省略，但这不一定总等价于同等比例的加速（锁不在关键路径时）。
* 主要收益来源有三类：
    1. 临界区并发执行（减少锁串行）
    2. 降低观察到的内存延迟（锁保持 shared，不用为 acquire 付出 cache miss+写请求）
    3. 降低一致性流量（少了锁的来回争夺）

### 重启阈值的影响
阈值太大可能因为反复投机造成一致性干扰，甚至让某些场景变慢；
阈值 0 或 1 通常更稳, 可以最大限度地减少因重复错误推测而导致的性能下降。

## 相关工作：事务内存 / OCC
事务内存/Transactional Memory 也想让并发临界区“像事务一样”成功就提交、失败就回滚。但SLE 强调自己不需要 ISA 扩展、不需要一致性协议扩展、不需要程序员参与，冲突就退回传统加锁路径。

（数据库并发控制领域）OCC 包含一个读取阶段，在此阶段访问对象（并可能更新这些对象的私有副本），随后是一个串行化的验证阶段，用于检查数据冲突（与其他事务的读/写冲突）。如果验证成功，则进入写入阶段。
数据库系统的特殊要求和保证使得 OCC 难以用于高性能应用。为了提供这些保证，必须在软件中存储大量的状态信息，从而导致巨大的开销。