# 3.X 面向跨模态特征对齐的对比损失设计

## 3.X.1 原始损失函数与跨模态场景下的失配

EfficientLoFTR 的原始损失函数由三项组成：粗匹配损失 $\mathcal{L}_c$（在 8 倍下采样网格上的二分类 focal 损失）、像素级匹配损失 $\mathcal{L}_f$（在粗匹配窗口内的二分类 focal 损失）以及亚像素回归损失 $\mathcal{L}_l$（在邻域 $3\times 3$ 内基于 soft-argmax 的 L2 回归）。三者构成一条以空间精度为轴的层级化精修链，分别将匹配精度从 patch 级（约 8 像素）依次精修至像素级与亚像素级。

该损失体系在同模态匹配任务（如 MegaDepth、ScanNet 上的 RGB-RGB 匹配）中表现良好，其有效性依赖于一个隐含假设：**两幅输入图像的低层视觉表观相近，使得对应位置的特征向量在原始特征空间中已具有可分辨的高相似度**。在该假设成立时，相似度矩阵中正样本位置的取值显著高于同行同列的其他位置，dual-softmax 概率分布锐利，focal 损失能够将正样本概率推向 1。

然而在红外—可见光（IR-VIS）跨模态匹配场景下，上述假设不再成立。红外模态主要响应物体的热辐射特性，可见光模态主要响应物体的反射光谱特性，二者所成像的物理量本质不同，导致对应物理点在两个模态下提取的特征向量先验地存在显著差异。这一差异引发原始损失函数在以下三方面发生信号失配，造成训练困难。

## 3.X.2 失配机制的三层分析

设 $\mathbf{f}_0^{(i)}$ 和 $\mathbf{f}_1^{(j)}$ 分别表示经粗匹配 Transformer 输出的 IR 与 VIS 特征 token，相似度矩阵定义为 $S_{ij} = \mathbf{f}_0^{(i)} \cdot \mathbf{f}_1^{(j)} / \tau_d$，其中 $\tau_d = 0.1$ 为 dual-softmax 温度。粗匹配概率为

$$
P_{ij} = \mathrm{softmax}_j(S)_{ij} \cdot \mathrm{softmax}_i(S)_{ij}, \qquad (3\text{-}1)
$$

粗匹配 focal 损失为

$$
\mathcal{L}_c = -\frac{1}{|\mathcal{P}|} \sum_{(i,j)\in \mathcal{P}} \alpha\,(1-P_{ij})^\gamma \log P_{ij}, \qquad (3\text{-}2)
$$

其中 $\mathcal{P}$ 为真值正样本集合，$\alpha = 0.25$，$\gamma = 2.0$。在跨模态场景下，该损失存在如下三层失配。

**(1) 概率压缩导致 focal 损失梯度对相对改进不敏感。** 当 IR-VIS 整体特征相似度偏低时，$S$ 矩阵中正负样本的差距显著小于同模态情形，经 dual-softmax 后正样本概率 $P_{ij}$ 长期集中于 $[0.05,\,0.10]$ 区段。对 focal 损失关于 $P_{ij}$ 求导可得

$$
\left|\frac{\partial \mathcal{L}_c}{\partial P_{ij}}\right| = \alpha (1-P_{ij})^{\gamma-1} \left|\gamma \log P_{ij} \cdot P_{ij} - (1-P_{ij}) \right| \big/ P_{ij}. \qquad (3\text{-}3)
$$

在该区段内，正样本概率从 0.05 提升至 0.10 时损失数值的变化不足 30%，损失对模型相对进步的反馈信号弱，模型容易在子优解附近停滞。

**(2) 互选硬约束抑制弱信息密度模态的梯度回传。** 公式 (3-1) 中 dual-softmax 的乘积结构要求 IR→VIS 与 VIS→IR 两个方向同时具备锐利分布。由于红外图像信息密度普遍低于可见光图像，IR 端的 $\mathrm{softmax}_j(S)$ 分布往往平坦，乘积结构使得 VIS 端原本锐利的分布被 IR 端的不确定性压低，导致 VIS 端特征即便已具备良好可分性，其梯度信号仍受 IR 端拖累而衰减。

**(3) 缺乏模长约束引入"模长捷径"。** dual-softmax 输入未对特征向量执行归一化，导致存在如下退化解：模型可通过整体放大特征模长来锐化 softmax 分布，从而降低 $\mathcal{L}_c$，但该路径并未实质改善特征向量方向上的语义对齐。该退化解在同模态训练中影响有限，但在引入模态嵌入（modality embedding）等显式跨模态结构时，会与模态嵌入争夺优化方向。

综上，原始损失 $\mathcal{L}_c$、$\mathcal{L}_f$、$\mathcal{L}_l$ 虽然在概念上隐含"对应位置特征应当相似"的监督意图，但该意图被 dual-softmax 这一中间运算的概率压缩、互选耦合与模长无关性三层机制层层衰减，难以有效传递至模型的特征表示空间。

## 3.X.3 对称 InfoNCE 对比损失的设计

针对 3.X.2 节分析的失配机制，本文在原始损失体系外引入一项跨模态对比损失 $\mathcal{L}_{\text{ct}}$，其设计原则如下：

1. **直接监督特征向量方向，绕过 dual-softmax 中间层**——使监督信号无需穿过概率压缩与互选耦合环节。
2. **对方向监督施加模长不变性约束**——通过 L2 归一化使损失仅取决于 cosine 相似度，消除"模长捷径"。
3. **解耦正向与反向匹配方向**——以独立的两个分类项分别监督 IR→VIS 和 VIS→IR，避免任一方向的弱信号拖累另一方向。
4. **扩展负样本采样域至 batch 内全部跨场景 token**——为低相似度区段提供持续的对比梯度。

具体地，记 batch 大小为 $B$，IR 与 VIS 在粗匹配 Transformer 出口处的 token 序列分别为 $\mathbf{F}_0 \in \mathbb{R}^{B\times L_0\times C}$ 与 $\mathbf{F}_1 \in \mathbb{R}^{B\times L_1\times C}$（$L_0, L_1$ 为 token 数，$C$ 为特征维度）。先对其执行 L2 归一化得到

$$
\hat{\mathbf{f}}_0^{(b,i)} = \frac{\mathbf{f}_0^{(b,i)}}{\|\mathbf{f}_0^{(b,i)}\|_2}, \qquad \hat{\mathbf{f}}_1^{(b,j)} = \frac{\mathbf{f}_1^{(b,j)}}{\|\mathbf{f}_1^{(b,j)}\|_2}. \qquad (3\text{-}4)
$$

对于由粗匹配真值给出的正样本三元组集合 $\mathcal{A} = \{(b, i, j)\}$，定义双向 InfoNCE 损失：

$$
\mathcal{L}_{i \to v} = -\frac{1}{|\mathcal{A}|} \sum_{(b,i,j) \in \mathcal{A}} \log \frac{\exp\!\left(\hat{\mathbf{f}}_0^{(b,i)} \cdot \hat{\mathbf{f}}_1^{(b,j)} / \tau_c\right)}{\sum_{b'=1}^{B}\sum_{j'=1}^{L_1} \exp\!\left(\hat{\mathbf{f}}_0^{(b,i)} \cdot \hat{\mathbf{f}}_1^{(b',j')} / \tau_c\right)}, \qquad (3\text{-}5)
$$

$$
\mathcal{L}_{v \to i} = -\frac{1}{|\mathcal{A}|} \sum_{(b,i,j) \in \mathcal{A}} \log \frac{\exp\!\left(\hat{\mathbf{f}}_1^{(b,j)} \cdot \hat{\mathbf{f}}_0^{(b,i)} / \tau_c\right)}{\sum_{b'=1}^{B}\sum_{i'=1}^{L_0} \exp\!\left(\hat{\mathbf{f}}_1^{(b,j)} \cdot \hat{\mathbf{f}}_0^{(b',i')} / \tau_c\right)}, \qquad (3\text{-}6)
$$

其中 $\tau_c$ 为对比损失温度。最终对比损失为两方向的均值：

$$
\mathcal{L}_{\text{ct}} = \frac{1}{2}\left(\mathcal{L}_{i\to v} + \mathcal{L}_{v\to i}\right). \qquad (3\text{-}7)
$$

该形式与图文对比学习中的 CLIP 损失同构，但有三处关键差异：(a) 锚点不是图像/文本整体表征，而是由几何监督预先给出的 token 级正样本对；(b) 正样本对在像素级具有严格的几何对应关系，而非弱配对关系；(c) 负样本来自同一 batch 内**所有其他位置的 token**（既包含本场景内其他位置，也包含其他场景内任意位置），构成 $B \cdot L_1$ 与 $B \cdot L_0$ 量级的大型负样本池。这一设计在保留 CLIP 对称结构的同时，将其从图像级对齐细化至 token 级几何对齐。

## 3.X.4 与原始损失体系的关系

$\mathcal{L}_{\text{ct}}$ 与原始三项损失在监督维度上正交：原始损失 $\mathcal{L}_c$、$\mathcal{L}_f$、$\mathcal{L}_l$ 沿空间分辨率轴递进精修匹配位置，而 $\mathcal{L}_{\text{ct}}$ 沿特征空间轴监督特征向量的方向一致性。两者作用于网络的不同模块出口（前者作用于 dual-softmax 概率，后者作用于 Transformer 原始 token），梯度反传路径独立。总损失定义为加权和：

$$
\mathcal{L} = w_c\,\mathcal{L}_c + w_f\,\mathcal{L}_f + w_l\,\mathcal{L}_l + w_{\text{ct}}\,\mathcal{L}_{\text{ct}}, \qquad (3\text{-}8)
$$

其中 $w_c = w_f = 1.0$、$w_l = 0.25$ 沿用原始 EfficientLoFTR 设置，$w_{\text{ct}} = 0.01$ 为本文新增项的权重。引入 $\mathcal{L}_{\text{ct}}$ 不修改原始三项损失的任何超参数与计算路径，保证了实验对照的可控性与向后兼容性——当 $w_{\text{ct}} = 0$ 时模型完全退化为原始 EfficientLoFTR。

特别地，$\mathcal{L}_{\text{ct}}$ 通过强制 L2 归一化（公式 (3-4)）从根本上消除了 3.X.2 节机制 (3) 所述的"模长捷径"，为本文后续章节中模态嵌入（modality embedding）等显式跨模态结构的引入提供了无干扰的优化空间。

## 3.X.5 超参数选取依据

对比损失中的两个关键超参数为温度 $\tau_c$ 与权重 $w_{\text{ct}}$，其取值依据如下。

**温度 $\tau_c = 0.1$。** 该值与原始 dual-softmax 温度 $\tau_d$ 保持一致，确保对比损失中的 logit 量级与匹配分支中的 logit 量级在同一数量级，便于训练初期两项损失梯度幅度相近。在 cosine 相似度 $\in [-1, 1]$ 的输入下，$\tau_c = 0.1$ 使得 logit 落在 $[-10, 10]$ 区间，cross-entropy 梯度在该区间内保持稳定的反馈强度，避免过低温度引起的训练发散与过高温度引起的负样本贡献消失。

**权重 $w_{\text{ct}} = 0.01$。** 该值的选取目标为：训练中后期，加权后的对比损失数值 $w_{\text{ct}} \cdot \mathcal{L}_{\text{ct}}$ 与加权后的粗匹配损失 $w_c \cdot \mathcal{L}_c$ 处于同一数量级（$10^{-1}$ 量级）。InfoNCE 形式损失在 $|\mathcal{A}|$ 量级负样本池下的典型数值范围为 3~6，乘以 0.01 后约为 0.03~0.06，与粗匹配 focal 损失的 0.1~0.5 数量级相当。这一设计使得对比项不会主宰梯度方向，仅作为对原始监督的辅助补强。

**batch 尺寸约束 $B \geq 2$。** 公式 (3-5)、(3-6) 中负样本池跨整个 batch 取样。当 $B = 1$ 时负样本池退化为单图像内的所有 token，与 dual-softmax 行/列归一化所覆盖的样本池基本重合，对比损失提供的额外监督信息消失。因此本文所有启用对比损失的实验均满足 $B \geq 2$。

## 3.X.6 小节小结

本节针对 EfficientLoFTR 原始损失体系在跨模态匹配场景下的三层失配机制（概率压缩导致的 focal 梯度饱和、互选硬约束导致的弱模态梯度抑制、模长无约束导致的退化解），设计了一项对称 InfoNCE 跨模态对比损失 $\mathcal{L}_{\text{ct}}$。该损失通过 L2 归一化下的 cosine 相似度直接监督 Transformer 出口处的特征向量方向，并采用 batch 内跨场景负样本池为低相似度区段提供持续梯度信号。$\mathcal{L}_{\text{ct}}$ 与原始三项损失沿空间分辨率轴的精修目标相互正交，二者的加权组合在不修改原始计算路径的前提下，为后续章节中显式跨模态结构（模态嵌入、模态自适应归一化等）的引入提供了几何与语义解耦的特征表示空间。








# 3.Y 跨模态结构归纳偏置：可学习模态嵌入

## 3.Y.1 监督信号之外的结构性缺失

3.X 节引入的对比损失 $\mathcal{L}_{\text{ct}}$ 在 Transformer 出口处直接监督特征向量方向，从损失层面缓解了 IR-VIS 特征空间的方向失配。然而该干预属于**监督端**调整，并未改变 Transformer 内部的计算结构。本节分析这一结构性缺失。

设 backbone 输出的 IR 与 VIS 粗特征图分别为 $\mathbf{X}_0, \mathbf{X}_1 \in \mathbb{R}^{B\times C\times H\times W}$。粗匹配 Transformer 的每个 self/cross-attention 层将其线性投影为查询、键、值：

$$
\mathbf{Q} = \mathbf{W}_q\,\mathbf{X}_\bullet, \quad \mathbf{K} = \mathbf{W}_k\,\mathbf{X}_\bullet, \quad \mathbf{V} = \mathbf{W}_v\,\mathbf{X}_\bullet, \qquad (3\text{-}9)
$$

其中 $\mathbf{W}_q, \mathbf{W}_k, \mathbf{W}_v \in \mathbb{R}^{C\times C}$ 为**两模态共享**的可学习参数矩阵。在该结构下，attention 仅能依据 token 经投影后的几何相似度 $\mathbf{Q}\mathbf{K}^\top$ 进行加权聚合，无法从输入中获取"该 token 来自哪一模态"的判别信息。结果是：

1. Transformer 对每个 token 的处理是**模态盲**的——同一查询 token 与一个 IR key 和一个 VIS key 计算的相似度，仅取决于二者在特征空间中的几何相似度，与其模态归属无关；
2. 跨模态 attention 的"模态适应"必须由数据反复教会，而非来自结构先验。

3.X 节的 $\mathcal{L}_{\text{ct}}$ 仅作用于 Transformer 出口，能够在收尾阶段拉近真值正样本对的特征方向，但并未在 attention 内部为模型提供模态识别能力。换言之，**$\mathcal{L}_{\text{ct}}$ 给出了纠偏目标，但未给出能够利用这一目标的内部信号通路**。本节针对此结构性缺失，引入一项轻量级可学习模块——模态嵌入（modality embedding）——作为对 $\mathcal{L}_{\text{ct}}$ 的结构化补足。

## 3.Y.2 注入位置的可达性分析

要使模态身份信号在 Transformer 内部可达，需选择一个能让该信号穿越后续所有计算路径而不被消去的注入位置。考察 EfficientLoFTR 的前向链路：

$$
\text{image} \xrightarrow{(a)} \text{backbone} \xrightarrow{(b)} \text{coarse Transformer} \xrightarrow{(c)} \text{fine\_preprocess} \xrightarrow{(d)} \text{fine\_matching}.
$$

定义"模态偏置"为一对可学习 $C$ 维常向量 $(\mathbf{e}_{\text{ir}}, \mathbf{e}_{\text{vis}}) \in \mathbb{R}^C \times \mathbb{R}^C$，分别按通道维广播加到 IR / VIS 特征上。逐位置分析其有效性如下。

**位置 (a)：原始图像或 backbone 内部。** backbone 中存在大量 BatchNorm 层，BN 的归一化对通道级常量偏置满足

$$
\mathrm{BN}(\mathbf{x} + \mathbf{b}) = \mathrm{BN}(\mathbf{x}), \quad \forall \mathbf{b} \in \mathbb{R}^C, \qquad (3\text{-}10)
$$

注入的常量模态偏置在通过第一个 BN 层后即被完全归零，信号不可达。

**位置 (b)：backbone 输出之后、Transformer 之前。** 此处后续运算为 LayerNorm 与公式 (3-9) 中的 Q/K/V 投影。LayerNorm 在通道维归一化但保留了 token 间的差异；而 Q/K/V 投影是矩阵乘法 $\mathbf{W}_q(\mathbf{X} + \mathbf{e}) = \mathbf{W}_q\mathbf{X} + \mathbf{W}_q\mathbf{e}$，**模态偏置同时进入查询、键、值三个分支**。由于粗匹配 Transformer 每层 attention 后均接残差连接 $\mathbf{x} \mapsto \mathbf{x} + \mathrm{Attention}(\mathbf{x})$，该偏置经一次注入后能够通过残差通路传递至所有后续层。即：**位置 (b) 是唯一可使模态信号穿越 attention 内部的位置。**

**位置 (c)：Transformer 出口之后、fine_preprocess 之前。** fine_preprocess 的两条 outconv 路径中各包含一个 BatchNorm2d，由公式 (3-10) 知偏置再次被消去，对 fine 路径无影响。

**位置 (d)：fine_preprocess 之后、fine_matching 之前。** fine_matching 在邻域窗口内计算双向 softmax 加 argmax：

$$
P_{lr} = \mathrm{softmax}_l(S)_{lr} \cdot \mathrm{softmax}_r(S)_{lr}, \quad (\hat{l}, \hat{r}) = \arg\max_{l,r} P_{lr}, \qquad (3\text{-}11)
$$

其中 $S_{lr} = \mathbf{f}_0^{(l)} \cdot \mathbf{f}_1^{(r)} / \sqrt{C}$。若给两侧分别加上常量偏置 $\mathbf{f}_0' = \mathbf{f}_0 + \mathbf{b}_{\text{ir}}$、$\mathbf{f}_1' = \mathbf{f}_1 + \mathbf{b}_{\text{vis}}$，内积展开后多出三项：$\langle \mathbf{b}_{\text{ir}}, \mathbf{f}_1^{(r)}\rangle$（仅与 $r$ 有关）、$\langle \mathbf{f}_0^{(l)}, \mathbf{b}_{\text{vis}}\rangle$（仅与 $l$ 有关）、$\langle \mathbf{b}_{\text{ir}}, \mathbf{b}_{\text{vis}}\rangle$（全局常量）。行向 softmax 与列向 softmax 分别将"仅与列有关项"与"仅与行有关项"完全消除（softmax 对加性常量不变），全局常量在归一化时同样被消去。故 argmax 位置不变，**位置 (d) 的注入对像素级匹配输出无任何影响。**

综合上述四个位置的可达性分析，得到表 3-1。

**表 3-1** 模态偏置在不同注入位置的可达性

| 注入位置 | 消去机制 | 实际有效性 |
|---|---|---|
| (a) backbone 内部 | BN 通道级归一化 | 无效 |
| **(b) Transformer 输入端** | （无消去机制） | **有效** |
| (c) fine_preprocess 输入端 | fine_preprocess 内 BN | 无效 |
| (d) fine_matching 输入端 | softmax + argmax 平移不变 | 无效 |

该分析表明：**结构性模态注入存在唯一可达位置 (b)**。本文据此选择在 backbone 输出之后、粗匹配 Transformer 之前引入模态嵌入。

## 3.Y.3 模态嵌入的设计与位置编码的关系

记 IR 与 VIS 粗特征图分别为 $\mathbf{X}_0, \mathbf{X}_1 \in \mathbb{R}^{B\times C\times H\times W}$，引入两个可学习 $C$ 维参数向量 $\mathbf{e}_{\text{ir}}, \mathbf{e}_{\text{vis}} \in \mathbb{R}^C$，将其按通道维广播加到对应模态的特征图上：

$$
\tilde{\mathbf{X}}_0 = \mathbf{X}_0 + \mathbf{e}_{\text{ir}} \otimes \mathbf{1}_{H\times W}, \qquad \tilde{\mathbf{X}}_1 = \mathbf{X}_1 + \mathbf{e}_{\text{vis}} \otimes \mathbf{1}_{H\times W}, \qquad (3\text{-}12)
$$

其中 $\otimes$ 表示外积广播。$\tilde{\mathbf{X}}_0, \tilde{\mathbf{X}}_1$ 随后送入粗匹配 Transformer。

**与位置编码的形式同构但语义正交。** 公式 (3-12) 在**注入形式**上与原始 LoFTR 采用的 2D 正弦位置编码（在 backbone 输出端加性注入 Transformer 输入）相同，但二者在编码内容上有本质差异：

- **位置编码**为每个空间位置 $(i,j)$ 提供一个唯一的 $C$ 维向量，编码的是**位置身份**，向量随空间位置变化；
- **模态嵌入**为每个模态提供一个唯一的 $C$ 维向量，编码的是**模态身份**，向量在整张图的空间维度上保持常量。

因此模态嵌入在功能定位上更接近自然语言处理中 BERT 模型采用的 segment embedding：BERT 用 segment-A / segment-B 两个可学习向量标识 token 来自哪个句子，本文用 modemb-IR / modemb-VIS 两个可学习向量标识 token 来自哪个模态。

此外，EfficientLoFTR 在原始 LoFTR 的基础上已将 2D 加性正弦位置编码替换为旋转位置编码（Rotary Position Embedding，RoPE）。RoPE 通过对 Q、K 进行复数旋转编码**相对位置**，作用于 attention 内部、仅影响 Q、K 而不影响 V。模态嵌入 (3-12) 与 RoPE 在作用维度上完全正交：

- 模态嵌入：输入端 / 加性 / 同时影响 Q、K、V / 绝对模态身份；
- RoPE：attention 内部 / 乘性旋转 / 仅影响 Q、K / 相对空间位置。

二者可在同一网络中独立工作，互不干扰。

## 3.Y.4 与对比损失 $\mathcal{L}_{\text{ct}}$ 的协同关系

模态嵌入与 3.X 节引入的对比损失 $\mathcal{L}_{\text{ct}}$ 构成"结构 × 监督"的正交组合，二者作用阶段与机制如表 3-2。

**表 3-2** 模态嵌入与对比损失的功能划分

| 项目 | 作用阶段 | 作用对象 | 机制类别 | 提供的能力 |
|---|---|---|---|---|
| 模态嵌入 $(\mathbf{e}_{\text{ir}}, \mathbf{e}_{\text{vis}})$ | 粗匹配 Transformer **入口** | Q、K、V 的输入 | 结构归纳偏置 | 使 attention 内部具备识别模态归属的能力 |
| 对比损失 $\mathcal{L}_{\text{ct}}$ | 粗匹配 Transformer **出口** | 输出 token 方向 | 监督信号 | 拉近真值正样本对的方向、推开跨场景负样本 |

从信息流角度看，模态嵌入在 Transformer 入口处为每个 token 打上不可消除的模态标签，使 attention 机制能够学到"对 IR 查询应优先 attend 至 VIS 中具有热-反射对应关系的位置"等模态条件路由策略；而 $\mathcal{L}_{\text{ct}}$ 在出口处直接监督路由结果，使 Transformer 学到的注意力策略对应到具有几何对齐的输出 token 对。两者在反向传播中协同：对比损失提供的梯度经过 Transformer 后，能够经由模态嵌入参数 $\mathbf{e}_{\text{ir}}, \mathbf{e}_{\text{vis}}$ 的更新被进一步内化为模型的结构性先验。

特别地，3.X.2 节机制 (3) 指出 dual-softmax 输入未归一化会引入"模长捷径"；$\mathcal{L}_{\text{ct}}$ 通过 L2 归一化（公式 (3-4)）从根本上消除了这一退化解。该消除恰好为模态嵌入提供了无干扰的优化空间——若不归一化，模态嵌入的优化可能与模长方向耦合，使其学到的不是模态身份而是模态相关的全局缩放因子。

## 3.Y.5 实现细节与初始化策略

**参数规模。** 模态嵌入仅新增 $2C = 512$ 个可学习参数（$C = 256$ 为粗匹配 Transformer 通道数），相对于 EfficientLoFTR 全模型约 $1.6 \times 10^7$ 参数，新增量在 $10^{-5}$ 量级，可视为零成本结构改动。

**零初始化策略。** $\mathbf{e}_{\text{ir}}, \mathbf{e}_{\text{vis}}$ 均以零向量初始化。在该初始化下，公式 (3-12) 中的加性偏置在训练第 0 步严格等于零，模型前向输出与未引入模态嵌入时**逐比特一致**。这一性质提供两个关键工程优势：

1. **从同模态预训练权重 finetune 的安全性**——当从 MegaDepth 等同模态数据集预训练得到的权重出发开始跨模态 finetune 时，新增参数不会在训练初期破坏已有性能；模态嵌入需在梯度推动下从零开始增长，其作用的出现严格由数据驱动。
2. **可观测的训练动态**——训练过程中持续记录 $\|\mathbf{e}_{\text{ir}}\|_2$ 与 $\|\mathbf{e}_{\text{vis}}\|_2$ 的范数演化曲线，可用作模态嵌入是否被实际激活的诊断指标：范数从 0 单调增长表明模型从对比损失中收到了"需要区分模态"的信号；范数长期停留在 0 附近则表明梯度信号过弱，提示需切换为小幅高斯初始化 $\mathbf{e} \sim \mathcal{N}(\mathbf{0},\,0.02^2 \mathbf{I})$。

**前向开销。** 公式 (3-12) 仅为一次按通道广播加法，浮点运算量为 $\mathcal{O}(BCHW)$，相对于 Transformer 自身的 $\mathcal{O}(BC^2 + BL^2 C)$（$L = HW$）量级运算可忽略不计。

## 3.Y.6 小节小结

本节在 3.X 节引入对比损失 $\mathcal{L}_{\text{ct}}$ 的基础上，进一步分析了 EfficientLoFTR 粗匹配 Transformer 的结构性模态盲问题——Q、K、V 投影矩阵在两模态间完全共享，attention 内部无任何机制可获知 token 的模态归属。为此本文引入一对可学习的 $C$ 维模态嵌入向量 $(\mathbf{e}_{\text{ir}}, \mathbf{e}_{\text{vis}})$，并通过对网络四个候选注入位置的可达性分析（表 3-1）证明：在 backbone 输出端、粗匹配 Transformer 输入端的加性注入是唯一可使模态信号同时作用于 Q、K、V 并通过残差通路传递至所有 attention 层的位置。模态嵌入在形式上与原始 LoFTR 的加性位置编码同构，但语义上对应自然语言处理中的 segment embedding，与 EfficientLoFTR 现行的 RoPE 在作用维度上完全正交。模态嵌入与对比损失分别构成结构归纳偏置与监督信号，共同作用于 Transformer 的入口与出口，形成对跨模态匹配的双侧约束。该模块仅引入 $2C = 512$ 个参数，采用零初始化策略保证从同模态预训练权重 finetune 的向后兼容性，并提供可观测的范数演化曲线作为训练动态诊断信号。

需指出，本节可达性分析（表 3-1）同时表明：模态嵌入的有效作用范围被限定在粗匹配阶段——其经 Transformer 残差通路传递到 fine_preprocess 输入端的残余信号将被该模块内的 BatchNorm 完全消去，更不可能影响 fine_matching 中基于 argmax 的像素级输出。该结构性局限指向跨模态匹配中独立于 Transformer 的精修阶段瓶颈，构成本文后续章节中模态特异性归一化（modality-specific BatchNorm）等 fine 路径干预的设计动机。

# official

[per-scene] (units: %)
  cloudy_cloudy_scene_1               auc@5:  2.143  auc@10:  7.941  auc@20: 16.837  pairs: 131
  cloudy_cloudy_scene_2               auc@5:  4.897  auc@10: 12.484  auc@20: 25.639  pairs: 197
  cloudy_cloudy_scene_3               auc@5:  0.541  auc@10:  1.807  auc@20:  8.705  pairs: 266
  cloudy_cloudy_scene_4               auc@5:  0.988  auc@10:  2.998  auc@20:  9.391  pairs: 311
  cloudy_cloudy_scene_5               auc@5:  1.923  auc@10:  8.006  auc@20: 19.969  pairs: 180
  cloudy_cloudy_scene_6               auc@5:  4.642  auc@10: 10.732  auc@20: 21.823  pairs: 297
  cloudy_sunny_scene_1                auc@5:  1.914  auc@10:  8.344  auc@20: 22.652  pairs: 195
  cloudy_sunny_scene_2                auc@5:  7.934  auc@10: 16.891  auc@20: 29.369  pairs: 270
  cloudy_sunny_scene_3                auc@5:  2.218  auc@10:  5.990  auc@20: 13.406  pairs: 289
  cloudy_sunny_scene_4                auc@5:  0.544  auc@10:  2.634  auc@20:  8.440  pairs: 454
---
[per-class] (XoFTR aggregiate_scenes equivalent, units: %)
  cloudy_cloudy                       auc@5:  2.522  auc@10:  7.328  auc@20: 17.061
  cloudy_sunny                        auc@5:  3.152  auc@10:  8.465  auc@20: 18.467
---
[overall] (units: %)
  all_class_mean                      auc@5:  2.837  auc@10:  7.896  auc@20: 17.764   # compare with MINIMA paper Table 3 ELoFTR: 2.88 / 7.88 / 17.72
num_matches: 239.76

roadscene

precision@1px: 0.0648
precision@3px: 0.2616
precision@5px: 0.3960


# v0 逐像素匹配

[per-scene] (units: %)
  cloudy_cloudy_scene_1               auc@5:  4.221  auc@10: 13.598  auc@20: 25.454  pairs: 131
  cloudy_cloudy_scene_2               auc@5:  6.105  auc@10: 15.224  auc@20: 30.192  pairs: 197
  cloudy_cloudy_scene_3               auc@5:  7.412  auc@10: 16.122  auc@20: 28.664  pairs: 266
  cloudy_cloudy_scene_4               auc@5:  5.570  auc@10: 13.260  auc@20: 24.101  pairs: 311
  cloudy_cloudy_scene_5               auc@5: 13.621  auc@10: 27.128  auc@20: 42.018  pairs: 180
  cloudy_cloudy_scene_6               auc@5:  2.477  auc@10:  5.132  auc@20: 10.556  pairs: 297
  cloudy_sunny_scene_1                auc@5:  1.586  auc@10:  6.588  auc@20: 18.508  pairs: 195
  cloudy_sunny_scene_2                auc@5:  5.592  auc@10: 14.356  auc@20: 29.124  pairs: 270
  cloudy_sunny_scene_3                auc@5:  3.170  auc@10:  9.917  auc@20: 20.843  pairs: 289
  cloudy_sunny_scene_4                auc@5:  0.484  auc@10:  3.252  auc@20: 10.027  pairs: 454
---
[per-class] (XoFTR aggregiate_scenes equivalent, units: %)
  cloudy_cloudy                       auc@5:  6.567  auc@10: 15.077  auc@20: 26.831
  cloudy_sunny                        auc@5:  2.708  auc@10:  8.528  auc@20: 19.626
---
[overall] (units: %)
  all_class_mean                      auc@5:  4.638  auc@10: 11.803  auc@20: 23.228   # compare with MINIMA paper Table 3 ELoFTR: 2.88 / 7.88 / 17.72
num_matches: 517.81

roadscene

precision@1px: 0.2320
precision@3px: 0.5665
precision@5px: 0.6533

# v12 msyn 逐像素匹配 832

[per-scene] (units: %)
  cloudy_cloudy_scene_1               auc@5:  2.786  auc@10:  7.700  auc@20: 13.965  pairs: 131
  cloudy_cloudy_scene_2               auc@5:  3.761  auc@10: 12.641  auc@20: 26.452  pairs: 197
  cloudy_cloudy_scene_3               auc@5:  7.028  auc@10: 14.011  auc@20: 23.096  pairs: 266
  cloudy_cloudy_scene_4               auc@5:  8.182  auc@10: 13.352  auc@20: 19.816  pairs: 311
  cloudy_cloudy_scene_5               auc@5:  9.194  auc@10: 20.383  auc@20: 34.686  pairs: 180
  cloudy_cloudy_scene_6               auc@5:  0.738  auc@10:  2.393  auc@20:  6.565  pairs: 297
  cloudy_sunny_scene_1                auc@5:  1.007  auc@10:  3.015  auc@20:  9.863  pairs: 195
  cloudy_sunny_scene_2                auc@5:  4.233  auc@10: 12.230  auc@20: 26.251  pairs: 270
  cloudy_sunny_scene_3                auc@5:  3.342  auc@10:  9.785  auc@20: 17.722  pairs: 289
  cloudy_sunny_scene_4                auc@5:  0.401  auc@10:  2.082  auc@20:  7.234  pairs: 454
---
[per-class] (XoFTR aggregiate_scenes equivalent, units: %)
  cloudy_cloudy                       auc@5:  5.281  auc@10: 11.747  auc@20: 20.763
  cloudy_sunny                        auc@5:  2.246  auc@10:  6.778  auc@20: 15.267
---
[overall] (units: %)
  all_class_mean                      auc@5:  3.764  auc@10:  9.262  auc@20: 18.015   # compare with MINIMA paper Table 3 ELoFTR: 2.88 / 7.88 / 17.72
num_matches: 483.98

roadscene

precision@1px: 0.0710
precision@3px: 0.4572
precision@5px: 0.7384

# v13 msyn 832

<!-- [per-scene] (units: %)
  cloudy_cloudy_scene_1               auc@5: 22.356  auc@10: 44.953  auc@20: 63.972  pairs: 131
  cloudy_cloudy_scene_2               auc@5: 16.500  auc@10: 33.762  auc@20: 53.965  pairs: 197
  cloudy_cloudy_scene_3               auc@5: 17.565  auc@10: 34.524  auc@20: 52.588  pairs: 266
  cloudy_cloudy_scene_4               auc@5: 17.857  auc@10: 34.815  auc@20: 53.172  pairs: 311
  cloudy_cloudy_scene_5               auc@5: 19.622  auc@10: 41.067  auc@20: 62.282  pairs: 180
  cloudy_cloudy_scene_6               auc@5:  8.766  auc@10: 19.863  auc@20: 36.589  pairs: 297
  cloudy_sunny_scene_1                auc@5: 16.974  auc@10: 34.755  auc@20: 56.337  pairs: 195
  cloudy_sunny_scene_2                auc@5: 13.450  auc@10: 28.171  auc@20: 48.703  pairs: 270
  cloudy_sunny_scene_3                auc@5: 12.855  auc@10: 27.787  auc@20: 45.099  pairs: 289
  cloudy_sunny_scene_4                auc@5:  4.592  auc@10: 17.457  auc@20: 36.540  pairs: 454
---
[per-class] (XoFTR aggregiate_scenes equivalent, units: %)
  cloudy_cloudy                       auc@5: 17.111  auc@10: 34.831  auc@20: 53.761
  cloudy_sunny                        auc@5: 11.968  auc@10: 27.043  auc@20: 46.670
---
[overall] (units: %)
  all_class_mean                      auc@5: 14.539  auc@10: 30.937  auc@20: 50.216   # compare with MINIMA paper Table 3 ELoFTR: 2.88 / 7.88 / 17.72
num_matches: 467.78 -->

  cloudy_cloudy                       auc@5: 19.326  auc@10: 37.665  auc@20: 56.197
  cloudy_sunny                        auc@5: 11.872  auc@10: 27.257  auc@20: 46.379
---
[overall] (units: %)
  all_class_mean                      auc@5: 15.599  auc@10: 32.461  auc@20: 51.288  

roadscene
<!-- 
precision@1px: 0.0774
precision@3px: 0.4482
precision@5px: 0.7291 -->

precision@1px: 0.0792
precision@3px: 0.4564
precision@5px: 0.7341

# v14 msyn 640

<!-- [per-scene] (units: %)
  cloudy_cloudy_scene_1               auc@5: 28.283  auc@10: 47.900  auc@20: 63.919  pairs: 131
  cloudy_cloudy_scene_2               auc@5: 16.955  auc@10: 34.029  auc@20: 53.919  pairs: 197
  cloudy_cloudy_scene_3               auc@5: 23.681  auc@10: 40.696  auc@20: 56.130  pairs: 266
  cloudy_cloudy_scene_4               auc@5: 17.229  auc@10: 33.408  auc@20: 50.120  pairs: 311
  cloudy_cloudy_scene_5               auc@5: 22.446  auc@10: 46.303  auc@20: 66.980  pairs: 180
  cloudy_cloudy_scene_6               auc@5: 11.419  auc@10: 24.475  auc@20: 41.517  pairs: 297
  cloudy_sunny_scene_1                auc@5: 16.994  auc@10: 37.167  auc@20: 56.553  pairs: 195
  cloudy_sunny_scene_2                auc@5: 12.914  auc@10: 28.034  auc@20: 49.093  pairs: 270
  cloudy_sunny_scene_3                auc@5: 13.610  auc@10: 29.850  auc@20: 46.770  pairs: 289
  cloudy_sunny_scene_4                auc@5:  3.846  auc@10: 15.856  auc@20: 34.685  pairs: 454
---
[per-class] (XoFTR aggregiate_scenes equivalent, units: %)
  cloudy_cloudy                       auc@5: 20.002  auc@10: 37.802  auc@20: 55.431
  cloudy_sunny                        auc@5: 11.841  auc@10: 27.727  auc@20: 46.775
---
[overall] (units: %)
  all_class_mean                      auc@5: 15.922  auc@10: 32.764  auc@20: 51.103   # compare with MINIMA paper Table 3 ELoFTR: 2.88 / 7.88 / 17.72
num_matches: 458.32 -->

[per-class] (XoFTR aggregiate_scenes equivalent, units: %)
  cloudy_cloudy                       auc@5: 18.605  auc@10: 36.957  auc@20: 54.856
  cloudy_sunny                        auc@5: 12.087  auc@10: 26.551  auc@20: 45.586
---
[overall] (units: %)
  all_class_mean                      auc@5: 15.346  auc@10: 31.754  auc@20: 50.221   

roadscene

<!-- precision@1px: 0.0805
precision@3px: 0.4641
precision@5px: 0.7353 -->

precision@1px: 0.0809
precision@3px: 0.4667
precision@5px: 0.7407

# v15 msyn 640 loss优化

[per-class] (XoFTR aggregiate_scenes equivalent, units: %)
  cloudy_cloudy                       auc@5: 18.224  auc@10: 35.581  auc@20: 53.008
  cloudy_sunny                        auc@5: 12.375  auc@10: 27.108  auc@20: 45.229
---
[overall] (units: %)
  all_class_mean                      auc@5: 15.300  auc@10: 31.344  auc@20: 49.119   

roadscene

precision@1px: 0.0815
precision@3px: 0.4679
precision@5px: 0.7401
<!-- precision@1px: 0.0879
precision@3px: 0.4899
precision@5px: 0.7552 -->

# v16 msyn 640 模态嵌入

roadscene

# v17 msyn 640 loss优化 + 模态嵌入

roadscene
