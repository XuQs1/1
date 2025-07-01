# TCGA数据库数据探索与多语言编程实践
### 学号：2351909 姓名：徐其升

## 一、目标概述
本作业基于TCGA数据库的真实PCPG癌症（嗜铬细胞瘤和副神经节瘤）研究数据，通过设计创新性编程练习，深化Shell、Python与R语言的基础及应用编程技能。核心在于技术实践，可以用于全方位考查运用三种编程语言解决数据处理挑战的能力。


## 二、作业详情

### （一）Shell编程练习
#### 题目1：创建数据存储与输出目录
- **目的**：规范项目文件结构，为后续数据处理做准备
- **知识点**：`mkdir`命令、`-p`参数（递归创建目录）
```bash
mkdir -p 'data'
mkdir -p 'py_output'
mkdir -p 'r_output'
echo "文件夹已创建"
```
- **预期结果**：成功创建`data`、`py_output`、`r_output`目录。
![题目1运行结果](/imgs/2025-06-28/4rf7Ul9VYcIVHWoB.png)

#### 题目2：移动原始数据至指定目录并重命名
- **目的**：整理数据文件，统一命名规范
- **知识点**：`mv`命令、批量文件操作
```bash
mv "TCGA.PCPG.sampleMap_HiSeqV2_PANCAN.gz" "data"
mv "survival_PCPG_survival.txt" "data"
echo "文件已移动至data文件夹"
```
- **预期结果**：原始数据文件被移动至`data`目录。

#### 题目3：解压缩RNA测序数据
- **目的**：获取可处理的原始数据文件
- **知识点**：`gzip`命令、`-d`参数（解压缩）
```bash
cd ./data
gzip -d TCGA.PCPG.sampleMap_HiSeqV2_PANCAN.gz
echo "文件‘TCGA.PCPG.sampleMap_HiSeqV2_PANCAN.gz’已被解压缩"
```
- **预期结果**：压缩文件解压缩为`TCGA.PCPG.sampleMap_HiSeqV2_PANCAN`。

#### 题目4：重命名数据文件并按列排序
- **目的**：规范文件名，便于后续读取
- **知识点**：`mv`命令、`sort`命令（`-k`参数指定排序列）
```bash
mv "TCGA.PCPG.sampleMap_HiSeqV2_PANCAN" "RNAseq.txt"
mv "survival_PCPG_survival.txt" "surival.txt"
sort -k1,1r -k2,2r surival.txt
```
- **预期结果**：文件重命名为`RNAseq.txt`和`surival.txt`，生存数据按首列降序排序。
- ![输入图片说明](/imgs/2025-06-28/7UFXbfbUJNeigS8O.png)

#### 题目5：批量重命名目录下所有文件，并添加前缀:data_
- **目的**：统一文件前缀，便于管理
- **知识点**：`for`循环、字符串拼接
```bash
for file in *; do 
    mv "$file" "data_$file"
done
```
- **预期结果**：`data`目录下所有文件被添加前缀`data_`。
![输入图片说明](/imgs/2025-06-28/BXg12HSBqFeNzkHV.png)

### （二）Python编程练习
#### 题目1：读取RNAseq与生存数据并转置
- **目的**：加载原始数据并转换为适合分析的格式
- **知识点**：`pd.read_csv`读取表格数据、`T`属性转置数据框
```python
import pandas as pd
file1 = '.\data\data_RNAseq.txt'
file2 = '.\data\data_surival.txt'
RNA_dataframe = pd.read_csv(file1, sep='\t', index_col=0)
RNA_dataframe = RNA_dataframe.T
survival= pd.read_csv(file2, sep='\t', index_col=0)
survival = survival_dataframe.dropna(
                     thresh=len(survival_dataframe)*0.8，axis=1)
```
- **预期结果**：RNAseq数据转置为样本行格式，生存数据过滤缺失值超过20%的列。

#### 题目2：匹配RNAseq与生存数据的样本并合并
- **目的**：确保分析基于相同样本集，整合表达与临床数据
- **知识点**：`index.intersection`获取公共样本、`pd.concat`合并数据框
```python
common_samples = RNA_dataframe.index.intersection(survival.index)
RNA_dataframe = RNA_dataframe.loc[common_samples]
survival = survival_dataframe.loc[common_samples]
merged_dataframe = pd.concat([RNA_dataframe, survival], axis=1)
print("合并后数据前5行：")
print(merged_dataframe.head())
```
- **预期结果**：输出合并后数据的前5行，包含基因表达与临床生存信息。
![输入图片说明](/imgs/2025-06-28/qEIxZWvtzbzy1Qp3.png)

#### 题目3：计算基因表达的均值、中位数和标准差
- **目的**：探索基因表达的整体分布特征
- **知识点**：`mean()`、`median()`、`std()`方法
```python
mean_e = RNA_dataframe.mean()
median_e = RNA_dataframe.median()
std_e = RNA_dataframe.std()
print("基因表达均值前5项：")
print(mean_e.head())
print("基因表达中位数前5项：")
print(median_e.head())
print("基因表达标准差前5项：")
print(std_e.head())
```
- **预期结果**：输出基因表达的均值、中位数和标准差，反映数据集中趋势与离散程度。
![输入图片说明](https://raw.githubusercontent.com/XuQs1/1/master/imgs/2025-06-28/kMm5Brnq5BBOX3fp.png)

#### 题目4：筛选在多数样本中高表达的基因
- **目的**：聚焦于具有生物学意义的活跃基因
- **知识点**：布尔索引、`sum()`计数、排序
```python
high_expression = RNA_dataframe.columns[(RNA_dataframe > 0.2).sum() > 187*0.8]
high_expression = high_expression.tolist()
high_expression_sorted = sorted(high_expression, 
									key=lambda gene: mean_e[gene],
									reverse=True)
print("高表达基因数量：", len(high_expression))
print("前5个高表达基因：", end='')
for gene in high_expression[:5]:
    print(f"{gene} ", end='')
```
- **预期结果**：输出高表达基因数量及前5个基因名称，基于在80%以上样本中表达值>0.2的标准。
![输入图片说明](https://raw.githubusercontent.com/XuQs1/1/master/imgs/2025-06-28/Fqp6zsrWtoaTCyNU.png)

#### 题目5：计算高表达基因与生存时间的相关性
- **目的**：探索基因表达与临床结局的潜在关联
- **知识点**：`corr()`方法、排序筛选强相关基因
```python
survival_time = 'DFI.time'
results = []
for gene in high_expression_genes_sorted:
    if gene in merged_dataframe.columns:
        correlation = merged_dataframe[[gene, survival_time]]
										      .corr().iloc[0, 1]
        correlation_results.append((gene, correlation))
correlation_df = pd.DataFrame(correlation_results,columns=['基因', '相关系数'])
correlation_df['相关系数绝对值'] = correlation_df['相关系数'].abs()
correlation_df = correlation_df.sort_values(by='相关系数绝对值', 
				ascending=False).drop(columns=['相关系数绝对值'])
print("与DFI.time相关性最强的前10个基因：")
print(correlation_df.head(10))
```
- **预期结果**：输出与DFI生存时间相关性最强的前10个基因及其相关系数，绝对值越大关联越强。
![输入图片说明](https://raw.githubusercontent.com/XuQs1/1/master/imgs/2025-06-28/6sZvJsKLM33I4vqa.png)

#### 题目6：绘制高相关基因与生存时间的散点图
- **目的**：直观展示基因表达与生存时间的关系模式
- **知识点**：`matplotlib.pyplot.scatter`绘图、子图布局
```python
top_genes = correlation_df['基因'].head(5)
plt.figure(figsize=(15, 5))
for i, gene in enumerate(top_genes):
    plt.subplot(1, 5, i + 1)
    plt.scatter(merged_dataframe[gene],
				merged_dataframe[survival_time], alpha=0.5)
    plt.title(f'{gene}与{survival_time}')
    plt.xlabel(gene)
    plt.ylabel(survival_time)
plt.tight_layout()
plt.show()
plt.savefig('.\py_output\基因与生存时间相关性散点图.png')
```
- **预期结果**：生成5个子图，每个子图展示一个基因与生存时间的散点分布，直观反映正相关或负相关趋势。（所用数据的相关性实在太差）
![输入图片说明](https://raw.githubusercontent.com/XuQs1/1/master/imgs/2025-06-28/eiKXq73TJFDYdWrB.png)

### （三）R语言编程练习
#### 知识点1：数据读取与转置
- **题目**：读取TCGA RNA测序与生存数据并转置
- **目的**：将基因表达数据转换为样本行格式，便于后续分析
- **知识点**：`read.table`函数、`t()`转置函数
```r
file1 <- "./data/data_RNAseq.txt"
file2 <- "./data/data_surival.txt"
rna_data <- read.table(file1, header=TRUE,sep="\t",row.names=1, check.names=FALSE)
surv_data <- read.table(file2, header=TRUE, sep="\t", row.names=1, check.names=FALSE)
rna_data <- t(rna_data)
rna_data <- as.data.frame(rna_data)
```
- **预期结果**：成功读取数据并转置，样本作为行名，基因作为列名。
![输入图片说明](https://raw.githubusercontent.com/XuQs1/1/master/imgs/2025-06-28/LFt2erNnbzQYdhmn.png)

#### 知识点2：样本交集匹配
- **题目**：匹配RNAseq与生存数据的样本ID
- **目的**：确保分析样本一致性，避免数据错位
- **知识点**：`intersect`函数、索引筛选
```r
common_samples <- intersect(rownames(rna_data), rownames(surv_data))
rna_data <- rna_data[common_samples, ]
surv_data <- surv_data[common_samples, ]
cat("成功匹配", length(common_samples), "个样本\n")
```
- **预期结果**：输出匹配的样本数量，确保后续分析基于相同样本集。
![输入图片说明](https://raw.githubusercontent.com/XuQs1/1/master/imgs/2025-06-28/x4YxaJyl2eh140WK.png)

#### 知识点3：QQ图正态性检验
- **题目**：随机基因表达分布的QQ图分析
- **目的**：检验基因表达值是否符合正态分布
- **知识点**：`ggplot2`绘图、`stat_qq`函数、`pivot_longer`数据长格式转换
```r
library(ggplot2)
library(tidyr)
rna_data$Sample <- rownames(rna_data)
set.seed(2351909)
all_genes <- colnames(rna_data)[!colnames(rna_data) %in% "Sample"]
random_genes <- sample(all_genes, 5)
rna_long <- rna_data %>% pivot_longer(cols = all_of(random_genes), names_to="Gene", values_to="Expression")
qq_plot <- ggplot(rna_long, aes(sample=Expression, color=Gene)) + 
  stat_qq() + facet_wrap(~Gene, scales="free") + 
  theme_minimal() + theme(legend.position="none") +
  labs(title="随机选择基因的QQ图分布", x="观测分位数", y="理论分位数")
print(qq_plot)
```
- **预期结果**：生成QQ图，若点分布接近对角线则表明数据近似正态分布。
![输入图片说明](https://raw.githubusercontent.com/XuQs1/1/master/imgs/2025-06-28/NHLVUFnYgssTZPIL.png)

#### 知识点4：主成分分析（PCA）
- **题目**：基因表达数据的PCA降维分析
- **目的**：揭示数据主要变异来源，识别样本分组特征
- **知识点**：`prcomp`函数、方差贡献率计算、二维可视化
```r
col_variances <- apply(rna_data_numeric, 2, var)
near_zero_var <- which(col_variances < 1e-10)
if(length(near_zero_var) > 0) {
  rna_data_filtered <- rna_data_numeric[,-near_zero_var]
} else {
  rna_data_filtered <- rna_data_numeric
}
rna_data_std <- scale(rna_data_filtered)
pca_result <- prcomp(rna_data_std, scale. = FALSE)
pca_data <- data.frame(PC1=pca_result$x[,1], PC2=pca_result$x[,2], Sample=rownames(pca_result$x))
pca_plot <- ggplot(pca_data, aes(x=PC1, y=PC2, label=Sample)) +
  geom_point(size=3, alpha=0.7) + 
  labs(title="基因表达数据的主成分分析",
       x=paste("PC1(方差解释率:", round(summary(pca_result)$importance[2,1]*100,1), "%)"),
       y=paste("PC2(方差解释率:", round(summary(pca_result)$importance[2,2]*100,1), "%)")) +
  theme_minimal() + theme(axis.text=element_text(size=8))
print(pca_plot)
```
- **预期结果**：PCA散点图展示样本在主成分空间的分布，横轴和纵轴标注方差解释率。
![输入图片说明](https://raw.githubusercontent.com/XuQs1/1/master/imgs/2025-06-28/fwN60klUXTx1ARRj.png)

#### 知识点5：热图可视化高变异基因
- **题目**：前10高变异基因的表达热图绘制
- **目的**：直观呈现样本间基因表达的相似性与差异性
- **知识点**：`scale`标准化、`ggplot2`热图绘制、颜色梯度映射
```r
normalized_rna_data <- scale(rna_data)
gene_var <- apply(normalized_rna_data, 2, var)
top10_genes <- names(sort(gene_var, decreasing=TRUE)[1:10])
normalized_top10 <- normalized_rna_data[, top10_genes]
melted_rna_data <- data.frame(
  Gene=rep(row.names(normalized_top10), each=ncol(normalized_top10)),
  Sample=rep(colnames(normalized_top10), times=nrow(normalized_top10)),
  Value=as.vector(t(normalized_top10))
)
heatmap_plot <- ggplot(melted_rna_data, aes(x=Sample, y=Gene, fill=Value)) +
  geom_tile() +
  scale_fill_gradient2(low="blue", high="red", mid="white", midpoint=0) +
  labs(title="前10高变异基因表达热图", x="基因", y="样本") +
  theme_minimal() + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank())
print(heatmap_plot)
```
- **预期结果**：热图通过颜色深浅展示基因表达量，红色代表高表达，蓝色代表低表达。
![输入图片说明](https://raw.githubusercontent.com/XuQs1/1/master/imgs/2025-06-28/W2ck7aEgIkAMxkAK.png)

#### 知识点6：生存时间差异性检验
- **题目**：OS与DFI生存时间的KS检验与QQ图比较
- **目的**：验证临床终点（总体生存时间与无病间期）的分布一致性
- **知识点**：`ks.test`函数、`qqplot`绘制分位数比较图
```r
ks_result <- ks.test(surv_data$OS.time, surv_data$DFI.time)
cat("OS与DFI生存时间的KS检验结果:\n")
print(ks_result)
qqplot(surv_data$OS.time, surv_data$DFI.time, main="OS.time vs DFI.time的QQ图比较",
       xlab="OS时间(天)", ylab="DFI时间(天)")
abline(0, 1, col="red", lty=2)
```
- **预期结果**：KS检验输出p值（若p<0.05则拒绝分布一致的原假设），QQ图展示两变量分位数对应关系。
![输入图片说明](https://raw.githubusercontent.com/XuQs1/1/master/imgs/2025-06-28/xzn2vxJh0NMKb0Lf.png)
![输入图片说明](https://raw.githubusercontent.com/XuQs1/1/master/imgs/2025-06-28/a9bta0OBTEkDIgWp.png)
## 三、语言差异与互补优势分析
### （一）Shell与脚本语言的定位差异
Shell语言更适合文件系统操作（如目录创建、文件移动、批量重命名）和流程自动化，适合数据预处理阶段的文件管理，但缺乏复杂数据处理与统计分析能力。
### （二）Python与R语言的优势互补
**Python**：
 优势在于数据清洗与预处理速度快，如 for 循环结构在大规模数据处理方面相较于R语言运行速度上明显要快，同时还可以跨领域应用（可结合机器学习库，但本报告中没有涉及）。
 **R语言**：
优势在于统计分析（如PCA、KS检验）与可视化（ggplot2专业绘图），同时在生物信息学领域具有更完善的生态（如edgeR、limma包，但本报告中没有涉及）。

结合三种语言的优势，本报告将Python/R语言用于进行数据处理与分析，Shell语言用于前置步骤批量处理原始文件，三者结合形成“文件管理→数据处理→统计分析”的完整工作流。
## 四、结果总结

1. **数据处理流程**：通过Shell完成数据文件管理，Python与R语言分别实现数据整合、统计分析与可视化，形成完整的TCGA数据探索工作流；
2. **技术覆盖**：Shell实现5个文件管理知识点，Python与R语言各完成10个数据处理与分析知识点；
3. **可视化成果**：生成QQ图、PCA散点图、热图及相关性散点图，直观展示基因表达特征与生存数据关联；
4.  **不足之处**：所选用的数据集比较简单，且临床数据的相关性不强，简单的数据处理手段可能无法得到更深入且全面的信息。
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTA0Mjg0NzcwOSwtMTQ3MjEyNjk4NiwxMz
U1NTcxNTY2LDM0NTcxNzI1Nyw3MjU0NDcxMDksMjIwNjcxNjk3
XX0=
-->