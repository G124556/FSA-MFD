# FSA-MFD: Frequency-domain Style Alignment for Change Detection

> **Frequency-domain style alignment for change detection via multi-dimensional feature decoding**  
> Bo Liu, YongChen Li  
> *Pattern Recognition Letters*, 2026

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Overview

FSA-MFD is a change detection network that addresses two fundamental challenges in remote sensing image analysis:

1. **Style inconsistency** between bi-temporal images caused by illumination, atmospheric conditions, seasonal variations, and sensor differences.
2. **Insufficient feature representation** in single feature decoding paradigms.

### Key Contributions

| Module | Description |
|--------|-------------|
| **FSA** (Frequency-domain Style Alignment) | Learnable Gaussian distribution masks for bidirectional amplitude spectrum exchange, eliminating style differences while preserving semantic information |
| **MDFD** (Multi-Dimensional Feature Decoding) | 4D feature tensor space (original + difference + fusion + frequency-domain features) with dual-path collaborative attention |
| **DMCL** (Dual-Memory Contrastive Learning) | Local (edge-aware batch) + Global (cross-batch prototype) memory banks for enhanced feature discriminability |

### Results

| Dataset | IoU | F1 | OA |
|---------|-----|----|----|
| LEVIR-CD | 85.0±0.3% | 92.3±0.2% | - |
| WHU-CD | 91.5±0.3% | 95.6±0.2% | - |
| CDD (DSIFN-CD) | 61.5±0.4% | 75.8±0.3% | - |

---

## 🏗️ Project Structure

```
FSA-MFD/
├── models/
│   ├── __init__.py
│   ├── fsa.py          # 频域风格对齐模块
│   ├── mdfd.py         # 多维特征解码模块
│   ├── dmcl.py         # 双记忆对比学习模块
│   └── fsamfd.py       # 完整网络架构
├── datasets/
│   ├── __init__.py
│   └── dataset.py      # 数据集加载 (LEVIR-CD, WHU-CD, CDD)
├── utils/
│   ├── __init__.py
│   ├── metrics.py      # F1, IoU, OA 评估指标
│   └── utils.py        # 日志、Checkpoint、可视化工具
├── configs/
│   ├── levir_cd.yaml   # LEVIR-CD 配置
│   ├── whu_cd.yaml     # WHU-CD 配置
│   └── cdd.yaml        # CDD 配置
├── train.py            # 训练脚本
├── test.py             # 测试/推理脚本
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# 克隆仓库
git clone https://github.com/YOUR_USERNAME/FSA-MFD.git
cd FSA-MFD

# 安装依赖
pip install -r requirements.txt
```

### 2. Data Preparation

所有数据集统一组织为以下格式:

```
data/
├── LEVIR-CD/
│   ├── train/
│   │   ├── A/       # 时相1图像 (.png)
│   │   ├── B/       # 时相2图像 (.png)
│   │   └── label/   # 变化标签 (0=不变, 255=变化)
│   ├── val/
│   └── test/
├── WHU-CD/
│   ├── train/  val/  test/
└── CDD/
    ├── train/  val/  test/
```

**数据集下载:**
- **LEVIR-CD**: https://justchenhao.github.io/LEVIR/
- **WHU-CD**: https://gpcv.whu.edu.cn/data/building_dataset.html
- **CDD**: https://ieee-dataport.org/documents/change-detection-dataset

### 3. Training

```bash
# 在 LEVIR-CD 上训练
python train.py --config configs/levir_cd.yaml --gpu 0

# 在 WHU-CD 上训练
python train.py --config configs/whu_cd.yaml --gpu 0

# 在 CDD 上训练
python train.py --config configs/cdd.yaml --gpu 0

# 多 GPU 训练
python train.py --config configs/levir_cd.yaml --gpu 0,1

# 从 checkpoint 恢复训练
python train.py --config configs/levir_cd.yaml --resume outputs/levir_cd/latest.pth
```

### 4. Testing

```bash
# 在测试集上评估
python test.py --config configs/levir_cd.yaml \
               --checkpoint outputs/levir_cd/best_model.pth \
               --save_pred

# 单张图像对推理
python test.py --img_t path/to/t1.png \
               --img_t1 path/to/t2.png \
               --checkpoint outputs/levir_cd/best_model.pth \
               --output result.png
```

---

## ⚙️ Configuration

配置文件位于 `configs/` 目录，支持以下参数:

```yaml
model:
  input_size: 256         # 输入图像尺寸
  pretrained: true        # ImageNet 预训练
  alpha: 0.2              # FSA 低频区域学习范围 (H方向)
  beta: 0.2               # FSA 低频区域学习范围 (W方向)
  bank_size: 1000         # 全局记忆库容量
  temperature: 0.07       # InfoNCE 温度 τ
  contrastive_weight: 0.1 # 对比损失权重 λ

train:
  epochs: 200
  optimizer: adamw
  lr: 1.0e-4
  scheduler: cosine
  amp: true               # 混合精度训练
```

---

## 🔬 Method Details

### FSA Module

通过 2D FFT 将特征分解为幅度谱 (风格信息) 和相位谱 (语义信息)。可学习的 Sigmoid 掩码 $M_i^t, M_i^{t+1}$ 控制双向幅度谱交换:

$$\hat{E}_i^t = \mathcal{F}^{-1}(M_i^t \odot A_i^{t+1} + (1-M_i^t) \odot A_i^t, P_i^t)$$

掩码学习区域限制在低频区域 $\{-\alpha H_i:\alpha H_i, -\beta W_i:\beta W_i\}$，$\alpha=\beta=0.2$。

### MDFD Module

构建四维特征张量空间并采用双路协作注意力:
- **多尺度自注意力路径**: Query 互换策略处理差异特征和融合特征
- **变化感知交叉注意力路径**: 变化感知掩码引导双时相原始特征交互

### DMCL Module

$$\mathcal{L}_{contrastive} = \mathcal{L}_{local} + 0.5\mathcal{L}_{global}$$

$$\mathcal{L}_{total} = \mathcal{L}_{CE} + 0.1\mathcal{L}_{contrastive}$$

---

## 📊 Ablation Study

| Configuration | LEVIR IoU | WHU IoU | CDD IoU |
|---------------|-----------|---------|---------|
| ResNet+MDFD (baseline) | 82.6 | 88.2 | 55.3 |
| +FSA | 83.9 | 89.8 | 57.8 |
| +DMCL | 83.2 | 89.1 | 56.9 |
| **Full Model** | **84.7** | **91.2** | **61.5** |

---

## 📝 Citation

如果本项目对您的研究有帮助，请引用:

```bibtex
@article{liu2026fsamfd,
  title={Frequency-domain style alignment for change detection via multi-dimensional feature decoding},
  author={Liu, Bo and Li, YongChen},
  journal={Pattern Recognition Letters},
  year={2026},
  publisher={Elsevier}
}
```

---

## 📄 License

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE)。

---

## 🙏 Acknowledgements

- 编码器使用 [ResNet50](https://arxiv.org/abs/1512.03385) (ImageNet 预训练)
- 数据集: LEVIR-CD, WHU-CD, CDD
- 感谢 [BIT](https://github.com/justchenhao/BIT_CD)、[SNUNet](https://github.com/RSCD-Lab/Siam-NestedUNet) 等开源工作的启发
