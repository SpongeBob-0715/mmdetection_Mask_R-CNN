# MMDetection 环境搭建指南

本项目旨在指导用户使用 Conda 和 OpenMMLab 工具链快速搭建 MMDetection 环境。

## 环境配置步骤

### 步骤 1：创建并激活 Conda 虚拟环境

```bash
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

### 步骤 2：安装 PyTorch（基于官方说明）

如果使用 GPU 平台：

```bash
conda install pytorch torchvision -c pytorch
```

> ⚠️ 请根据你的实际设备和 CUDA 版本选择合适的 PyTorch 安装方式。可参考 [PyTorch 官方安装指南](https://pytorch.org/get-started/locally/)。

### 步骤 3：使用 MIM 安装 MMEngine 和 MMCV

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```

### 步骤 4：安装 MMDetection

```bash
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
```

说明：

- `-v`：启用详细输出，方便调试安装过程。
- `-e`：以可编辑模式安装，支持本地代码修改后无需重新安装。

## 模型训练

### 训练 Mask R-CNN 模型

```bash
cd mmdetection
bash run_train_mask_rcnn.sh
```

### 训练 Sparse R-CNN 模型

```bash
cd mmdetection
bash run_train_sparse_rcnn.sh
```

## 参考链接

- [OpenMMLab 官方文档](https://openmmlab.com/)
- [MMDetection GitHub 仓库](https://github.com/open-mmlab/mmdetection)
