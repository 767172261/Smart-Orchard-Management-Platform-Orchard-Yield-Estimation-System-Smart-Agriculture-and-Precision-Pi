智慧果园管理平台 果园产量估算系统 智能农业与精准采摘 柑橘果实检测数据集 柑桔果实检测数据集在深度学习领域的应用
@[toc]

农业领域-柑橘果实检测数据集
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e31dd963288141cf940c56a0977913fb.png)


### **柑橘果实检测数据集（YOLO格式）**

| **项目** | **内容** |
|----------|----------|
| **数据集名称** | 柑橘果实检测数据集（Citrus Fruit Detection Dataset） |
| **任务类型** | 目标检测（单类别检测） |
| **适用模型** | YOLOv5、YOLOv8 等系列模型（支持 YOLO 格式） |
| **图像总数** | 4,855 张（2913 + 971 + 971） |
| **类别数量** | 1 类 |
| **类别标签** | `'orange'`（泛指柑橘类果实，如橙子、橘子等） |
| **标注格式** | YOLO `.txt` 格式（每图对应一个标签文件）<br>格式：`class_id center_x center_y w h`（归一化坐标） |
| **数据划分** | 已按 `train : val : test = 3:1:1` 比例划分 |

---![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a9a2ed46f7974473800f69abe62b3537.png)


### **数据集划分详情**

| **划分类型** | **图像数量** | **占比** | **建议路径** |
|--------------|--------------|----------|----------------|
| **训练集 (train)** | 2,913 张 | ~60% | `images/train/` |
| **验证集 (val)** | 971 张 | ~20% | `images/val/` |
| **测试集 (test)** | 971 张 | ~20% | `images/test/` |

> ✅ 所有标签文件保存在对应 `labels/train/`、`labels/val/`、`labels/test/` 目录下

---![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a65b8738b42547509a51c5de4fcd78f0.png)


### **数据集特点**

- ✅ **真实果园场景**：包含不同光照、遮挡、密集果实、枝叶干扰等复杂环境  
- ✅ **高多样性**：涵盖不同品种、成熟度、背景（绿叶、土壤、天空）的柑橘图像  
- ✅ **YOLO标准格式**：可直接用于 YOLO 系列模型训练，无需格式转换  
- ✅ **大规模数据集**：近 5,000 张图像，适合训练鲁棒性强的检测模型  

---
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/acae06dd8f0c4d2b9b4f790804fac630.png)

### **YOLO `data.yaml` 配置文件模板**

```yaml
# data.yaml
train: ./images/train
val: ./images/val
test: ./images/test

# 类别数量
nc: 1

# 类别名称
names: ['orange']
```

> ✅ 将此文件保存在项目根目录，用于 YOLO 训练命令调用。

---![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/12612671624a46e9ae659d83ff114448.png)


### **推荐训练命令（YOLOv8 示例）**

```bash
yolo train \
    model=yolov8n.pt \
    data=data.yaml \
    epochs=100 \
    imgsz=640 \
    batch=32 \
    name=citrus_orange_detection
```

> 📌 建议：使用 `yolov8s` 或 `yolov8m` 提升小果实检测精度

---

## ✅ 主要应用领域

| **应用领域** | **具体用途说明** |
|--------------|------------------|
| **1. 智能农业与精准采摘** | 用于农业机器人或无人机识别柑橘位置，实现自动采摘或产量预估。 |
| **2. 果园产量估算系统** | 通过图像中果实数量与分布，结合面积统计，预测果园总产量。 |
| **3. 智慧果园管理平台** | 集成到果园监控系统中，实现果实生长状态动态监测与预警。 |
| **4. 农业无人机巡检** | 搭载摄像头的无人机自动飞行拍摄，AI识别果实成熟度与分布。 |
| **5. 农产品品质控制** | 在分拣线上检测果实大小、数量、缺陷（可扩展标注），实现自动化分选。 |
| **6. 农业科研与教学** | 用于农业AI算法研究、计算机视觉课程设计、毕业项目等。 |
| **7. 数字农业与数字孪生** | 构建果园数字模型，模拟生长过程，优化种植策略。 |

---

### ✅ 潜在扩展方向

- **多类别扩展**：增加 `'lemon'`、`'lime'`、`'tangerine'` 等细分品类  
- **实例分割**：将 `.txt` 框转为掩码，实现果实轮廓精确提取（可用于体积估算）  
- **成熟度分类**：结合颜色与大小，判断果实是否可采摘  
- **3D定位**：结合双目相机或深度图，实现果实空间定位  
- ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/541e55d2536e4256a96834d43f8125a4.png)

 YOLOv8 训练与部署指南，包含：

- ✅ 数据集结构说明
- ✅ 从 **CUDA 驱动 → Anaconda → Python 虚拟环境 → 安装依赖**
- ✅ `data.yaml` 配置文件
- ✅ 使用 **YOLOv8 官方预训练模型** 的 **训练、推理代码**

以下文字及代码仅供参考学习使用。
---

## ✅ 一、数据集信息

| 类别 | 数量 | 图片数量 |
|------|------|----------|
| 类别名称 | `orange` | 1类 |
| 训练集（train） | `images/train/`, `labels/train/` | 2913 张 |
| 验证集（val）   | `images/val/`,   `labels/val/`   | 971 张 |
| 测试集（test）  | `images/test/`,  `labels/test/`  | 971 张 |

> 所有标签为 YOLO 格式 `.txt` 文件，每行：
```
0 center_x center_y width height
```
（`0` 表示类别 ID）

---

## ✅ 二、目录结构

确保你的项目结构如下：

```
citrus_detection/
├── images/
│   ├── train/     # 2913 张
│   ├── val/       # 971 张
│   └── test/      # 971 张
├── labels/
│   ├── train/     # 2913 个 .txt
│   ├── val/       # 971 个 .txt
│   └── test/      # 971 个 .txt
├── data.yaml      # 数据配置
└── train.py       # 训练脚本
```

---

## ✅ 三、环境搭建全流程

### 1. 确认 CUDA 驱动支持

打开命令提示符或终端，运行：
```bash
nvidia-smi
```
查看顶部显示的 **CUDA Version**（如 12.4），确保 ≥ 11.8

> 若无输出，请安装 NVIDIA 显卡驱动：[https://www.nvidia.cn/Download/index.aspx](https://www.nvidia.cn/Download/index.aspx)

---

### 2. 安装 Anaconda

- 下载地址：[https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)
- 安装完成后打开 **Anaconda Prompt**（Windows）或终端（Mac/Linux）

---

### 3. 创建 Python 虚拟环境并安装依赖

```bash
# 创建虚拟环境
conda create -n citrus_det python=3.9
conda activate citrus_det

# 安装 PyTorch + CUDA（以 CUDA 11.8 为例）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 或使用 conda（推荐）
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 安装 YOLOv8 所需库
pip install ultralytics opencv-python numpy matplotlib tqdm pillow

# 验证 GPU 是否可用
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

✅ 正确输出应为：
```
2.3.0+cu118
True
```

---

## ✅ 四、创建 `data.yaml` 配置文件

在项目根目录创建 `data.yaml`：

```yaml
# data.yaml
train: ./images/train
val: ./images/val
test: ./images/test

# 类别数量
nc: 1

# 类别名称
names:
  - orange
```

> ⚠️ 注意：
- 路径建议使用相对路径或确保脚本运行位置正确
- 所有 `.txt` 文件中的类别 ID 必须是 `0`（因为只有 1 类）

---

## ✅ 五、训练代码（调用官方 YOLOv8 预训练模型）

```python
# train.py
from ultralytics import YOLO

# 加载官方预训练模型（推荐使用 yolov8n 或 yolov8s）
model = YOLO('yolov8n.pt')  # 可替换为 yolov8s.pt 提升精度

# 开始训练
results = model.train(
    data='data.yaml',           # 数据集配置文件
    epochs=150,                 # 柑橘数据建议训练 100~200 轮
    imgsz=640,                  # 输入尺寸（可尝试 800 提升小果实检测）
    batch=16,                   # 根据显存调整（24GB 显卡可用 16~32）
    name='citrus_yolov8n_150e', # 实验名称
    device=0,                   # 使用 GPU 0，多卡写 [0,1]
    workers=8,                  # 数据加载线程
    optimizer='AdamW',          # 更适合小目标检测
    lr0=0.001,                  # 初始学习率
    patience=30,                # 早停：30轮无提升则停止
    project='runs/train',       # 结果保存路径
    cache=False,                # 可设 True 加快训练（内存足够）
    close_mosaic=10             # 最后10轮关闭 Mosaic 增强
)

print("✅ 柑橘检测模型训练完成！")
```

### 📌 模型选择建议：

| 模型 | 显存需求 | 速度 | 精度 | 推荐场景 |
|------|----------|------|------|----------|
| `yolov8n.pt` | ~4GB | ⚡⚡⚡ | ★★★☆ | 快速部署、边缘设备 |
| `yolov8s.pt` | ~6GB | ⚡⚡ | ★★★★ | **推荐（平衡）** |
| `yolov8m.pt` | ~8GB | ⚡ | ★★★★★ | 高精度检测（小果实） |

> 💡 柑橘果园中果实较小，建议使用 `yolov8s` 或 `yolov8m`

---

## ✅ 六、推理代码（检测新图像）

```python
# predict.py
from ultralytics import YOLO

# 加载训练好的最佳模型
model = YOLO('runs/train/citrus_yolov8n_150e/weights/best.pt')

# 单张图片检测
results = model.predict(
    source='citrus_orchard.jpg',  # 输入图片路径
    conf=0.25,                   # 置信度阈值（农业检测建议 0.2~0.4）
    iou=0.45,
    save=True,                   # 保存带框图像
    project='runs/predict',
    name='citrus_result',
    show=False
)

# 打印结果
for result in results:
    boxes = result.boxes
    for box in boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        label = model.names[cls_id]
        print(f"🍊 检测到: {label}, 置信度: {conf:.3f}")
```

### 批量推理测试集：
```python
results = model.predict(source='images/test/', save=True)
```

---

## ✅ 七、评估代码（在测试集上评估性能）

```python
# evaluate.py
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('runs/train/citrus_yolov8n_150e/weights/best.pt')

# 在 test 集上评估
metrics = model.val(
    data='data.yaml',
    split='test',
    batch=16,
    imgsz=640,
    save_json=True,
    project='runs/val',
    name='citrus_metrics'
)

# 输出关键指标
print(f"📊 mAP@50: {metrics.box.map50:.4f}")
print(f"📊 mAP@50:95: {metrics.box.map:.4f}")
print(f"📊 Precision: {metrics.box.p:.4f}")
print(f"📊 Recall: {metrics.box.r:.4f}")
```

### 预期性能（使用 yolov8s）：
```
mAP@50: 0.85 ~ 0.92
mAP@50:95: 0.50 ~ 0.60
```
> ✅ 达到果园自动化采摘、产量预估可用水平

---

## ✅ 八、训练技巧（提升柑橘检测精度）

### 1. 增强小目标检测能力
```python
model.train(
    ...
    imgsz=800,                  # 提高分辨率利于小果实检测
    mosaic=1.0,
    copy_paste=0.3,             # 随机复制粘贴果实（适合密集果园）
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=20,
    translate=0.2,
    scale=0.5
)
```

### 2. 使用更大的模型
```python
model = YOLO('yolov8m.pt')
```

### 3. 继续训练（微调）
```python
model = YOLO('runs/train/citrus_yolov8n_150e/weights/best.pt')
model.train(resume=True, epochs=200)
```

---

## ✅ 九、导出模型（用于部署）

```python
# 导出为 ONNX（通用部署）
model.export(format='onnx', dynamic=True, simplify=True)

# 导出为 TensorRT（NVIDIA 加速）
model.export(format='engine', device=0)

# 导出为 CoreML（iOS）
model.export(format='coreml')
```
生成 `best.onnx`，可用于无人机、机器人、Flask API 等。

---


