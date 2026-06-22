# DHC 复现说明（BIBM baseline 存档版）

本仓库基于 [DHC 官方代码](https://github.com/xmed-lab/DHC) 整理，已在本地完成 Synapse 20% labeled 设置下的推理与可视化复现。

## 1. 环境

```bash
pip install torch torchvision medpy SimpleITK surface-distance h5py tqdm tensorboard matplotlib
source env.sh   # export PYTHONPATH=$(pwd)/code:$PYTHONPATH
```

推荐：Python 3.8+，CUDA GPU。

## 2. 数据准备

1. 下载 DHC 官方预处理 Synapse 数据：  
   https://hkustconnect-my.sharepoint.com/:f:/g/personal/hwanggr_connect_ust_hk/EmOL8Cn-GTBJtOjg6zNgsPABdF6TgoWtRac4FwGqfFxLvQ?e=a1xaDJ

2. 解压到项目根目录，目录结构应为：

```text
./synapse_data/
├── npy/
│   ├── 0001_image.npy
│   ├── 0001_label.npy
│   └── ...
└── splits/
    ├── train.txt
    ├── eval.txt
    ├── test.txt
    ├── labeled_20p.txt
    └── unlabeled_20p.txt
```

本仓库 `splits/` 为备份划分文件，可与 `synapse_data/splits/` 对照使用。

## 3. 下载预训练权重

官方 20% Synapse 权重：  
https://drive.google.com/drive/folders/1aUU2KvNUVAYLo4qqvT5JBd7hHzo_4K1Q?usp=drive_link

下载 3-fold DHC 权重后，放置为：

```text
logs/Task_synapse_20p/dhc/fold1/ckpts/best_model.pth
logs/Task_synapse_20p/dhc/fold2/ckpts/best_model.pth
logs/Task_synapse_20p/dhc/fold3/ckpts/best_model.pth
```

`best_model.pth` 内含 `A`、`B` 两个子网络 state_dict。

或放到 `weights/fold{1,2,3}/ckpts/best_model.pth`，再执行：

```bash
bash scripts/setup_for_inference.sh
```

## 4. 官方流程：测试 & 评估

```bash
source env.sh

# 单 fold 推理
python code/test.py --task synapse --exp Task_synapse_20p/dhc/fold1 -g 0 --cps AB

# 3-fold 汇总评估（期望 Final Avg Dice ≈ 47%）
python code/evaluate_Ntimes_npy.py --task synapse --exp Task_synapse_20p/dhc --folds 3 --cps AB
```

完整训练（可选）：

```bash
bash train3times_seeds_20p.sh -c 0 -t synapse -m dhc -e '' -l 3e-2 -w 0.1
```

## 5. 可视化推理（本仓库扩展）

使用 `inference_Synapse_CPS.py` + `networks/` + `utils/test_util_vnet_AB.py`。

**重要**：DHC 权重必须用 `--data_format npy`，不能用 h5 流程。

```bash
# 从 best_model.pth 导出 A/B 权重
python scripts/export_ab_weights.py --ckpt logs/Task_synapse_20p/dhc/fold1/ckpts/best_model.pth --out_dir tmp_ckpts

# 全器官、全切片可视化（6 cases × 240 slices × 3 png）
python inference_Synapse_CPS.py \
  --data_format npy \
  --npy_dir ./synapse_data/npy \
  --model_path_A ./tmp_ckpts/dhc_best_A.pth \
  --model_path_B ./tmp_ckpts/dhc_best_B.pth \
  --export_vis 1 \
  --vis_dir ./vis_synapse \
  --model_tag DHC_fold1_npy_allorgans \
  --planes axial,coronal \
  --target_classes 1,2,3,4,5,6,7,8,9,10,11,12,13 \
  --save_all_slices 1
```

可视化默认 test cases：`0004, 0007, 0010, 0033, 0035, 0036`。  
DHC 官方 test set：`0008, 0027, 0029, 0033, 0037, 0039`（用 `--cases` 指定）。

## 6. 已验证指标

| 流程 | 测试集 | 平均 Dice |
|------|--------|-----------|
| `code/test.py` 官方 | 0008,0027,0029,0033,0037,0039 | **47.03%** |
| `inference_Synapse_CPS.py` | 0004,0007,0010,0033,0035,0036 | **44.07%** |

## 7. 仓库未包含的大文件

以下需自行下载，已在 `.gitignore` 中排除：

- `synapse_data/`（~GB 级 npy）
- `weights/`、`logs/`（checkpoint）
- `vis_synapse/`（可视化 PNG 输出）
