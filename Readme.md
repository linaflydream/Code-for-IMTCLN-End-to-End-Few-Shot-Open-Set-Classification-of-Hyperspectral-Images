# IMTCLN

## Incremental Multitask Contrastive Learning Network (IMTCLN) End-to-End Few-Shot Open-Set Classification of Hyperspectral Images

This repository contains the official implementation of the paper:

> **Incremental Multitask Contrastive Learning Network for End-to-End Few-Shot Open-Set Classification of Hyperspectral Images**  
> *Na Li, Xiaopeng Song, Wenxiang Zhu, Yongxu Liu, Chuang Li, Weitao Zhang, Yinghui Quan*  
> IEEE Transactions on Geoscience and Remote Sensing, 2025  
> [DOI: 10.1109/TGRS.2025.3588540](https://doi.org/10.1109/TGRS.2025.3588540)



## 📦 Installation

### Requirements

- **Python** 3.11

- **PyTorch**: 2.0.1+ (with GPU acceleration support)

- **RAM**: Minimum 32GB

- **GPU Memory**: Minimum 16GB

### Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt
```

---

## 🗂️ Datasets

We support three public HSI datasets:

- **Indian Pines (IP)**
- **Salinas Valley (SA)**
- **University of Pavia (UP)**

You can download them from the following links:

- [Indian Pines](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)
- [Salinas](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)
- [Pavia University](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)

Place the datasets in the `data/` folder with the following structure:

```
data/
    indian.mat
    label.mat
  
```

---

## 🧪 Training & Evaluation

### Training

To train the model on the Indian Pines dataset with 5 labeled samples per class:

```bash
python indian_main.py
```



## 📄 Citation

If you use this code or find our work helpful, please cite:

```bibtex
@article{li2025incremental,
  title={Incremental Multitask Contrastive Learning Network for End-to-End Few-Shot Open-Set Classification of Hyperspectral Images},
  author={Li, Na and Song, Xiaopeng and Zhu, Wenxiang and Liu, Yongxu and Li, Chuang and Zhang, Weitao and Quan, Yinghui},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2025},
  volume={63},
  pages={1--16},
  doi={10.1109/TGRS.2025.3588540}
}
```

---

---

## 📜 License

This project is for academic use only. 

---


