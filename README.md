# [TadML: A fast temporal action detection with Mechanics-MLP](https://github.com/BonedDeng/TadML)



## Introduction

•To the best of our knowledge,TadML achieves state-of-the-art or highly com petitive performance on standard datasets while surpassing the inference speed of previous methods by a large margin and the inference speed on model is astounding 4.44 video per second on THUMOS14.
*•* TadML proves that optical flow data is not indispensable in TAD apllication.Neck layers is the key to improve performance in both RGB stream and two stream.*βGiou* is More suitable for Tad
*•* We first propose a MLP model based on the newtonian mechanics, which prove MLP also suitable for TAD.It achieves highly competitive performance when using RGB and optical flow.

![](./Figure.jpg)

## Updates

---

## TODOs



- [x]  add mlp model code
- [ ] add inference code
- [ ] add training code
- [ ] support training/inference with video input

## Main Results

- THUMOS14

| Type      | Method    | RGB stream | 0.3   | 0.4   | 0.5   | 0.6   | 0.7   | Avg     |
| --------- | --------- | ---------- | ----- | ----- | ----- | ----- | ----- | ------- |
| Two-stage | BMN       | X          | 56    | 47.4  | 38.8  | 29.7  | 20.5  | 38.48   |
| Two-stage | BSN++     | X          | 59.9  | 45.9  | 41.3  | 31.9  | 22.8  | 40.36   |
| Two-stage | TAL-net   | X          | 53.2  | 48.5  | 42.8  | 33.8  | 20.8  | 39.8    |
| Two-stage | ContextLoc| X          | 68.3  | 63.8  | 54.3  | 41.8  | 26.2  | 50.88   |
| One-stage | TadTR     | X          | 62.4  | 57.4  | 49.2  | 37.8  | 26.3  | 46.6    |
| One-stage | **TadML**     | X          | 68.78 | 64.66 | 56.61 | 45.40 | 31.88 | **53.46**   |
| One-stage | **TadML**    | ✓          | 73.29 | 69.73 | 62.53 | 53.36 | 39.60 | **59.70**   |

## Install
# Compilation

Part of NMS is implemented in C++. The code can be compiled by

```shell
cd /train
python setup.py install --user
cd ..
The code should be recompiled every time you update PyTorch.
```
```
Folder Structure:
.
├── README.md
├── data/
│   ├── thumos/
│   │   ├── annotations/
│   │   ├── i3d_features/
│   │   └── ...
│   └── ...
├── hyps/
├── libs/
└── ckpt/
```

### Requirements

```linux
pip install -r requirements.txt
```
* Train TadMLP with thumos dataset and save ckpt in root directory.
```shell
python mian.py /hyps/thumos_i3d.yaml --output reproduce
```


## Citing

```
@article{TadML,
  title={TadML: A fast temporal action detection with Mechanics-MLP},
  auther={Bowen deng;Shuangliang Zhao;Dongchang Liu}
  year={2023}
}
```

## Contact

For questions and suggestions, please contact Bowen Deng.
