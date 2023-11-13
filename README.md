# GenPose: Generative Category-level Object Pose Estimation via Diffusion Models (NeurIPS 2023)
This is the official Pytorch implementation of paper <a href="https://arxiv.org/pdf/2306.10531.pdf">GenPose: Generative Category-level Object Pose Estimation via Diffusion Models</a>. For more information about this paper, please refer to our <a href="https://sites.google.com/view/genpose">project page</a>.

![Pipeline](/assets/pipeline.png)

## Abstract
  Object pose estimation plays a vital role in embodied AI and computer vision, enabling intelligent agents to comprehend and interact with their surroundings. Despite the practicality of category-level pose estimation, current approaches encounter challenges with partially observed point clouds, known as the multi-hypothesis issue. In this study, we propose a novel solution by reframing category- level object pose estimation as conditional generative modeling, departing from traditional point-to-point regression. Leveraging score-based diffusion models, we estimate object poses by sampling candidates from the diffusion model and aggregating them through a two-step process: filtering out outliers via likelihood estimation and subsequently mean-pooling the remaining candidates. To avoid the costly integration process when estimating the likelihood, we introduce an alternative method that trains an energy-based model from the original score- based model, enabling end-to-end likelihood estimation. Our approach achieves state-of-the-art performance on the REAL275 dataset and demonstrates promising generalizability to novel categories sharing similar symmetric properties without fine-tuning. Furthermore, it can readily adapt to object pose tracking tasks, yielding comparable results to the current state-of-the-art baselines. 

## Requirements
- Ubuntu 20.04
- Python 3.8.15
- Pytorch 1.12.0
- Pytorch3d 0.7.2
- CUDA 11.3
- 1 * NVIDIA RTX 3090

## Installation

- ### Install pytorch
``` bash
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
```


- ### Install pytorch3d from a local clone
``` bash
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
git checkout -f v0.7.2
pip install -e .
```

- ### Install from requirements.txt
``` bash
pip install -r requirements.txt 
```

- ### Compile pointnet2
``` bash
cd networks/pts_encoder/pointnet2_utils/pointnet2
python setup.py install
```

## Download dataset and models
- Download camera_train, camera_val, real_train, real_test, ground-truth annotations and mesh models provided by <a href ="https://github.com/hughw19/NOCS_CVPR2019">NOCS</a>. Unzip and organize these files in $ROOT/data as follows:
``` bash
data
├── CAMERA
│   ├── train
│   └── val
├── Real
│   ├── train
│   └── test
├── gts
│   ├── val
│   └── real_test
└── obj_models
    ├── train
    ├── val
    ├── real_train
    └── real_test
```

- Preprocess NOCS files following <a  href ="https://github.com/mentian/object-deformnet">SPD</a>. 

We provide the preprocessed testing data (REAL275) and checkpoints <a href="https://drive.google.com/file/d/1UrSVb7qYIOGWSB4z6W8IrBdRm-8M0Q-N/view?usp=sharing">here</a> for a quick evaluation. Download and organize the files in $ROOT/results as follows:
``` bash
results
├── ckpts
│   ├── EnergyNet
│   │   └── ckpt_genpose.pth
│   └── ScoreNet
│       └── ckpt_genpose.pth
├── evaluation_results
│   ├── segmentation_logs_real_test.txt
│   └── segmentation_results_real_test.pkl
└── mrcnn_results
    ├── real_test
    └── val
```
The *ckpts* are the trained models of GenPose.

The *evaluation_results* are the preprocessed testing data, which contains the segmentation results of Mask R-CNN, the segmented pointclouds of obejcts, and the ground-truth poses. 
  
The *mrcnn_results* are the segmentation results from <a href="https://drive.google.com/file/d/1p72NdY4Bie_sra9U8zoUNI4fTrQZdbnc/view">here</a> provided by <a href="https://github.com/mentian/object-deformnet">SPD</a>.

**Note**: You need to preprocess the dataset as mentioned before first if you want to evaluate on CAMERA dataset.

## Training
Set the parameter '--data_path' in scripts/train_score.sh and scripts/train_energy.sh to your own path of NOCS dataset.

- ### Score network
Train the score network to generate the pose candidates.
``` bash
bash scripts/train_score.sh
```
- ### Energy network
Train the energy network to aggragate the pose candidates.
``` bash
bash scripts/train_energy.sh
```

## Evaluation
Set the parameter *--data_path* in *scripts/eval_single.sh* to your own path of NOCS dataset.

- ### Evaluate on REAL275 dataset.
Set the parameter *--test_source* in *scripts/eval_single.sh* to *'real_test'* and run:
``` bash
bash scripts/eval_single.sh
```
- ### Evaluate on CAMERA dataset.
Set the parameter *--test_source* in *scripts/eval_single.sh* to *'val'* and run:
``` bash
bash scripts_eval_pose_estimation.sh
```

## Citation
If you find our work useful in your research, please consider citing:
``` bash
@article{zhang2023genpose,
  title={GenPose: Generative Category-level Object Pose Estimation via Diffusion Models},
  author={Zhang, Jiyao and Wu, Mingdong and Dong, Hao},
  journal={Advances in neural information processing systems},
  year={2023}
}
```

## Contact
If you have any questions, please feel free to contact us:

Jiyao Zhang: [jiyaozhang@stu.pku.edu.cn](mailto:jiyaozhang@stu.pku.edu.cn), Mingdong Wu: [wmingd@pku.edu.cn](mailto:wmingd@pku.edu.cn), Hao Dong: [hao.dong@pku.edu.cn](mailto:hao.dong@pku.edu.cn).

## License
This project is released under the MIT license. See [LICENSE](LICENSE) for additional details.
