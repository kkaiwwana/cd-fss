# On Efficient Cross-domain Few-shot Learning

## Quick Start
1. Download the checkpoint of SAM and place it at `weights/sam_vit_b_01ec64.pth`
2. Prepare your swanlab API key, which is an alternative to wandb; You can also quickly switch to wandb by updating several keywords in the do.
3. Download your dataset (CrackVision 12K can be found [here](https://rdr.ucl.ac.uk/articles/dataset/CrackVision12K/26946472)) and the link to our TinyCrack dataset can be found in the paper.
4. Simple training in source domain with `python scripts/run.py runner=train exp.cmt=[YOUR_COMMENT]` and adapte to target domain with `python scripts/run.py runner=adapt exp.cmt=[YOUR_COMMENT]`. You can update config files, which is organized clearly to select different model setups. You can also simply append/override keywords in config with Hydra.
5. Track your training in Swanlab/Wandb. All metrics/visualizations will be dumped there (See examples below).
<img width="3318" height="1707" alt="image" src="https://github.com/user-attachments/assets/fd4c0408-b564-41c8-a079-f0c0000cb95c" />

## Closing Comments
The code is applicable to general CDFSS tasks, as long as the dataset class is implemented. This repo also fixes the data **leakage issues** (if you read Chinese, I have an [article](https://zhuanlan.zhihu.com/p/1913244721020662145) for this) found in some existing repositories.
Notably, this repo uses Hydra + Wandb (SwanLab) + PyTorch Lightning, making training, configuration, and experiment tracking extremely simple. Feel free to reuse it for your own projects.
***This project was implemented from April to July 2025, back when AI coding still producing tons of bugs and probably hadn't changed the world yet. So, 99% of this project was coded by myself. I've told many people that I think coding is the most enjoyable part of doing research. It's pure. It's logic.
To a time that may never return.***
