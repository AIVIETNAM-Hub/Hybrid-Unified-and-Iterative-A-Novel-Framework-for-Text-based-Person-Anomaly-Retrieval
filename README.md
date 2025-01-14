## Contributors:
Hybrid-Unified-and-Iterative-A-Novel-Framework-for-Text-based-Person-Anomaly-Retrieval

Authors: Tien-Huy Nguyen, Huu-Loc Tran,Huu-Phong Phan-Nguyen, Quang-Vinh Dinh

## News
- [01/21/2025] Check out our checkpoints, fine-tuned on PAB dataset, now available on [Link]. 

- [01/14/2025] We decided to open source to research community, ensuring re-produce our method and helping them improve our baseline in the future.

- [12/29/2024] We achieved SOTA performance (89.23\% at R1 and 99.90\% at R10)on large-scale image- text Pedestrian Anomaly Behavior (PAB) dataset.


## Introduction
In this work, we present a novel hybrid, unified and iterative method. This method comnbined local and global perspectives, Unified Image-Text (UIT) Modeling and iterative ensemble. All of them helps model to learn more efficient and effective representation. Therefore, our primary contributions are:

- A hybrid approach blending local and global perspectives (LHP), enhancing the model’s ability to utilize both finegrained and holistic visual information.
- Unified Image-Text (UIT) Modeling integrates MIM, MLM, ITC and ITM tasks, leveraging LHP based feature selection for efficient and accurate multi-modal representation learning.
- We introduce a novel iterative ensemble algorithm that utilizes
the results of multiple models more effectively, helping to
improve the overall performance.
- Our comprehensive experiments demonstrate the effectiveness of our proposed method, achieving SOTA performance in text-based person anomaly retrieval on real-world datasets.


## Environment Setup

If using tuning or inferencing LHP, then do the following step:

```
cd ./pab
conda create --name lhp
conda activate lhp
pip3 install -r requirements_lhp.txt
```

else for UIT, then do the following step:

```
cd ./pab
conda create --name uit
conda activate uit
pip3 install -r requirements_uit.txt
```





## Folder Directory

The directory structures in our work is as follows
```
data/PAB/
pab/
└── data/
    └── PAB/
        └── annotation/
        └── name-masked_test-set/
        └── train/
    └── blip/
    └── beit3/
    └── checkpoint/
    └── ...
```

## Training

#### Finetuning LHP model:
```
cd ./unilm/beit3

CUDA_VISIBLE_DEVICES=0 python3 run_beit3_finetuning.py \
    --model beit3_large_patch16_384 \
	--task 356 \
	--drop_path 0.16 \
	--checkpoint_activations \
	--sentencepiece_model ./beit3.spm \
	--weight_decay 0.05 \
	--layer_decay 0.85 \
	--batch_size 184 \
	--update_freq 1 \
	--save_ckpt_freq 1 \
	--finetune ./beit3_large_patch16_384_coco_retrieval.pth \
	--data_path ../../data/PAB/ \
    --output_dir ./lhp_reproduce \
    --log_dir ./lhp_reproduce/log \
	--seed 16 \
	--save_ckpt \
	--input_size 384 \
	--lr 1e-5 \
	--warmup_steps 440 \
	--epochs 4
```

#### Finetuning UIT model:
```
cd ./WWW2025Workshop_mimcode/CMP_Baseline_Code
python3 run.py --task "cmp" 
                --dist "f4" 
                --output_dir "output/cmp"
```



## Inference

Firstly, you need to put checkpoint file into checkpoint directory

Secondly, to make iterative ensemble, you can run iteartively each model in the following step:

#### For LHP model:
```
python3 ./unilm/beit3/inference.py --checkpoint ./checkpoint/lhp/lhp_beit3.pth 
                                --tokenizer ./checkpoint/lhp/beit3.spm 
                                --image_folder ./data/PAB/name-masked_test-set/gallery --annotation ./data/PAB/name-masked_test-set/query.json --save_score ./sims_score/score_beit3_reproduce.pt  --output_file ./predictions/score_beit3_reproduce.txt
```

#### For BLIP2 model:
```
python3 ./blip2_infer.py --image_folder ./data/PAB/name-masked_test-set/gallery                 --annotation ./data/PAB/name-masked_test-set/query.json                    --save_score ./sims_score/score_blip2_reproduce.pt                      --output_file ./predictions/score_blip2.txt
```

#### For CLIP model:
```
python3 ./clip_infer.py --image_folder ./data/PAB/name-masked_test-set/gallery              --annotation ./data/PAB/name-masked_test-set/query.json                 --save_score ./sims_score/score_clip_reproduce.pt 
                     --output_file ./predictions/score_clip.txt
```


Finally, run iterative ensemble:

```
python3 WWW2025Workshop_mimcode/CMP_Baseline_Code/inference.py --config WWW2025Workshop_mimcode/CMP_Baseline_Code/configs/infer.yaml 
                                    --task cmp 
                                    --output_dir output 
                                    --checkpoint ./checkpoint/mim/mim.pth 
                                    --output_file reproduce.txt 
                                    --beit3_weight 0.925 
                                    --beit3_score ./sims_score/score_beit3_reproduce.pt 
                                    --blip2_weight 0.9 
                                    --blip2_score ./sims_score/score_blip2_reproduce.pt 
                                    --clip_weight 0.9 
                                    --clip_score ./sims_score/score_clip_reproduce.pt 
```

