TULIPの学習結果
[02:35:30.640977] Epoch: [599] Total time: 0:08:37 (0.1970 s / it)
[02:35:30.642320] Averaged stats: lr: 0.000000  loss: 0.0028 (0.0029)
[02:35:31.409578] Training time 3 days, 17:08:43
[02:35:31.409769] Training finished
このように表示された
評価実験
(.venv_tulip) wakamatsu@1973700X:~/ITS/TULIP_backup$ bash bash_scripts/tulip_evaluation_kitti.sh
| distributed init (rank 0): env://, gpu 0
[18:55:42.790068] job dir: /home/wakamatsu/ITS/TULIP_backup/tulip
[18:55:42.790341] Namespace(accum_iter=1,
batch_size=64,
blr=0.001,
circular_padding=True,
data_path_high_res='./dataset/KITTI/',
data_path_low_res='./dataset/KITTI/',
dataset_select='kitti',
device='cuda',
dist_backend='nccl',
dist_on_itp=False,
dist_url='env://',
distributed=True,
entity='myentity',
epochs=400,
eval=True,
gpu=0,
grid_size=0.1,
img_size_high_res=[64,
1024],
img_size_low_res=[16,
1024],
in_chans=1,
keep_close_scan=False,
keep_far_scan=False,
local_rank=-1,
log_dir='./output_dir',
log_transform=True,
lr=None,
mc_drop=True,
min_lr=0.0,
model_select='tulip_base',
noise_threshold=0.03,
num_mcdropout_iterations=50,
num_workers=10,
output_dir='./output_dir',
patch_size=[1,
4],
patch_unmerging=True,
pin_mem=True,
pixel_shuffle=True,
project_name='kitti_evaluation',
rank=0,
remove_mask_token=False,
resume='./experiment/kitti/tulip_base/checkpoint-599.pth',
roll=False,
run_name='tulip_base',
save_frequency=100,
save_pcd=True,
seed=0,
start_epoch=0,
swin_v2=False,
wandb_disabled=True,
warmup_epochs=40,
weight_decay=0.05,
window_size=[2,
8],
world_size=1)
環境はこの様になっている


==============================
  TULIP 最終評価スコア（平均値）
==============================
MAE : 0.00330
CHAMFER_DIST : 0.08653
IOU : 0.42380
PRECISION : 0.60192
RECALL : 0.58496
F1 : 0.59327
==============================

MAE
テント正解の点が平均してどれくらいズレているのかけいさんしたもの
$$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} | y_i - \hat{y}_i |$$

RMSE
ズレを単純に足すのではなく「スレを二乗してかr平均し、最後にルートをかける」という

intensityがなかったので、今コマンドで実行を進める
