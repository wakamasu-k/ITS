# Git 日常操作マニュアル

対象プロジェクト：

```bash
~/ITS
```

接続先 GitHub：

```text
https://github.com/wakamasu-k/ITS
```

---

## 1. 最初にプロジェクトフォルダへ移動する

Git 操作をするときは，まず対象のプロジェクトフォルダへ移動します．

```bash
cd ~/ITS
```

現在地を確認します．

```bash
pwd
```

以下のように表示されれば OK です．

```bash
/home/wakamatsu/ITS
```

---

## 2. まず状態確認をする

Git を使うときは，最初に必ず状態を確認します．

```bash
git status
```

### 何も変更がない場合

```bash
nothing to commit, working tree clean
```

これは，ローカルの変更がなく，commit するものがない状態です．

### 変更されたファイルがある場合

```bash
Changes not staged for commit:
```

これは，ファイルは変更されているが，まだ commit 対象に入っていない状態です．

### 新しいファイルがある場合

```bash
Untracked files:
```

これは，Git がまだ管理していない新規ファイルがある状態です．

---

## 3. 普段の基本操作

基本は以下の流れです．

```bash
git status
git add ファイル名
git commit -m "変更内容"
git push
```

例：

```bash
git status
git add handover_shimizu/src/main.py
git commit -m "Update matching code"
git push
```

日本語の commit メッセージでも問題ありません．

```bash
git commit -m "特徴点マッチングのコードを更新"
```

---

## 4. 複数ファイルをまとめて追加する

変更したファイルをまとめて追加したい場合は，以下を使います．

```bash
git add .
```

ただし，研究プロジェクトでは，モデル重み，点群，画像，データセットなどが混ざりやすいため，commit 前に必ず確認します．

```bash
git diff --cached --name-only
```

---

## 5. commit 前に重いファイルが入っていないか確認する

以下のコマンドで，commit 対象に重いファイルが含まれていないか確認します．

```bash
git diff --cached --name-only | grep -E "\.(pth|pt|ckpt|npy|npz|pcd|bin|png|jpg|jpeg|gif)$"
```

何も表示されなければ OK です．

もし何か表示された場合，そのファイルは GitHub に上げない方がよい可能性があります．  
commit 対象から外すには以下を使います．

```bash
git restore --staged ファイル名
```

例：

```bash
git restore --staged model_camera_trained.pth
```

---

## 6. 安全にソースコードだけ追加する方法

研究用プロジェクトでは，全部まとめて追加するより，コード系ファイルだけ追加する方が安全です．

```bash
find TULIP_backup handover_shimizu shimizu \
  -type f \( \
  -name "*.py" -o \
  -name "*.ipynb" -o \
  -name "*.md" -o \
  -name "*.txt" -o \
  -name "*.yaml" -o \
  -name "*.yml" -o \
  -name "*.json" -o \
  -name "*.sh" -o \
  -name "requirements.txt" -o \
  -name "LICENSE" -o \
  -name ".gitignore" \
  \) -print0 | xargs -0 git add
```

追加後，以下で確認します．

```bash
git status
git diff --cached --name-only
```

問題なければ commit します．

```bash
git commit -m "Add project source code"
git push
```

---

## 7. GitHub から最新版を取得する

作業前に GitHub 側の最新版を取り込みたい場合は，以下を実行します．

```bash
git pull
```

もし以下のようなエラーが出た場合：

```bash
non-fast-forward
```

または，分岐した履歴を統合する必要がある場合は，以下を使います．

```bash
git pull --no-rebase origin main
```

初回接続時など，履歴が別々の場合は以下を使うことがあります．

```bash
git pull --no-rebase origin main --allow-unrelated-histories
```

---

## 8. `.gitignore` を更新する

`.gitignore` を編集します．

```bash
nano .gitignore
```

編集後，以下で commit して push します．

```bash
git add .gitignore
git commit -m "Update gitignore"
git push
```

---

## 9. 研究プロジェクトで `.gitignore` に入れるもの

以下は基本的に GitHub に上げない方がよいです．

```gitignore
# virtual environment
.venv/
venv/
env/

# editor
.vscode/

# Python cache
__pycache__/
*.pyc
*.pyo
*.pyd

# Jupyter
.ipynb_checkpoints/

# model weights
*.pth
*.pt
*.ckpt
*.onnx

# numpy / point cloud / large data
*.npy
*.npz
*.pcd
*.bin

# images
*.png
*.jpg
*.jpeg
*.gif

# outputs
outputs/
output_dir/
results/
runs/
checkpoints/
wandb/
logs/

# datasets
database_images/
kitti_train/
sampled_camera_images_all/
sampled_intensity_images_all/
```

---

## 10. すでに Git に追加してしまったファイルを外す

`.gitignore` に追加しても，すでに Git 管理されているファイルには効きません．  
その場合は，Git 管理からだけ外します．  
ローカルのファイル自体は消えません．

```bash
git rm --cached ファイル名
```

フォルダの場合：

```bash
git rm --cached -r フォルダ名
```

例：

```bash
git rm --cached -r .venv
git rm --cached model_camera_trained.pth
```

その後：

```bash
git add .gitignore
git commit -m "Remove ignored files from tracking"
git push
```

---

## 11. よくあるエラーと対応

### `nothing added to commit but untracked files present`

意味：

新しいファイルはあるが，まだ `git add` していない状態です．

対応：

```bash
git add ファイル名
git commit -m "Add files"
git push
```

---

### `non-fast-forward`

意味：

GitHub 側に，ローカルにはない変更があります．

対応：

```bash
git pull --no-rebase origin main
git push
```

---

### `remote origin already exists`

意味：

すでに GitHub の接続先が登録されています．

確認：

```bash
git remote -v
```

接続先を変更：

```bash
git remote set-url origin https://github.com/wakamasu-k/ITS.git
```

---

### `warning: adding embedded git repository`

意味：

Git リポジトリの中に，さらに別の Git リポジトリが入っています．

確認：

```bash
find . -mindepth 2 -name .git -type d
```

中の `.git` を削除します．

```bash
rm -rf 対象フォルダ/.git
```

ただし，以下は絶対に消してはいけません．

```bash
rm -rf .git
```

これは現在の ITS プロジェクト本体の Git 管理情報です．

---

## 12. 日常的なおすすめ手順

### 作業前

```bash
cd ~/ITS
git status
git pull
```

### 作業後

```bash
git status
git add 変更したファイル名
git diff --cached --name-only
git commit -m "変更内容"
git push
```

### 重いファイル確認

```bash
git diff --cached --name-only | grep -E "\.(pth|pt|ckpt|npy|npz|pcd|bin|png|jpg|jpeg|gif)$"
```

---

## 13. 最低限覚える版

普段はこれだけで大丈夫です．

```bash
cd ~/ITS
git status
git add 変更したファイル名
git commit -m "変更内容"
git push
```

まとめて追加したいとき：

```bash
git add .
git diff --cached --name-only
git commit -m "変更内容"
git push
```

ただし，`.pth`，`.npy`，`.pcd`，画像，データセットなどが入っていないかは必ず確認します．

---

## 14. 今回のプロジェクトで特に注意すること

以下のようなファイルは，GitHub に上げない方がよいです．

- `.venv/`
- `.pth`
- `.pt`
- `.npy`
- `.npz`
- `.pcd`
- `.bin`
- 大量の画像
- データセット
- 実験出力
- `wandb/`
- `checkpoints/`

GitHub には，基本的に以下を上げます．

- Python コード
- 設定ファイル
- README
- requirements.txt
- 実行スクリプト
- メモ用 Markdown
- 小さいサンプルファイル
