## init conda environment:
conda create -n BertGCN python=3.8 -y
conda activate BertGCN

## install dependencies:
pip install torch==2.2.1+cu118 torchaudio==2.2.1+cu118 torchvision==0.17.1+cu118 torchdata==0.7.1 --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets nltk scipy pytorch-ignite scikit-learn pydantic tqdm

## install dgl:
pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html

## verification:
python -c "import torch; import torchdata; import dgl; print(f'Torch version: {torch.__version__}'); print(f'Torch CUDA available?: {torch.cuda.is_available()}'); print(f'Torch CUDA version: {torch.version.cuda}'); print(f'TorchData version: {torchdata.__version__}'); print(f'DGL version: {dgl.__version__}')"

## sample run:
python prepare_hf_dataset.py --dataset isarcasm
python build_graph.py isarcasm --seed 42
python train_bert_gcn.py --dataset isarcasm --seed 42 --device cuda --nb_epochs 50 --bert_init jcblaise/roberta-tagalog-base