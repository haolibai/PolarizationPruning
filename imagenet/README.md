# Train on ImageNet

> Note: We recommend to use 4x GTX 1080Ti (or higher) to train on ImageNet data. In this way, training usually takes about 1.5 days.

Dataset:
1. Download the ImageNet dataset
2. Add a softlink (optional)

    ```bash
    ln -s path/to/imagenet ./ImageNet
    ```
3. Create checkpoint directory
    ```bash
    mkdir ./checkpoints/  # checkpoints and tensorboard events
    mkdir ./backup_ckpt/  # backup checkpoints every five epochs
    ```

    > Note: Checkpoints will be named as `checkpoint.pth.tar` and `model_best.pth.tar`. The program will overwrite existing checkpoints in the checkpoint directory. Make sure to use a different checkpoint directory name in different experiments. We recommend using the experiment name as directory, e.g., `--save ./resnet50-imagenet`.

## Training

### ResNet-50

1. Sparsity training
    ```bash
    # train sparse model
    python -u main.py ./ImageNet -loss zol -b 256 --lbd 8e-5 --t 1.4 --lr 1e-1 1e-2 1e-3 --decay-epoch 30 60 --epochs 90 --arch resnet50 --workers 25 --world-size 1 --dist-url tcp://localhost:23456 --dist-backend gloo --rank 0 --save ./checkpoints/ --backup-path ./backup_ckpt/
    ```

2. Prune the sparse model
    ```bash
    # prune
    python resprune-expand.py ./ImageNet --pruning-strategy grad --no-cuda --model ./checkpoints/model_best.pth.tar --save ./checkpoints/
    ```
3. Fine-tune the pruned model

    ```bash
    # fine-tune
    python -u main_finetune.py ./ImageNet --arch resnet50 --epoch 100 --lr 1e-2 1e-3 1e-4 3e-5 --decay-epoch 30 60 80 --refine ./checkpoints/pruned.pth.tar --expand -b 256 --workers 25 --world-size 1 --dist-url tcp://localhost:23456 --dist-backend gloo --rank 0 --save ./checkpoints/ --backup-path ./backup_ckpt/
    ```

### VGG-11

1. Sparsity training

    ```bash
    # sparsity training
    python -u main.py ./ImageNet -loss zol -b 256 --lbd 5e-5 --t 1.8 --lr 1e-1 1e-2 1e-3 --epochs 90 --decay-epoch 30 60 --arch vgg11 --workers 25 --world-size 1 --dist-url tcp://localhost:23456 --dist-backend gloo --rank 0 --save ./checkpoints/ --backup-path ./backup_ckpt/
    ```

2. Pruning

    ```bash
    # pruning
    python prune.py --data ./ImageNet --pruning-strategy grad --no-cuda --model ./checkpoints/model_best.pth.tar --save ./checkpoints/
    ```

3. Fine-tuning
    ```bash
    # fine-tuning
    python -u main_finetune.py ./ImageNet --arch vgg11 --epoch 100 --lr 1e-2 1e-3 1e-4 3e-5 --decay-epoch 30 60 80 --refine ./checkpoints/pruned.pth.tar -b 256 --workers 25 --world-size 1 --dist-url tcp://localhost:23456 --dist-backend gloo --rank 0 --save ./checkpoints/ --backup-path ./backup_ckpt/

    ```



## Note

### Learning rate

The learning rate is determined by two arguments: 

- `--lr`: a list of learning rates in different stages.
- `--decay-epoch`: the epoch to decay learning rate.

