Code mostly generated with ChatGPT

Transformer policy CEM
```
train_cem --num-epochs 1000 --num-rollouts-per-epoch 200 --num-eval-rollouts-per-epoch 20 --device cuda:0  --num-train-ops-per-epoch 200  --max-steps 200 --learning-rate 3e-4 --batch-size 20 --num-best 20 --tb ./tb/tns6
```

MLP Policy CEM
```
train_cem --num-epochs 1000 --num-rollouts-per-epoch 200 --num-eval-rollouts-per-epoch 20 --device cuda:0  --num-train-ops-per-epoch 200  --max-steps 200 --learning-rate 3e-4 --batch-size 20 --num-best 20 --tb ./tb/cem2 --policy-type mlp --hidden-size 64
```
