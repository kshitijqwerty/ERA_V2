# Assignment 18 

### Improving Vanilla Transformer Training
This repo contains the Model and training script for training a Vanilla Tranformer from Scratch on English to Italian Translation tokenised dataset. This repo adds faster training using One Cycle Policy and smart batching for reduced token sizes per batch while training.

Trained on 4070 Ti Super with 18 Epochs with final loss of 1.671 (Target under 1.8 within 18 epochs)

## Training Logs
```
Using device : cuda
Max length of the source sentence : 309
Max length of the source target : 274
0
Processing Epoch 00: 100%|██████████| 1213/1213 [01:37<00:00, 12.48it/s, loss=5.946]
1
Processing Epoch 01: 100%|██████████| 1213/1213 [01:35<00:00, 12.75it/s, loss=5.785]
2
Processing Epoch 02: 100%|██████████| 1213/1213 [01:33<00:00, 12.94it/s, loss=4.966]
3
Processing Epoch 03: 100%|██████████| 1213/1213 [01:37<00:00, 12.44it/s, loss=4.954]
4
Processing Epoch 04: 100%|██████████| 1213/1213 [01:33<00:00, 12.95it/s, loss=4.892]
5
Processing Epoch 05: 100%|██████████| 1213/1213 [04:58<00:00,  4.07it/s, loss=3.761]
6
Processing Epoch 06: 100%|██████████| 1213/1213 [01:39<00:00, 12.19it/s, loss=3.873]
7
Processing Epoch 07: 100%|██████████| 1213/1213 [01:33<00:00, 12.95it/s, loss=3.754]
8
Processing Epoch 08: 100%|██████████| 1213/1213 [01:33<00:00, 12.92it/s, loss=3.045]
9
Processing Epoch 09: 100%|██████████| 1213/1213 [01:40<00:00, 12.06it/s, loss=2.810]
10
Processing Epoch 10: 100%|██████████| 1213/1213 [01:33<00:00, 13.04it/s, loss=2.828]
11
Processing Epoch 11: 100%|██████████| 1213/1213 [01:31<00:00, 13.22it/s, loss=2.464]
12
Processing Epoch 12: 100%|██████████| 1213/1213 [01:32<00:00, 13.09it/s, loss=2.075]
13
Processing Epoch 13: 100%|██████████| 1213/1213 [02:00<00:00, 10.05it/s, loss=1.979]
14
Processing Epoch 14: 100%|██████████| 1213/1213 [01:32<00:00, 13.06it/s, loss=2.038]
15
Processing Epoch 15: 100%|██████████| 1213/1213 [02:54<00:00,  6.96it/s, loss=1.873]
16
Processing Epoch 16: 100%|██████████| 1213/1213 [01:38<00:00, 12.28it/s, loss=1.714]
17
Processing Epoch 17: 100%|██████████| 1213/1213 [08:22<00:00,  2.42it/s, loss=1.671]
```