  ## Image and DVS dataset
In this subpackage we conduct experiments on the MNIST, Fashion-MNIST, DVS-Gesture and DVS-Cifar10 experiments.

You can modify the dataset detail in "datasets.py" like number of frames or sequence length.
Code Running: 
  ```bash
python train_gesture_dvs_snn.py --dataset DVS-Gesture --parts 28 --batch_size 32 --nhid 512 --lr 3e-3 --when 20 30 --epochs 100 --optim Adamax --log-interval 8
  ```
