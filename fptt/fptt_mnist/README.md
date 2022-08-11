  ## P/S-MNIST Task
In this subpackage we conduct experiments on the Sequential MNIST experiments. 
  The SNN model was implemented in "snn_model_LIF.py". You can run experiments via following command 
  ```bash
  python train_mnist_snn_v2.py --dataset MNIST-10 --parts 784 --batch_size 512 --nhid 256 --alpha 0.1 --optim Adamax --lr 3e-3 --beta 0.5 --clip 1.
  ```
