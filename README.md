# Training spiking neural networks with Forward Porpogation Through Time (FPTT) 
--------------

This repository contains code to reproduce the key findings of "Training spiking neural networks with Forward Porpogation Through Time".
This code implements the spiking recurrent networks with **Liquid Time-Constant** spiking neurons (LTC) on [Pytorch]([PyTorch](https://pytorch.org/)) for various tasks. 
This is scientific software, and as such subject to many modifications; we aim to further improve the software to become more user-friendly and extendible in the future. 

## Datasets
-----
1. S/P-MNIST, R-MNIST: This dataset can easily be found in torchvision.datasets.MNIST([MNIST](https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST))
2. Fashion-MNIST: This dataset can easily access via torchvision.datasets.FashionMNIST([FashionMNIST](https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html#torchvision.datasets.FashionMNIST))
2. DVS dataset: [SpikingJelly]([SpikingJelly](https://github.com/fangwei123456/spikingjelly)) includes neuromorphic datasets ([Gesture128-DVS](https://openaccess.thecvf.com/content_cvpr_2017/html/Amir_A_Low_Power_CVPR_2017_paper.html) and [Cifar10-DVS]((https://internal-journal.frontiersin.org/articles/10.3389/fnins.2017.00309/full)).You can also download the datasets from [official sit]( https://research.ibm.com/interactive/dvsgesture/). Our prerpocess of DVS datasets also support in  [SpikingJelly]([SpikingJelly](https://github.com/fangwei123456/spikingjelly)). 
4.  PASCAL Visual Object Classes (VOC) dataset([VOC](http://host.robots.ox.ac.uk/pascal/VOC/)) contains 20 object categories. Each image in this dataset has pixel-level segmentation annotations, bounding box annotations, and object class annotations. This dataset has been widely used as a benchmark for object detection, semantic segmentation, and classification tasks. In this paper, **SP**iking-**Y**OLO (SPYv4) network was trained and tested on VOC07+12.

## Requirements
-----
0. Pyhton 3.8.10
1. A working version of python and Pytorch This should be easy: either use the Google Colab facilities, or do a simple installation on your laptop could probabily using pip. ([Start Locally | PyTorch](https://pytorch.org/get-started/locally/)) **torch==1.7.1**
2. SpikingJelly([SpikingJelly](https://github.com/fangwei123456/spikingjelly))
3. For object detection taskes, it requires OpenCV 2

## FPTT posude code
-----
```python
for i in range(sequence_len): # read the sequence
    if i ==0:
        model.init_h(x_in.shape[0]) # At first step initialize the hidden states
    else:
        model.h = list(v.detach() for v in model.h) # detach computation graph from previous timestep
    out = model.forward_t(x_in[:,:,i]) # read input and generate output
    loss_c = (i)/sequence_len*criterion(out, targets) # get prediction loss 
    loss_r = get_regularizer_named_params(named_params, _lambda=1.0 ) # get regularizer loss
    loss = loss_c+loss_r
    optimizer.zero_grad()
    loss.backward() # calculate gradient of current timestep
    optimizer.step() # update the network
    post_optimizer_updates( named_params, epoch) # update trace \bar{w} and \delta{l}
```
## Object detection Demo
----
A video demo of **SP**iking-**Y**OLO (SPYv4) :

[![SPYv4](https://i9.ytimg.com/vi_webp/Ue1_RJVfDcw/mqdefault.webp?v=629a1ba9&sqp=CKzo-ZYG&rs=AOn4CLA6pqYdK9OaH4LKNlqGixLDATNG6A)](https://www.youtube.com/watch?v=Ue1_RJVfDcw&ab_channel=BojianYin)

## Running code
---
You can find more details in readme file of each task.
1.  [Adding task](https://github.com/byin-cwi/sFPTT/tree/main/fptt/fptt_AddTask)
2.  [P/S-MNIST task](https://github.com/byin-cwi/sFPTT/tree/main/fptt/fptt_mnist)
3.  [Image and DVS task](https://github.com/byin-cwi/sFPTT/tree/main/fptt/fptt_img)
4.  [Spiking YOLO Demo](https://github.com/byin-cwi/sFPTT/tree/main/fptt/yolo_demo_show)


Finally, we’d love to hear from you if you have any comments or suggestions.


### References
----

[1]. https://github.com/bubbliiiing/yolov4-tiny-pytorch

## License

MIT

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [dill]: <https://github.com/joemccann/dillinger>
   [git-repo-url]: <https://github.com/joemccann/dillinger.git>
   [john gruber]: <http://daringfireball.net>
   [df1]: <http://daringfireball.net/projects/markdown/>
   [markdown-it]: <https://github.com/markdown-it/markdown-it>
   [Ace Editor]: <http://ace.ajax.org>
   [node.js]: <http://nodejs.org>
   [Twitter Bootstrap]: <http://twitter.github.com/bootstrap/>
   [jQuery]: <http://jquery.com>
   [@tjholowaychuk]: <http://twitter.com/tjholowaychuk>
   [express]: <http://expressjs.com>
   [AngularJS]: <http://angularjs.org>
   [Gulp]: <http://gulpjs.com>

   [PlDb]: <https://github.com/joemccann/dillinger/tree/master/plugins/dropbox/README.md>
   [PlGh]: <https://github.com/joemccann/dillinger/tree/master/plugins/github/README.md>
   [PlGd]: <https://github.com/joemccann/dillinger/tree/master/plugins/googledrive/README.md>
   [PlOd]: <https://github.com/joemccann/dillinger/tree/master/plugins/onedrive/README.md>
   [PlMe]: <https://github.com/joemccann/dillinger/tree/master/plugins/medium/README.md>
   [PlGa]: <https://github.com/RahulHP/dillinger/blob/master/plugins/googleanalytics/README.md>
