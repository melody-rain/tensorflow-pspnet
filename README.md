# Tensorflow-PSPNet

This is an improvement of the implementation of [pudae's](https://github.com/pudae/tensorflow-pspnet). The improvements include:
1. Support ResNet-101, which is the original backbone network of [hszhao's PSPNet](https://github.com/hszhao/PSPNet).
2. Support auxiliary loss.
3. Support weighted softmax loss.
4. Convert the [pspnet101_VOC2012.caffemodel](https://github.com/hszhao/PSPNet) to Tensorflow model, which can be downloaded from [Baidu Yun Pan](https://pan.baidu.com/s/1i6QaK4X) or [Goole Drive](https://drive.google.com/open?id=18Gi3vHQYSp9s5-l_cSzp8zo4D8dpGqlD).
Download it and you can test VOC2012's images with ```test_segmentation.sh```.

The way to convert the caffe model to TF model can be done with [caffe-tensorflow](https://github.com/melody-rain/caffe-tensorflow). With an ugly way I converted the caffe model's names to the name scopes of PSPNet defined in ```pspnet_v1.py```.
It only supports pspnet101_VOC2012.caffemodel. For other networks there might be some slight modifications, such as the size of the input image.
  
# Prepare your dataset
Create your dataset description according to ```ade20k.py``` and register it in ```dataset_factory.py```.
Run ```convert_data.sh``` to convert your dataset to ```tfrecord```.

# Training
To train your dataset, modify the parameters in ```train_pspnet.sh```. Set ```DATASET_NAME``` according to your needs and run the scrpit:
```
./train_pspnet.sh
```

# Evaluation
Modify ```eval_pspnet.sh``` and run it:
```
./eval_pspnet.sh
```

# Inference
Set the image your want to test in ```test_segmentation.sh``` and run:
```
./test_segmentation.sh
```

