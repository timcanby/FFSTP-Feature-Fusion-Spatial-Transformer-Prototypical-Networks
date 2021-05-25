# FFSTP-Feature-Fusion-Spatial-Transformer-Prototypical-Networks
sample code for 'FFSTP: Feature Fusion Spatial Transformer Prototypical Networks'

This is a simple training demo for Feature Fusion Spatial Transformer Prototypical Networks.
(2021.5.26:*Update)



In classical materials, seals are often covered by handwritten words.

![](train.gif)

To split the seal areas from background ,we can project RGB three channels information of image into three-dimensional space

automatically extracts areas with more red components by using k-means clustering
![](cross.gif)

-------

The default value of K is 3.
Adjusting the results of seal‘s area extraction by adjusting the value of K
run:
```python
python3 Binarization.py --ImageSelecter=imageDir(default='test.jpg') --clusters=K
Ex.python Binarization.py --ImageSelecter=test.jpg  --clusters=3

```
to get results of Binarization.The generated file will be named Binarization(0-n).jpg
Select the file you want to segment to single character.

run:
```python
python3 SealsCharacters_Segmentation.py --ImageSelecter=imageDir(default='Binarization2.jpg') 
Ex.python3 SealsCharacters_Segmentation.py --ImageSelecter=Binarization2.jpg
```
Then you can see the Characters Segmentation result in the result folder.

![](image/2.gif)

Authors
-------

- @Kangying Li 


License
