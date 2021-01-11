# CSE6250_project_team45
# Novel Application of Deep Learning in Detection of COVID-19
## Team Members
Kefei Wang, Wuguo Chen, Yi Gu, Xidu Qiu
## Author
name|email|gtid
:-:|:-:|:-:
Kefei Wang|kwang466@gatech.edu|kwang466
Wuguo Chen| wchen603@gatech.edu|wchen603
Yi Gu|xqiu48@gatech.edu|xqiu48
Yi Gu|ygu308@gatech.edu|ygu308

  


## Environment setup:

The environment is 
Python version: Python 3.7.9

To run the model, used the following environment, activated through conda
```
conada env create -f env.yml
```

* To load extract images and move to the data folder:
```
python dataPrep.py
```


* To train the base model:
```
python modelTunning.py
```

* To train the final optimal model:
```
python finalModel.py
```

* To visualize HEATMAP using final model:
```
python DenseNetHeatmap.py
```