# CS570

1. Download the dataset.
~~~
python kth_download.py
mv kth-human-motion/* kth
rm kth-human-motion -r
~~~

Make the downloaded file in the folder kth. As ./kth/boxing/person01_boxing_d1_uncomp.avi.

2. Make the dataset for training.
~~~
mkdir data
python KTH.py
~~~

3. Run the model
~~~
python run.py
~~~


### KTH Dataset
@inproceedings{inproceedings,
author = {Roth, Peter M. and Mauthner, Thomas and Khan, Inayatullah and Bischof, Horst},
year = {2009},
month = {11},
pages = {546 - 553},
title = {Efficient human action recognition by cascaded linear classifcation},
doi = {10.1109/ICCVW.2009.5457655}
}