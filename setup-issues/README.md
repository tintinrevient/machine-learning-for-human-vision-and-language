## Issues encountered during the installation of R, RStudio & Keras on Mac OS

### Python - SSL: CERTIFICATE_VERIFY_FAILED

The error message reads as below:
```
 Error in py_call_impl(callable, dots$args, dots$keywords) : 
  Exception: URL fetch failure on https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb.npz: None -- [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed (_ssl.c:748) 
```

The solution is to install the newer certificate, with steps as below:
1. run the /Applications/Python 3.6/Install Certificates.command bash script to use Python DMG installer.


### Reuters Dataset for R

The error message reads as below:
```
 Error in py_call_impl(callable, dots$args, dots$keywords) : 
  ValueError: Object arrays cannot be loaded when allow_pickle=False
```

This is due to a change in numpy which makes necessary a corresponding change in tensorflow.

The solutions are as the following two:

1. install the current tensorflow nightly version instead (this already has the above fix):
```
> install_tensorflow(version="nightly")
```

2. downgrade numpy to a version below 1.16.3:
```
$ pip uninstall numpy
$ pip install --upgrade numpy==1.16.1
```

### Check Python used in RStudio

Execute the following command in the RStudio console:
```
library(reticulate)
py_config()
```

Below is the corresponding Python information:
```
python:         /Users/zhaoshu/.virtualenvs/r-reticulate/bin/python
libpython:      /Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/config-3.6m-darwin/libpython3.6.dylib
pythonhome:     /Library/Frameworks/Python.framework/Versions/3.6:/Library/Frameworks/Python.framework/Versions/3.6
version:        3.6.2 (v3.6.2:5fd33b5926, Jul 16 2017, 20:11:06)  [GCC 4.2.1 (Apple Inc. build 5666) (dot 3)]
numpy:          /Users/zhaoshu/.virtualenvs/r-reticulate/lib/python3.6/site-packages/numpy
numpy_version:  1.17.1
tensorflow:     /Users/zhaoshu/.virtualenvs/r-reticulate/lib/python3.6/site-packages/tensorflow

python versions found: 
 /Users/zhaoshu/.virtualenvs/r-reticulate/bin/python
 /usr/bin/python
 /usr/local/bin/python
 /usr/local/bin/python3
```

To downgrade RStudio's python's numpy version, "/Users/zhaoshu/.virtualenvs/r-reticulate/bin/pip" should be used, instead of the generic pip command in Mac OS, and RStudio's python's site packages' path is "/Users/zhaoshu/.virtualenvs/r-reticulate/lib/python3.6/site-packages".

To check the current numpy version, execute the following command:
```
$ /Users/zhaoshu/.virtualenvs/r-reticulate/bin/python

>>> import numpy
>>> print (numpy.__version__)
```

### Python - PIL.Image

The error message reads as below:
```
ImportError: Could not import PIL.Image. The use of `load_img` requires PIL.
```

The solution is to execute the command as below to install pillow:
```
$ /Users/zhaoshu/.virtualenvs/r-reticulate/bin/pip install pillow
```

## References

* https://github.com/tensorflow/tensorflow/issues/10779
* https://github.com/rstudio/keras/issues/765
* https://rstudio.github.io/reticulate/articles/versions.html
