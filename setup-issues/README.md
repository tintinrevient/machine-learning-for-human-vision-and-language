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
install_tensorflow(version="nightly")
```

2. downgrade numpy to a version below 1.16.3:
```
pip uninstall numpy
pip install --upgrade numpy==1.16.1
```

## References

* https://github.com/tensorflow/tensorflow/issues/10779
* https://github.com/rstudio/keras/issues/765
