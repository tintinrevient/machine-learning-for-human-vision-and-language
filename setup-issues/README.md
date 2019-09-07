## Issues encountered during the installation of R, RStudio & Keras on Mac OS

### Python - SSL: CERTIFICATE_VERIFY_FAILED

The error message reads as below:
```
 Error in py_call_impl(callable, dots$args, dots$keywords) : 
  Exception: URL fetch failure on https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb.npz: None -- [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed (_ssl.c:748) 
```

The solution is to install the newer certificate, with steps as below:
1. run the /Applications/Python 3.6/Install Certificates.command bash script to use Python DMG installer.


## References

* https://github.com/tensorflow/tensorflow/issues/10779
