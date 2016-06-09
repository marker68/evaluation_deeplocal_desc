Feature extraction
======================

We will use `caffe` for feature extractions. All steps are described in [Caffe tutorial](http://caffe.berkeleyvision.org/gathered/examples/feature_extraction.html).
However, if you want to save feature blobs in `fvecs` format, just replace `caffe\tools\extract_features.cpp` with the source code in this directory and rebuild `caffe`.