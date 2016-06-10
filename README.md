Evaluation of Deep Local Descriptors
============

# Introduction

This repository contains the source code of patches/features extraction and PCA learning.

# Requirements

* OpenCV 3.0 or newer.
* OpenBLAS + LAPACKE (for PCA operations).

Tested on Mac OSX with GNU C/C++ compilers.

# Installation

## Compilation
Run
```
$ mkdir build && cd build
$ cmake .. -DOPENBLAS_INCLUDE_DIR=</path/to/OpenBLAS/include>
$ make all
```
to compile the source code.

## Usage

Now to extract the local deep features, see [here](./caffe/README.md).

To extract patches from an image:
```
$ ./main -c extract_patches -i ../IMG_1069.JPG -o ./patches -r 20
```

To apply pca to extracted features:
```
$ ./main -c compute_pca -i raw_data.fvecs -o raw_data_128.fvecs -r 128 -D 4096
```
where `-D 4096` is the dimensionality of the features, and `-r 128` is the number of principal components (PCs) to be learnt.

# Licenses

This project only officially support GNU GPLv3 license.
Please see [LICENSE](./LICENSE.md).

# Contact

Please do not hesitate to contact me at [tuan.nguyenanh@hotmail.com](mailto:tuan.nguyenanh@hotmail.com) or [t_nguyen@hal.t.u-tokyo.ac.jp](mailto:t_nguyen@hal.t.u-tokyo.ac.jp).

# Citation

If you use this source code, please cite the following reference:

```
@techreport{TuanNguyen2016,
	author = {Nguyen, Tuan Anh and Duta, Ionut Cosmin and Yamasaki, Toshihiko and Aizawa, Kiyoharu},
	institution = {The University of Tokyo},
	title = {{Evaluation of Deep Features with PCA for Fine-grained Image Retrieval}},
	year = {2016}
}
```
