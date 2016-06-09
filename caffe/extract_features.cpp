//
// Created by TuanNguyen on 2016/03/14.
//

#include <stdio.h>  // for snprintf
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using boost::shared_ptr;
using std::string;
namespace db = caffe::db;

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv);
template<typename DataType>
inline void save_data(
    const char * filename,
    DataType * data,
    size_t ndata,
    size_t points,
    const int * id,
    bool save_id,
    bool verbose);

int main(int argc, char** argv) {
  return feature_extraction_pipeline<float>(argc, argv);
  //  return feature_extraction_pipeline<double>(argc, argv);
}

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  const int num_required_args = 6;
  if (argc < num_required_args) {
      LOG(ERROR)<<
      "This program takes in a trained network and an input data layer, and then"
          " extract features of the input data produced by the net.\n"
          "Usage: extract_features  pretrained_net_param"
          "  feature_extraction_proto_file  extract_feature_blob_name"
          "  save_feature_dataset_name  num_mini_batches"
          "  [CPU/GPU] [DEVICE_ID=0]\n";
      return 1;
    }
  int arg_pos = num_required_args;

  arg_pos = num_required_args;
  if (argc > arg_pos && strcmp(argv[arg_pos], "GPU") == 0) {
      LOG(ERROR)<< "Using GPU";
      uint device_id = 0;
      if (argc > arg_pos + 1) {
          device_id = atoi(argv[arg_pos + 1]);
          CHECK_GE(device_id, 0);
        }
      LOG(ERROR) << "Using Device_id=" << device_id;
      Caffe::SetDevice(device_id);
      Caffe::set_mode(Caffe::GPU);
    } else {
      LOG(ERROR) << "Using CPU";
      Caffe::set_mode(Caffe::CPU);
    }

  arg_pos = 0;  // the name of the executable
  std::string pretrained_binary_proto(argv[++arg_pos]);
  std::string feature_extraction_proto(argv[++arg_pos]);
  shared_ptr<Net<Dtype> > feature_extraction_net(
      new Net<Dtype>(feature_extraction_proto, caffe::TEST));
  feature_extraction_net->CopyTrainedLayersFrom(pretrained_binary_proto);

  std::string blob_names(argv[++arg_pos]);

  std::string dataset_names(argv[++arg_pos]);
  CHECK(feature_extraction_net->has_blob(blob_names))
  << "Unknown feature blob name " << blob_names
  << " in the network " << feature_extraction_proto;

  int num_mini_batches = atoi(argv[++arg_pos]);

  LOG(ERROR)<< "Extracting Features";

  std::vector<Blob<float>*> input_vec;
  int image_indices;
  size_t size = 0, prev;
  Dtype * all_features = (Dtype *)::operator new(100 * sizeof(Dtype));
  Dtype * ptr = all_features;
  for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index) {
      feature_extraction_net->Forward(input_vec);
      const shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net
          ->blob_by_name(blob_names);
      int batch_size = feature_blob->num();
      prev = size;
      size +=  feature_blob->count();
      all_features = (Dtype *)realloc(all_features, size * sizeof(Dtype));
      ptr = all_features + prev;
      int dim_features = feature_blob->count() / batch_size;
      const Dtype* feature_blob_data;
      for (int n = 0; n < batch_size; ++n) {
          feature_blob_data = feature_blob->cpu_data() +
                              feature_blob->offset(n);
          memcpy(ptr, feature_blob_data, dim_features * sizeof(Dtype));
          ptr += dim_features;

          ++image_indices;
          if (image_indices % 1000 == 0) {
              LOG(ERROR)<< "Extracted features of " << image_indices <<
              " query images for feature blob " << blob_names;
            }
        }  // for (int n = 0; n < batch_size; ++n)
    }  // for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index)
  LOG(ERROR)<< "Extracted features of " << image_indices <<
  " query images for feature blob " << blob_names;
  int * id;
  save_data<Dtype>(dataset_names.c_str(), all_features, size, 1, id, false, true);

  LOG(ERROR)<< "Successfully extracted the features!";
  return 0;
}

template<typename DataType>
inline void save_data(
    const char * filename,
    DataType * data,
    size_t ndata,
    size_t points,
    const int * id,
    bool save_id,
    bool verbose) {
  size_t size;
  // Save the frame data
  int fd = open(filename, O_RDWR | O_CREAT | O_TRUNC,(mode_t)0600); // file description
  if(fd < 0) {
      if(verbose)
        std::cerr << "Cannot open the file " << filename << std::endl;
      exit(1);
    }

  // The number of bytes to be written out.
  size = ndata * points * sizeof(DataType);
  if(save_id)
    size += ndata * sizeof(int);

  int result = lseek(fd, size, SEEK_SET);
  if (result == -1) {
      close(fd);
      if(verbose)
        std::cerr << "Error calling lseek() to 'stretch' the file" <<  std::endl;
      exit(1);
    }

  int status = write(fd, "", 1);
  if(status != 1) {
      if(verbose)
        std::cerr << "Cannot write to file" << std::endl;
      exit(1);
    }

  unsigned char * fd_map = (unsigned char *)mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (fd_map == MAP_FAILED) {
      close(fd);
      if(verbose)
        std::cerr << "Error mmapping the file" << std::endl;
      exit(1);
    }
  // Output the encoded data
  size_t bytes = 0;
  DataType tmp;
  size_t base = 0;
  size_t i, j;
  for(i = 0; i < ndata; i++) {
      if(save_id) {
          memcpy(&fd_map[bytes], id + i, sizeof(int));
          bytes += sizeof(int);
        }
      for(j = 0; j < points; j++) {
          tmp = data[base++];
          memcpy(&fd_map[bytes], &tmp, sizeof(DataType));
          bytes += sizeof(tmp);
        }
    }

  if(verbose)
    std::cout << "Read " << bytes << " byte(s) from memory and saved "
    << size << " byte(s) to disk" << std::endl;

  if (munmap(fd_map, size) == -1) {
      if(verbose)
        std::cerr << "Error un-mmapping the file" << std::endl;
    }
  close(fd);
}
