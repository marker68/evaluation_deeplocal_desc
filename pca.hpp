/*
 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
//
// Created by TuanNguyen on 2016/03/14.
//

#ifndef EVALUATION_DEEPLOCALDESC_PCA_HPP
#define EVALUATION_DEEPLOCALDESC_PCA_HPP

#include <iostream>
#include <algorithm>
#include <vector>
#include <cstring>
#include <cblas.h>
#include <lapacke.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

inline void vector_mean_columns (
    float *data,
    float *&v,
    size_t numRows,
    size_t dimension)
{
  size_t i, j;
  float tmp;
  v = (float *) ::operator new (dimension * sizeof (float));
  for (i = 0; i < dimension; i++)
    {
      tmp = 0.0;
#if defined(_OPENMP)  && _OPENMP >= 201307
      #pragma omp parallel for simd reduction(+:tmp)
#pragma ivdep
#elif defined(_OPENMP) && _OPENMP < 201307
#pragma omp parallel for reduction(+:tmp)
#endif
      for (j = 0; j < numRows; j++)
        {
          tmp += data[i + j * dimension];
        }

      v[i] = tmp / numRows;
    }
}

inline void vector_mean_rows (
    float *data,
    float *&v,
    size_t numColumns,
    size_t numRows)
{
  size_t i, j;
  v = (float *) ::operator new (numRows * sizeof (float));
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (i = 0; i < numRows; i++)
    {
      float tmp = 0.0;
      float *v_tmp = data + i * numColumns;
#if defined(_OPENMP)  && _OPENMP >= 201307
      #pragma omp simd reduction(+:tmp)
#pragma ivdep
#endif
      for (j = 0; j < numColumns; j++)
        {
          tmp += v_tmp[j];
        }
      v[i] = tmp / numColumns;
    }
}

inline void subtract_mean (
    float *&data,
    float *v,
    size_t numRows,
    size_t dimension)
{
  size_t i, j;
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (i = 0; i < numRows; i++)
    {
      float *v_tmp = data + i * dimension;
#if defined(_OPENMP)  && _OPENMP >= 201307
      #pragma omp simd
#pragma ivdep
#endif
      for (j = 0; j < dimension; j++)
        {
          v_tmp[j] -= v[j];
        }
    }
}

inline void cov (
    float *&v,
    float *&C,
    size_t n,
    size_t d)
{
  float *mv;
  vector_mean_columns (v, mv, n, d);
  subtract_mean (v, mv, n, d);

  // Compute the co-variance matrix C
  // Since E[v] = 0 now, so C=1/(n-1) * v' * v
  C = (float *) ::operator new (d * d * sizeof (float));
  cblas_sgemm (
      CblasRowMajor,
      CblasTrans,
      CblasNoTrans,
      d, d, n,
      1.0f / (n - 1),
      v, d,
      v, d,
      0.0f,
      C, d);

  ::delete mv;
}

inline void eigx (
    float *C,
    float *&w,
    float *&z,
    size_t d,
    size_t kl,
    size_t ku)
{
  int found, *isuppz;

  size_t k = ku - kl + 1;
  w = (float *) ::operator new (d * sizeof (float)); // eigenvalues
  z = (float *) ::operator new (d * d * sizeof (float)); // eigenvectors ASC
  isuppz = (int *) ::operator new ((d << 1) * sizeof (float));

  LAPACKE_ssyevr (
      LAPACK_ROW_MAJOR,
      'V',
      'I',
      'U',
      d,
      C,
      d,
      0.0f,
      0.0f,
      kl,
      ku,
      LAPACKE_slamch ('S'),
      &found,
      w,
      z,
      d,
      isuppz);

  ::delete isuppz;
}

inline void pca (
    float *&data,
    float *&pca,
    size_t N,
    size_t D,
    size_t L)
{
  size_t i, j;
  float *C;

  // Compute the co-variance
  cov (data, C, N, D);
  float *c = (float *) ::operator new (D * sizeof (float));
#if defined(_OPENMP)  && _OPENMP >= 201307
  #pragma omp parallel for simd
#pragma ivdep
#endif
  for (i = 0; i < D; i++)
    c[i] = sqrt (C[i * D + i]);


  // Solve eigenvalues-eigenvectors problem of C
  // Note: C is symmetric!
  float *w;
  if (L >= D) L = D;

  // Take L eigen-vectors in corresponding to L maximum eigenvalues
  // Note that, w is sorted in ASC order.
  eigx (C, w, pca, D, D - L + 1, D);


  // Convert to DESC order
  float *tmp = pca;
  for (i = 0; i < D; i++)
    {
      std::reverse (tmp, tmp + L);
      tmp += L;
    }

  // Rotate the matrix data by z: data = data * z
  tmp = (float *) ::operator new (N * L * sizeof (float));
#if defined(_OPENMP)
#pragma omp parallel for private(j)
#endif
  for (i = 0; i < N; i++)
    {
#if defined(_OPENMP)  && _OPENMP >= 201307
      #pragma omp parallel for simd
#pragma ivdep
#endif
      for (j = 0; j < D; j++)
        {
          data[i * D + j] /= c[j];
        }
    }

  cblas_sgemm (
      CblasRowMajor,
      CblasNoTrans,
      CblasNoTrans,
      N, L, D,
      1.0f,
      data, D,
      pca, L,
      0.0f,
      tmp, L);
  memcpy (data, tmp, N * L * sizeof (float));
}

inline size_t get_file_size (const char *filename)
{
  if (filename == nullptr)
    {
      return -1;
    }
  size_t size, start;
  FILE *fp;
  try
    {
      fp = fopen (filename, "rb");
      start = ftell (fp);
      fseek (fp, 0, SEEK_END);
      size = ftell (fp);
      fseek (fp, start, SEEK_SET);
      fclose (fp);
    }
  catch (std::exception &e)
    {
      return -1;
    }
  return size;
}

template<typename DataType>
inline size_t load_data (
    std::string fName,
    DataType *&data,
    int *&ids,
    int header,
    int d)
{
  const char *filename = fName.c_str ();
  // Precheck
  if (filename == nullptr)
    {
      return -1;
    }
  if (d <= 0)
    {
      return -1;
    }
  int fd = open (filename, O_RDONLY);
  if (fd < 0)
    {
      return -1;
    }

  size_t size = get_file_size (filename);
  unsigned char *mapped; // Read file as bytes
  /* Mapping the file */
  mapped = (unsigned char *) mmap (0, size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (mapped == MAP_FAILED)
    {
      return -1;
    }

  size_t i, j, base = 0, count = 0, count_id = 0;
  DataType f[1];
  int id[1];
  size_t d1 = d * sizeof (DataType) + header;
  size_t ss = header > sizeof (DataType) ? header : sizeof (DataType);
  unsigned char uc, buf[ss];
  size_t total_row = size / d1;
  try
    {
      data = (DataType *) ::operator new (total_row * d * sizeof (DataType));
      ids = (int *) ::operator new (total_row * sizeof (int));

      /* Load data */
      while (count < total_row * d && base < size)
        {
          // load header
          for (i = 0; i < header; i++)
            {
              buf[i] = mapped[base++];
            }
          memcpy (id, buf, sizeof (int));
          ids[count_id++] = id[0];
          // load data
          for (i = 0; i < d; i++)
            {
              for (j = 0; j < sizeof (DataType); j++)
                {
                  buf[j] = mapped[base++];
                }
              memcpy (f, buf, sizeof (DataType));
              data[count++] = f[0];
            }
        }
    }
  catch (std::exception &e)
    {
      return -1;
    }

  if (munmap (mapped, size) != 0)
    {
      return -1;
    }
  close (fd);

  return count / d;
}

template<typename DataType>
inline bool save_data (
    std::string fName,
    DataType *data,
    size_t ndata,
    size_t points,
    const int *id,
    bool save_id)
{
  const char *filename = fName.c_str ();
  size_t size;
  // Save the frame data
  int fd = open (filename, O_RDWR | O_CREAT | O_TRUNC, (mode_t) 0600); // file description
  if (fd < 0)
    {
      return false;
    }

  // The number of bytes to be written out.
  size = ndata * points * sizeof (DataType);
  if (save_id)
    size += ndata * sizeof (int);

  int result = lseek (fd, size, SEEK_SET);
  if (result == -1)
    {
      close (fd);
      return false;
    }

  int status = write (fd, "", 1);
  if (status != 1)
    {
      close(fd);
      return false;
    }

  unsigned char *fd_map = (unsigned char *) mmap (0, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (fd_map == MAP_FAILED)
    {
      close (fd);
      return false;
    }
  // Output the encoded data
  size_t bytes = 0;
  DataType tmp;
  size_t base = 0;
  size_t i, j;
  try
    {
      for (i = 0; i < ndata; i++)
        {
          if (save_id)
            {
              memcpy (&fd_map[bytes], id + i, sizeof (int));
              bytes += sizeof (int);
            }
          for (j = 0; j < points; j++)
            {
              tmp = data[base++];
              memcpy (&fd_map[bytes], &tmp, sizeof (DataType));
              bytes += sizeof (tmp);
            }
        }
    }
  catch (std::exception &e)
    {
      // logger
      return false;
    }

  if (munmap (fd_map, size) == -1)
    {
      close (fd);
      return false;
    }
  close (fd);
  return true;
}

#endif //EVALUATION_DEEPLOCALDESC_PCA_HPP
