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

#ifndef EVALUATION_DEEPLOCALDESC_EXTRACT_PATCHES_HPP
#define EVALUATION_DEEPLOCALDESC_EXTRACT_PATCHES_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
// Only support OpenCV 3.x!
#include <opencv2/xfeatures2d.hpp>
#include <fstream>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif

inline bool extract_patches (
    const std::string &filename,
    std::vector<cv::Mat> &patches, // patches will be stored here
    const int &radius,
    int &nframes
)
{
  try
    {
      // Read image data
      cv::Mat img = cv::imread (filename);
      cv::Mat img_gray;
      cv::cvtColor (img,img_gray,CV_BGR2GRAY);
      std::vector<cv::KeyPoint> keypoints;
      cv::Ptr<cv::Feature2D> feature2d = cv::xfeatures2d::SIFT::create (); // Extract all keypoints with DoG-Lowe's SIFT
      feature2d->detect (img_gray, keypoints);

      // Restore data in array format
      nframes = keypoints.size ();

      int i;
      int st_x, st_y, ed_x, ed_y;
      for (i = 0; i < nframes; i++)
        {
          st_x = (keypoints[i].pt.x - radius >= 0) ? (keypoints[i].pt.x - radius) : 0;
          st_y = (keypoints[i].pt.y - radius >= 0) ? (keypoints[i].pt.y - radius) : 0;
          ed_x = (keypoints[i].pt.x + radius < img.cols) ? (keypoints[i].pt.x + radius) : (img.cols - 1);
          ed_y = (keypoints[i].pt.y + radius < img.rows) ? (keypoints[i].pt.y + radius) : (img.rows - 1);
          patches.push_back(img.colRange (st_x, ed_x).rowRange (st_y, ed_y));
        }

      // Clean up
      img.release ();
      img_gray.release ();
      keypoints.clear ();
      feature2d->clear ();
    }
  catch (cv::Exception &e)
    {
      return false;
    }
  catch (std::exception &e)
    {
      return false;
    }
  catch (...)
    {
      return false;
    }

  return true;
}

#endif //EVALUATION_DEEPLOCALDESC_EXTRACT_PATCHES_HPP
