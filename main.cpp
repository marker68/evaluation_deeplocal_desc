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
#include <iostream>
#include <pca.hpp>
#include <extract_patches.hpp>

using namespace std;
int main (int argc, char * argv[])
{
  int result;
  std::string command;
  std::string input_file;
  std::string output_file;
  unsigned int D, radius;

  while((result=getopt(argc,argv,"c:i:d:r:o:"))!=-1){
      switch(result){
          case 'c':
            command = std::string(optarg);
            break;

          case 'i':
            input_file = std::string(optarg);
            break;

          case 'd':
            D = atoi(optarg);
            break;

          case 'r':
            radius = atoi(optarg);
            break;

          case 'o':
            output_file = std::string(optarg);
            break;

          case ':':
            std::cout << result << " needs value" << std::endl;
            break;

          case '?':
            std::cout << "Unknown" << std::endl;
            break;
        }
    }

  if (command == "compute_pca")
    {
      std::cout << "We will learn PCA" << std::endl;
      float * data, * pc;
      int * id;
      // Load feature data
      size_t n = load_data (input_file,data,id,0,D);
      ::delete id;
      // Learn PCs: mean subtraction is already included.
      pca (data, pc, n, D, radius);
      // Save PCs data
      if (save_data (output_file, pc, D, radius, nullptr, false) < 0)
        {
          std::cerr << "Cannot save data" << std::endl;
        }
      ::delete data;
      ::delete pc;
    }
  else if (command == "extract_patches")
    {
      std::cout << "We will extract patches and save to output folders" << std::endl;
      int nframes;
      std::vector<cv::Mat> patches;
      extract_patches (input_file, patches, radius, nframes);
      for (int i = 0; i < patches.size (); ++i)
        {
          cv::imwrite (output_file + "/" + std::to_string (i+1) + ".jpg", patches.at (i));
        }
      patches.clear ();
    }

  return 0;
}
