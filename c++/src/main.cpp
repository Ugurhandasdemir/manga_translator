#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>

using namespace std;
namespace fs = std::filesystem;


int main(int argc, char** argv){

    string img_folder = "/mnt/windows/linux/Python/manga/test";
    vector<string> images;

    for (const auto& kayit : fs::directory_iterator(img_folder)){
        
        if(!kayit.is_regular_file()){
                     continue;
        }

        string uzanti = kayit.path().extension().string();

        if(uzanti == ".jpg" || uzanti == ".jpge" || uzanti == ".png"){
            images.push_back(kayit.path().string());
        }
    }

        for (int i = 0; i < images.size(); i++){
         cout << "[" << i + 1 << "/" << images.size() << "] "
             << images[i] << endl;
        }
    }

