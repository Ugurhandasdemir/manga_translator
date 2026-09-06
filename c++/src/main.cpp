#include "find_and_yolo.h"

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;

int main(int argc, char** argv) {

    string img_folder = "/mnt/windows/linux/Python/manga/test";

    vector<string> images = find_images(img_folder);

    cout << "Toplam " << images.size() << " resim bulundu" << endl;

    for (int i = 0; i < images.size(); i++) {
        cout << "[" << i + 1 << "/" << images.size() << "] "
             << images[i] << endl;
    }

    return 0;
}
