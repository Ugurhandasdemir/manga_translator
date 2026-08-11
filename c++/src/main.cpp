#include<opencv2/opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;

int main(int argc, char** argv){

    Mat img = imread("/mnt/windows/linux/Python/manga/test/1.jpg",1);

    if(img.empty()){
        cout<<"Could not read image "<<endl;
        return 1;
    }

    imshow("Display window", img);
    waitKey(0);
    destroyAllWindows();
}