#include "find_and_yolo.h"

#include <filesystem>
#include <algorithm>

using namespace std;
namespace fs = std::filesystem;
 
vector<string> find_images(const string& klasor) {
    vector<string> images;

    for(const auto&kayit : fs::directory_iterator(klasor)) {
        if(!kayit.is_regular_file()) {
            continue;
        }

        string uzanti = kayit.path().extension().string();

        if(uzanti == ".jpg" || uzanti == ".jpeg" || uzanti == ".png"){
            images.push_back(kayit.path().string());
        }
    }
    sort(images.begin(), images.end());

    return images;
}

cv::dnn:Net load_model(const string& model_path) {
    cv::dnn::Net net = cv::dnn::readNetFromONNX(model_path);

    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_GPU);

    return net;
}

static cv::Mat letterbox(const cv::Mat& src, int size, float& scale, int& pad_x, int& pad_y){

    scale = min(float(size)/ src.cols, float(size)/src.rows);

    int new_w = int(src.cols * scale);
    int new_h = int(src.rows * scale);

    pad_x = (size - new_w) / 2;
    pad_y = (size - new_h) / 2;

    cv::Mat small;
    cv::resize(src, small, cv::Size(new_w, new_h));

    cv::Mat frame(size, size, CV_8UC3, cv::Scalar(114, 114, 144));
    small.copyTo(frame(cv::Rect(pad_x, pad_y, new_w, new_h)));

    return frame;
}

vector<Bubble> detect(cv::dnn::Net& net, const cv::Mat& img) {

    float scale;
    int pad_x, pad_y;
    cv::Mat frame = letterbox(img, INPUT_SIZE, scale, pad_x, pad_y);
    
    cv::Mat blob = cv::dnn:blobFromImage(
        frame,
        1.0/255.0,
        cv:Size(INPUT_SIZE, INPUT_SIZE),
        cv::Scalar(),
        true,
        false
    );

    net.setInput(blob);
    cv::Mat out = net.forward();

    

}

