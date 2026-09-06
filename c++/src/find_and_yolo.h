#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

const int INPUT_SIZE = 1280;
const float CONF_THRESHOLD = 0.15f;
const float NMS_THRESHOLD = 0.50f;

struct Bubble {
    cv::Rect box;
    float conf;
    int class_id;
}


std::vector<std::string> find_images(const std::string& klasor);

cv::dnn::Net load_model(const std::string& model_path);

std::vector<Bubble> detect(cv::dnn::Net& net, const cv::Mat& img);

