#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <list>
#include <cmath>
#include <cstring>

using namespace cv;


Mat transformation(256, 9, CV_32FC1);
Mat average(1, 256, CV_32FC1);
cv::Ptr<ml::SVM> svm = ml::SVM::create();


class Screener {
    private:
        static int clearThreshold;

    public:
        static bool judge(Mat rawData) {
            //Initial status.
            Mat img;
            int height;
            int width;
            //Import.
            cvtColor(rawData, img, CV_RGB2GRAY);
            width = img.rows;
            height = img.cols;
            //Convert to 2D matrix.
            int intValue [height][width];
            cv::Mat_<cv::Vec3b>::iterator data = img.begin<cv::Vec3b>();
            for(int i = 0; i< height; i++) {
                for(int j = 0; j < width; j++) {
                    intValue[i][j] = (int) (*data++)[0];
                }
            }
            //Apply Sobel on x and y directions and add together.
            Mat grad_x, grad_y;
            Mat abs_grad_x, abs_grad_y, dst;
            cv::Sobel(img, grad_x, CV_16S, 1, 0, 3, 1, 1);
            cv::convertScaleAbs(grad_x, abs_grad_x);
            cv::Sobel(img, grad_y, CV_16S, 0, 1, 3, 1, 1);
            cv::convertScaleAbs(grad_y, abs_grad_y);
            cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst);
            //Create a look-up table.
            cv::Mat_<cv::Vec3b>::iterator it = dst.begin<cv::Vec3b>();
            int counter = 0; //Record the total number of pixels involved.
            int res = 0;
            int sub[(height << 1) + 1][(width << 1) + 1];
            memset(sub, 0, sizeof(sub));
            for(int i = 1; i < width; i++) {
                sub[1][i << 1] = abs(intValue[0][i] - intValue[0][i - 1]);
            }
            for(int i = 1; i < height; i++) {
                sub[i << 1][1] = abs(intValue[i][0] - intValue[i - 1][0]);
                for(int j = 1; j < width; j++) {
                    sub[(i << 1) + 1][j << 1] = abs(intValue[i][j] - intValue[i - 1][j]);
                    sub[i << 1][(j << 1) + 1] = abs(intValue[i][j] - intValue[i][j - 1]);
                    sub[i << 1][j << 1] = (abs(intValue[i][j] - intValue[i - 1][j - 1]));
                }
            }
            //Compute clarity.
            for(int i = 0; i < height; i++) {
                for(int j = 0; j < width; j++) {
                    if(((int) (*it++)[0] >= 20)) {
                        res += (((sub[i << 1][(j << 1) + 1] +
                                  sub[(i << 1) + 2][(j << 1) + 1] +
                                  sub[(i << 1) + 1][j << 1] +
                                  sub[(i << 1) + 1][(j << 1) + 2]) << 1) +
                                (sub[i << 1][j << 1] +
                                 sub[(i << 1) + 2][j << 1] +
                                 sub[i << 1][(j << 1) + 2] +
                                 sub[(i << 1) + 2][(j << 1) + 2]));
                        counter++;
                    }
                }
            }
            //If clarity is bad, return a blank image.
            if((double)res / counter < clearThreshold) {
                return false;
            }
            //Crate a histogram.
            cv::Mat dstHist;
            int dims = 1;
            float hranges[] = {0, 255};
            const float *ranges[] = {hranges};
            int size = 256;
            int channels = 0;
            float weight = width * height;
            calcHist(&img, 1, &channels, Mat(), dstHist, dims, &size, ranges);
            //Transpose.
            Mat converted;
            cv::transpose(dstHist, converted);
            //Normalize.
            for(int i = 0; i < 256; i++) {
                converted.at<float>(0,i) /= weight;
            }
            //Mean->0
            Mat centerfree;
            cv::subtract(converted, average, centerfree);
            //Apply PCA.
            Mat reduced(1, 9, CV_32FC1);
            reduced = centerfree * transformation;
            //SVM process
            if((svm->predict(reduced)) == 0) {
                return false;
            }
            return true;
        }

        static void setClearThreshold(int input) {
            clearThreshold = input;
        }
};

int Screener::clearThreshold = 45;

int main() {
    std::cout << "Hello!" << std::endl;
    //Import average and transformation matrix.
    cv::FileStorage fs("mat.xml",cv::FileStorage::READ);
    fs["transformation"] >> transformation;
    fs["average"] >> average;
    //Import SVM model.
    svm = cv::ml::SVM::load("test_svm.xml");
    //Initialize.
    std::string input;
    std::string outputFileName;
    std::string outputPath;
    int accumulator = 0;
    int inputnum;
    std::cout << "Input the path for output." << std::endl;
    std::cin  >> outputPath;
    std::cout << "Input the threshold for clarity." << std::endl;
    std::cin  >> inputnum;
    Screener::setClearThreshold(inputnum);
    while(1) {
        std::cout << "Input \"1\" means you offer us a path to a file contains with a list of paths to images." << std::endl;
        std::cout << "Input \"2\" means you type paths one by one and quit with \"Q\"." << std::endl;
        std::cout << "Input \"3\" to terminate this program." << std::endl;
        std::cin >> inputnum;
        if(inputnum == 1) {
            std::cin >> input;
            std::ifstream in(input);
            std::string filePath;
            while(in >> filePath) {
                Mat src = imread(filePath);
                if(src.empty()) {
                    continue;
                }
                if(Screener::judge(src)) {
                    outputFileName = outputPath + "/" + std::to_string(++accumulator) + ".png";
                    imwrite(outputFileName, src);
                }
            }
            continue;
        }
        if(inputnum == 2) {
            while(1) {
                std::cin >> input;
                if(input == "Q") {
                    break;
                }
                Mat src = imread(input);
                if (src.empty()) {
                    continue;
                }
                if (Screener::judge(src)) {
                    outputFileName = outputPath + "/" + std::to_string(++accumulator) + ".png";
                    imwrite(outputFileName, src);
                }
            }
            continue;
        }
        if(inputnum == 3) {
            break;
        }
    }
    return 0;
}