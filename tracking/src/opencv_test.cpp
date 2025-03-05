
// -- STL -- //
#include <random>
#include <numeric>

// -- Project -- //
#include <opencv2/opencv.hpp>

using namespace std;

int main(){
	// Draw data
	cv::Mat image(1000, 1000, CV_8UC3);
	image.setTo(0);
	cv::line(image, cv::Point(0, 0), cv::Point(1000, 1000), cv::Scalar(127, 127, 127), 2, CV_8U);
	cv::waitKey(0);
	
	cv::namedWindow("image", cv::WINDOW_NORMAL);
	cv::imshow("image", image);
	cv::waitKey(0);
}