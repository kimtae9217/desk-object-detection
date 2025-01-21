#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(void){
	//string image_path = "/Users/taewonkim/GitHub/desk-object-detection/mandrill.bmp";
	//Mat mat = imread(image_path);
	Mat mat = imread("mandrill.bmp");
	imshow("pams", mat);
	waitKey(0);

	return 0;
}
