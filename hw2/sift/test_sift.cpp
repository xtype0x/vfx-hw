#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <cmath>
#include <iostream>
#include <vector>
#include "sift.h"

using namespace cv;
using namespace std;


int main(int argc, char const *argv[])
{
	Mat image = imread("../test-case/grail00.jpg",0);

	if(!image.data){//check if image is created
		cerr<<"The image is not found"<<endl;
		exit(1);
	}

	Sift s("../test-case/grail00.jpg",4,2);
	s.do_sift();

	return 0;
}

