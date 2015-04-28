#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <cmath>
#include <iostream>
#include <vector>
#include "image_stitch.h"
#include "sift.h"

using namespace cv;
using namespace std;


int main(int argc, char const *argv[])
{
	ImageStitch("list.txt");
	
	

	return 0;
}

