#ifndef IMAGE_STITCH_H
#define IMAGE_STITCH_H

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>
#include "kdtree.h"
#include "sift.h"

using namespace cv;
using namespace std;

class ImageStitch
{
public:
	ImageStitch(char* filelist);
	~ImageStitch();
	void stitch();
private:
	void findKNN();
	vector<string> filenames;

};

#endif