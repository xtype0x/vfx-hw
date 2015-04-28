#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>

#include "sift.h"
#include "image_stitch.h"

using namespace cv;
using namespace std;

ImageStitch::ImageStitch(char *filelist){
	int count = 0;
	char buf[100];
	Sift *sift;
	fstream fs(filelist, ios::in);
	while(fs.getline(buf,100)){
		string newStr = buf;
		filenames.push_back(newStr);
		sift = new Sift(buf,4,2);
		sift->do_sift();
		descs.push_back(sift->getDescriptors());
		delete sift;
		cout<<count++<<" load done!!"<<endl;
	}
	cout<<"loading complete"<<endl;
}

ImageStitch::~ImageStitch(){
	
}