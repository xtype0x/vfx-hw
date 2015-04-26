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
	char buf[100];
	for(int i = 0; i < 18; i++){
		sprintf(buf, "../test-case/grail%02d.jpg",i);
		Sift s(buf,4,2);
		s.do_sift();
		cout<<i<<": descriptor num:"<<s.getDescriptors().size()<<endl;
	}
	

	return 0;
}

