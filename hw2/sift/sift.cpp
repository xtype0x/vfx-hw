#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>
#include <vector>

#include "sift.h"

using namespace cv;
using namespace std;

#define _USE_MATH_DEFINES

#define SIGMA_ANTIALIAS 0.5
#define SIGMA_PREBLUR 1.0
#define CURVATURE_THRESHOLD 5.0
#define CONTRAST_THRESHOLD 0.03
#define NUM_BINS 36
#define MAX_KERNEL_SIZE 20
#define FEATURE_WINDOW_SIZE 16
#define DESC_NUM_BINS 8
#define FVSIZE 128
#define	FV_THRESHOLD 0.2

Sift::Sift(const char* filename, unsigned o, unsigned i){
	srcImage = new Mat(imread(filename,0));
	octaves = o;
	intervals = i;

	generate_list();
}

Sift::Sift(Mat image, unsigned o, unsigned i){
	srcImage = new Mat(image);
	octaves = o;
	intervals = i;

	generate_list();
}

Sift::~Sift(){
	delete srcImage;
}

void Sift::do_sift(){

}

void Sift::generate_list(){
	//cout<< srcImage->size().width<<endl;
	gList = new Mat**[octaves];
	for(int i = 0; i < octaves; i++){

	}
}

void Sift::build_scale_space(){

}

void Sift::detect_extrema(){

}

