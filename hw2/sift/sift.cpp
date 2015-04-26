#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>
#include <vector>
#include <iostream>

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
	//read image, pixel value is between 0~1.0
	srcImage = new Mat(imread(filename,CV_LOAD_IMAGE_GRAYSCALE));
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
	for (int i = 0; i < octaves; ++i){
		for(int j = 0; j < intervals+3; j++)delete gList[i][j];
		delete [] gList[i];
		for(int j = 0; j < intervals+2; j++)delete dogList[i][j];
		delete [] dogList[i];
		for(int j = 0; j < intervals; j++)delete extrema[i][j];
		delete [] extrema[i];
		delete [] absSigma[i];
	}
	delete [] gList;
	delete [] dogList;
	delete [] extrema;
	delete [] absSigma;
}

void Sift::do_sift(){
	build_scale_space();
	detect_extrema();
	assign_orientations();
	extract_keypoint_descriptors();
}

void Sift::generate_list(){
	//cout<< srcImage->size().width<<endl;
	gList = new Mat**[octaves];
	dogList = new Mat**[octaves];
	extrema = new Mat**[octaves];
	absSigma = new double*[octaves];

	for(int i = 0; i < octaves; i++){
		gList[i] = new Mat*[intervals+3];
		for(int j = 0; j < intervals+3; j++)gList[i][j] = new Mat();
		dogList[i] = new Mat*[intervals+2];
		for(int j = 0; j < intervals+2; j++)dogList[i][j] = new Mat();
		extrema[i] = new Mat*[intervals];
		for(int j = 0; j < intervals; j++)extrema[i][j] = new Mat();
		absSigma[i] = new double[intervals+3];
	}
}

//build scale space and DoG images
void Sift::build_scale_space(){
	//create doubleing point grayscale image
	Mat *grayImg = new Mat(srcImage->size(),CV_64FC1);
	Mat *tempImg = new Mat(*srcImage);

	//copy src image to gray img, and change value to 0~1.0
	for(int i = 0; i < grayImg->rows; i++)
		for(int j = 0; j < grayImg->cols; j++)
			grayImg->at<double>(i,j) = (double)srcImage->at<uchar>(i,j) / 255.0;

	//blur gray image with sigma 0.5
	GaussianBlur(*grayImg,*grayImg,Size(0,0),SIGMA_ANTIALIAS);

	//create a double dimension image and resize gray img to gList[0][0]
	gList[0][0]->create(grayImg->rows * 2, grayImg->cols * 2, CV_64FC1);
	pyrUp(*grayImg, *gList[0][0],gList[0][0]->size());

	//preblur gList[0][0]
	GaussianBlur(*gList[0][0],*gList[0][0],Size(0,0),SIGMA_PREBLUR);

	double initSigma = sqrt(2.0f);
	absSigma[0][0] = initSigma * 0.5;

	double sigma, sigma_f;
	int currentRows, currentCols;
	for(int i = 0; i < octaves; i++){
		//reset sigma
		sigma = initSigma;
		currentRows = gList[i][0]->rows;
		currentCols = gList[i][0]->cols;

		for(int j = 1; j < intervals + 3; j++){
			//init gList[i][j]
			gList[i][j]->create(currentRows, currentCols, CV_64FC1);
			//calculate sigma to blur current image
			sigma_f = sqrt(pow(2.0,2.0/intervals)-1) * sigma;
			sigma = pow(2.0,1.0/intervals) * sigma;
			absSigma[i][j] = sigma * 0.5 * pow(2.0f, (double)i);

			GaussianBlur(*gList[i][j-1],*gList[i][j],Size(0,0),sigma_f);

			//calculate the DoG image
			*dogList[i][j-1] = *gList[i][j-1] - *gList[i][j];

		}

		//create image for next iterative
		if(i < octaves -1){
			gList[i+1][0]->create(currentRows/2, currentCols/2, CV_64FC1);
			pyrDown(*gList[i][0], *gList[i+1][0], gList[i+1][0]->size());
			absSigma[i+1][0] = absSigma[i][intervals];
		}
	}
	delete grayImg;
	delete tempImg;
}

void Sift::detect_extrema(){
	//variables
	double curv_ratio;
	double curv_threshold = (CURVATURE_THRESHOLD+1)*(CURVATURE_THRESHOLD+1)/CURVATURE_THRESHOLD;
	Mat *middle, *up, *down;
	int scale, kp_num=0, kp_reject_num=0;//scale and keypoint variable
	double dxx, dyy, dxy, trH, detH;

	//detect extrema in Dog images
	for(int i = 0; i < octaves; i++){
		scale = (int)pow(2.0, (double)i);

		for(int j = 1; j <= intervals; j++){
			//initialize extrema image and image middle,up and down
			extrema[i][j-1]->create(gList[i][0]->rows, gList[i][0]->cols, CV_8U);
			extrema[i][j-1]->setTo(Scalar(0));
			middle = new Mat(*dogList[i][j]);
			up = new Mat(*dogList[i][j+1]);
			down = new Mat(*dogList[i][j-1]);

			for(int xi = 1; xi < dogList[i][j]->rows; xi++){
				for(int yi = 1; yi < dogList[i][j]->cols; yi++){
					//check if current point is local maxima or minimum
					bool is_local = false;
					double current_point = middle->at<double>(xi,yi);

					//check maximum
					if(	current_point > middle->at<double>(xi-1,yi) &&
						current_point > middle->at<double>(xi+1,yi) &&
						current_point > middle->at<double>(xi-1,yi-1) &&
						current_point > middle->at<double>(xi+1,yi-1) &&
						current_point > middle->at<double>(xi-1,yi+1) &&
						current_point > middle->at<double>(xi+1,yi+1) &&
						current_point > middle->at<double>(xi,yi-1) &&
						current_point > middle->at<double>(xi,yi+1) &&
						current_point > down->at<double>(xi,yi) &&
						current_point > down->at<double>(xi-1,yi) &&
						current_point > down->at<double>(xi+1,yi) &&
						current_point > down->at<double>(xi-1,yi-1) &&
						current_point > down->at<double>(xi+1,yi-1) &&
						current_point > down->at<double>(xi-1,yi+1) &&
						current_point > down->at<double>(xi+1,yi+1) &&
						current_point > down->at<double>(xi,yi-1) &&
						current_point > down->at<double>(xi,yi+1) &&
						current_point > up->at<double>(xi,yi) &&
						current_point > up->at<double>(xi-1,yi) &&
						current_point > up->at<double>(xi+1,yi) &&
						current_point > up->at<double>(xi-1,yi-1) &&
						current_point > up->at<double>(xi+1,yi-1) &&
						current_point > up->at<double>(xi-1,yi+1) &&
						current_point > up->at<double>(xi+1,yi+1) &&
						current_point > up->at<double>(xi,yi-1) &&
						current_point > up->at<double>(xi,yi+1) 
					){
						is_local = true;
						kp_num++;
						extrema[i][j-1]->at<uchar>(xi,yi) = 255;
					}else if(	current_point < middle->at<double>(xi-1,yi) &&
						current_point < middle->at<double>(xi+1,yi) &&
						current_point < middle->at<double>(xi-1,yi-1) &&
						current_point < middle->at<double>(xi+1,yi-1) &&
						current_point < middle->at<double>(xi-1,yi+1) &&
						current_point < middle->at<double>(xi+1,yi+1) &&
						current_point < middle->at<double>(xi,yi-1) &&
						current_point < middle->at<double>(xi,yi+1) &&
						current_point < down->at<double>(xi,yi) &&
						current_point < down->at<double>(xi-1,yi) &&
						current_point < down->at<double>(xi+1,yi) &&
						current_point < down->at<double>(xi-1,yi-1) &&
						current_point < down->at<double>(xi+1,yi-1) &&
						current_point < down->at<double>(xi-1,yi+1) &&
						current_point < down->at<double>(xi+1,yi+1) &&
						current_point < down->at<double>(xi,yi-1) &&
						current_point < down->at<double>(xi,yi+1) &&
						current_point < up->at<double>(xi,yi) &&
						current_point < up->at<double>(xi-1,yi) &&
						current_point < up->at<double>(xi+1,yi) &&
						current_point < up->at<double>(xi-1,yi-1) &&
						current_point < up->at<double>(xi+1,yi-1) &&
						current_point < up->at<double>(xi-1,yi+1) &&
						current_point < up->at<double>(xi+1,yi+1) &&
						current_point < up->at<double>(xi,yi-1) &&
						current_point < up->at<double>(xi,yi+1) 
					){
						is_local = true;
						kp_num++;
						extrema[i][j-1]->at<uchar>(xi,yi) = 255;
					}
					//check contrast
					if(is_local && fabs(current_point) < CONTRAST_THRESHOLD){
						extrema[i][j-1]->at<uchar>(xi,yi) = 0;
						kp_num--;kp_reject_num++;
						is_local = false;
					}
					//check edge
					if(is_local){
						dxx = middle->at<double>(xi,yi-1) + middle->at<double>(xi,yi+1) - 2 * middle->at<double>(xi,yi);
						dyy = middle->at<double>(xi-1,yi) + middle->at<double>(xi+1,yi) - 2 * middle->at<double>(xi,yi);
						dxy = (middle->at<double>(xi-1,yi-1) + middle->at<double>(xi+1,yi+1) - middle->at<double>(xi+1,yi-1) - middle->at<double>(xi-1,yi+1))/4;

						trH = dxx + dyy;
						detH = dxx*dyy - dxy*dxy;
						curv_ratio = trH*trH/detH;

						if(detH < 0 || curv_ratio < curv_threshold){
							extrema[i][j-1]->at<uchar>(xi,yi) = 0;
							kp_num--;kp_reject_num++;
							is_local=false;
						}
					}
				}
			}
			delete middle;
			delete up;
			delete down;
		}
	}
	keypoint_num = kp_num;
	// cout<<kp_num<<endl;
	// namedWindow("gg");
	// imshow("gg",extrema[1][0]);
	// waitKey(0);
}

void Sift::assign_orientations(){
	//initialize images holdingmagnitude and direction of gradient for all blurred images
	Mat ***magnitude = new Mat**[octaves];
	Mat ***orientation = new Mat**[octaves];
	for(int i = 0; i < octaves; i++){
		magnitude[i] = new Mat*[intervals];
		orientation[i] = new Mat*[intervals];
	}

	double dx, dy, ori;
	for(int i = 0; i < octaves; i++){
		for(int j = 1; j <= intervals; j++){
			magnitude[i][j-1] = new Mat(gList[i][j]->rows, gList[i][j]->cols, CV_64FC1);
			magnitude[i][j-1]->setTo(Scalar(0.0));
			orientation[i][j-1] = new Mat(gList[i][j]->rows, gList[i][j]->cols, CV_64F);
			orientation[i][j-1]->setTo(Scalar(0.0));

			//calculate magnitude and orientation
			for(int xi = 1; xi < gList[i][j]->size().width-1; xi++){
				for(int yi = 1; yi < gList[i][j]->size().height-1; yi++){
					dx = gList[i][j]->at<double>(yi,xi+1) - gList[i][j]->at<double>(yi,xi-1);
					dy = gList[i][j]->at<double>(yi+1, xi) - gList[i][j]->at<double>(yi-1, xi);

					//save magnitude and orientation
					magnitude[i][j-1]->at<double>(yi,xi) = sqrt(dx*dx + dy*dy);
					ori = atan(dy/dx);
					orientation[i][j-1]->at<double>(yi,xi) = ori;
				}
			}
		}
	}
	//histograms
	Mat *weight, *mask;
	Mat *X, *xInv;
	int scale, half_size, sample_degree, max_peak_index;
	double sample, max_peak;
	double x1, x2, x3, y1, y2, y3, x0, x0_n;
	double b[3];
	double* hist_orient = new double[NUM_BINS];

	for(int i = 0; i < octaves; i++){
		scale = (unsigned)pow(2.0,(double)i);

		for(int j = 1; j < intervals+1; j++){
			weight = new Mat(gList[i][0]->rows, gList[i][0]->cols, CV_64FC1);
			mask = new Mat(gList[i][0]->rows, gList[i][0]->cols, CV_64FC1);
			GaussianBlur(*magnitude[i][j-1], *weight, Size(0,0), 1.5*absSigma[i][j]);
			mask->setTo(Scalar(0.0));

			//get half size of gaussian kernel
			half_size = (int)((1.5*absSigma[i][j]-0.8)/0.3+1.5);
			if(half_size<0)half_size=0;

			//iterate through all points of extrema
			for(int xi = 0; xi < gList[i][0]->rows; xi++){
				for(int yi = 0; yi < gList[i][0]->cols; yi++){
					//if keypoint
					if(extrema[i][j-1]->at<uchar>(xi,yi) != 0){
						//reset histogram
						for(int k = 0; k < NUM_BINS; k++)hist_orient[k]=0.0;

						for(int kk = -half_size; kk <= half_size; kk++){
							for(int tt = -half_size; tt <= half_size; tt++){
								if(yi+tt < 0 || yi+tt >= gList[i][0]->cols || xi+kk < 0 || xi+kk >= gList[i][0]->cols)
									continue;

								//sample orientation
								sample = orientation[i][j-1]->at<double>(xi+kk,yi+tt);
								//if(sample <= -M_PI || sample > M_PI)

								sample+=M_PI;

								sample_degree = sample * 180 / M_PI;
								hist_orient[(int)sample_degree/(360/NUM_BINS)] += weight->at<double>(xi+kk, yi+tt);
								mask->at<double>(xi+kk, yi+tt) = 1.0;
							}
						}

						max_peak = hist_orient[0];
						max_peak_index = 0;
						for(int k = 1; k < NUM_BINS; k++){
							if(hist_orient[k] > max_peak){
								max_peak = hist_orient[k];
								max_peak_index = k;
							}
						}

						//list of magnitudes and orientations at the current extrema
						vector<double> orien;
						vector<double> mag;
						for(int k = 0; k< NUM_BINS; k++){
							//check if peak is good
							if(hist_orient[k] > 0.8*max_peak){
								x1 = k-1;
								x2 = k;
								y2 = hist_orient[k];
								x3 = k+1;

								if(k==0){
									y1 = hist_orient[NUM_BINS-1];
									y3 = hist_orient[1];
								}else if(k == NUM_BINS-1){
									y1 = hist_orient[NUM_BINS-1];
									y3 = hist_orient[0];
								}else{
									y1 = hist_orient[k-1];
									y3 = hist_orient[k+1];
								}

								X = new Mat(3,3,CV_64FC1);
								X->at<double>(0,0)=x1*x1; X->at<double>(1,0)=x1; X->at<double>(2,0)=1;
								X->at<double>(0,1)=x2*x2; X->at<double>(1,1)=x2; X->at<double>(2,1)=1;
								X->at<double>(0,2)=x3*x3; X->at<double>(1,2)=x3; X->at<double>(2,2)=1;

								xInv = new Mat(X->inv());

								for(int bb = 0; bb < 3; bb++)
									b[bb] = xInv->at<double>(bb,0)*y1 + xInv->at<double>(bb,0)*y2 + xInv->at<double>(bb,0)*y3;

								x0 = -b[1] / b[0] / 2;

								if(fabs(x0) > 2*NUM_BINS) x0 = x2;
								while(x0 < 0)x0 += NUM_BINS;
								while(x0 >= NUM_BINS)x0 -= NUM_BINS;

								x0_n = x0*(2*M_PI/NUM_BINS);
								x0_n -= M_PI;

								orien.push_back(x0_n);
								mag.push_back(hist_orient[k]);
								delete X;
								delete xInv;
							}
						}

						//save the keypoint
						keypoints.push_back(Keypoint(xi*scale/2, yi*scale/2, mag, orien, i*intervals+j-1));
					}
				}
			}

			delete mask;
			delete weight;
		}
	}
	
	//release allocate memory
	for(int i = 0; i < octaves; i++){
		for(int j = 0; j < intervals; j++){
			delete magnitude[i][j];
			delete orientation[i][j];
		}
		delete [] magnitude[i];
		delete [] orientation[i];
	}
	delete [] magnitude;
	delete [] orientation;
	delete [] hist_orient;
	//cout<<"kp size"<<keypoints.size()<<endl;
}

void Sift::extract_keypoint_descriptors(){
	//initialize interpolated magnitude and orientation
	Mat ***interpolatedMagnitude = new Mat**[octaves];
	Mat ***interpolatedOrientation = new Mat**[octaves];
	for(int i = 0; i < octaves; i++){
		interpolatedMagnitude[i] = new Mat*[intervals];
		interpolatedOrientation[i] = new Mat*[intervals];
	}

	int _rows, _cols, iii, jjj;
	double dx, dy;
	Mat *tempImg;
	for(int i = 0; i < octaves; i++){
		for(int j = 1; j <= intervals; j++){
			_rows = gList[i][j]->rows;
			_cols = gList[i][j]->cols;
			tempImg = new Mat(_rows*2, _cols*2, CV_64FC1);
			tempImg->setTo(Scalar(0.0));

			pyrUp(*gList[i][j], *tempImg, tempImg->size());
			interpolatedMagnitude[i][j-1] = new Mat(_rows+1, _cols+1, CV_64FC1);
			interpolatedMagnitude[i][j-1]->setTo(Scalar(0.0));
			interpolatedOrientation[i][j-1] = new Mat(_rows+1, _cols+1, CV_64FC1);
			interpolatedOrientation[i][j-1]->setTo(Scalar(0.0));
	

			for(double ii = 1.5; ii < _cols - 1.5; ii++){
				for(double jj = 1.5; jj < _rows - 1.5; jj++){
					dx = (gList[i][j]->at<double>(jj, ii+1.5) + gList[i][j]->at<double>(jj, ii+0.5) - gList[i][j]->at<double>(jj, ii-1.5) - gList[i][j]->at<double>(jj, i-0.5)) / 2;
					dy = (gList[i][j]->at<double>(jj+1.5, ii) + gList[i][j]->at<double>(jj+0.5, ii) - gList[i][j]->at<double>(jj-1.5, ii) - gList[i][j]->at<double>(jj-0.5, i)) / 2;

					iii = ii+1;
					jjj = jj+1;
					interpolatedMagnitude[i][j-1]->at<double>(jjj,iii) = sqrt(dx*dx+dy*dy);
					interpolatedOrientation[i][j-1]->at<double>(jjj,iii) = (atan2(dy,dx)==M_PI)? -M_PI:atan2(dy,dx);
				}
			}

			for(int k = 0; k < _cols+1; k++){
				interpolatedMagnitude[i][j-1]->at<double>(0,k) = 0;
				interpolatedMagnitude[i][j-1]->at<double>(_rows,k) = 0;
				interpolatedOrientation[i][j-1]->at<double>(0,k) = 0;
				interpolatedOrientation[i][j-1]->at<double>(_rows,k) = 0;
			}
			for(int k = 0; k < _rows+1; k++){
				interpolatedMagnitude[i][j-1]->at<double>(k,0) = 0;
				interpolatedMagnitude[i][j-1]->at<double>(k,_cols) = 0;
				interpolatedOrientation[i][j-1]->at<double>(k,0) = 0;
				interpolatedOrientation[i][j-1]->at<double>(k,_cols) = 0;
			}

			delete tempImg;
		}
	}
	

	
	vector<double> hist(DESC_NUM_BINS, 0.0);
	Mat *weight = new Mat(FEATURE_WINDOW_SIZE, FEATURE_WINDOW_SIZE, CV_64FC1);
	int scale, ii, jj, width, height;
	int half_size, sample_orien_degree, bin;
	int start_i, start_j,limit_i, limit_j;
	double kp_xi, kp_yi, desc_xi, desc_yi, main_mag, main_orien, sample_orien, bin_f, norm;
	vector<double> mag, orien;

	for(int i = 0; i <keypoint_num; i++){
		scale = keypoints[i].scale;
		kp_xi = keypoints[i].xi;
		kp_yi = keypoints[i].yi;
		desc_xi = kp_xi, desc_yi = kp_yi;

		ii = (int)(kp_xi*2) / (int)(pow(2.0, (double)scale/intervals));
		jj = (int)(kp_yi*2) / (int)(pow(2.0, (double)scale/intervals));

		width = gList[scale/intervals][0]->cols;
		height = gList[scale/intervals][0]->rows;

		mag = keypoints[i].magnitudes;
		orien = keypoints[i].orientations;

		//find maximum
		main_mag = mag[0];
		main_orien = orien[0];
		for(int j = 0; j < mag.size(); j++){
			if(mag[j] > main_mag){
				main_mag = mag[j]; main_orien = orien[j];
			}
		}
		half_size = FEATURE_WINDOW_SIZE / 2;
		Mat *G_mat = build_interpolated_gaussian_table(FEATURE_WINDOW_SIZE, 0.5*FEATURE_WINDOW_SIZE);
		vector<double> fv(FVSIZE);
		for(int j = 0; j < FEATURE_WINDOW_SIZE; j++){
			for(int k = 0; k < FEATURE_WINDOW_SIZE; k++){
				if(ii+j+1 < half_size || ii+j+1 > width+half_size || jj+k+1 < half_size || jj+k+1 > half_size+height)
					weight->at<double>(k,j) = 0.0;
				else
					weight->at<double>(k,j) = G_mat->at<double>(k,j) * interpolatedMagnitude[scale/intervals][scale%intervals]->at<double>(jj+k+1-half_size, ii+j+1-half_size);
			}
		}
		delete G_mat;
		//splitting into 16 4*4 blocks
		for(int j = 0; j < FEATURE_WINDOW_SIZE/4; j++){
			for(int k = 0; k < FEATURE_WINDOW_SIZE/4; k++){
				for(int t = 0; t < DESC_NUM_BINS; t++) hist[t] = 0.0;

				start_i = ii - half_size + 1 + (half_size * j / 2);
				start_j = jj - half_size + 1 + (half_size * k / 2);
				limit_i = ii + half_size / 2 * (j-1);
				limit_j = jj + half_size / 2 * (k-1);

				for(int si = start_i; si <= limit_i; si++){
					for(int sj = start_j; sj <= limit_j; sj++){
						if(si < 0 || si > width || sj < 0 || sj > height)continue;

						sample_orien = interpolatedOrientation[scale/intervals][scale%intervals]->at<double>(sj,si);
						sample_orien -= main_orien;
						while(sample_orien < 0) sample_orien += (2 * M_PI);
						while(sample_orien > 2 * M_PI) sample_orien -= (2 * M_PI);

						sample_orien_degree = sample_orien*180/M_PI;
						bin = sample_orien_degree/(360/DESC_NUM_BINS);
						bin_f = (double)sample_orien_degree/(double)(360/DESC_NUM_BINS);
						hist[bin] += (1- fabs(bin_f-bin-0.5)) * weight->at<double>(sj+half_size-jj-1,si+half_size-ii-1);
					}
				}
				for(int t = 0; t < DESC_NUM_BINS; t++)
					fv[(j*FEATURE_WINDOW_SIZE/4+k)*DESC_NUM_BINS + t] = hist[t];
			}
		}

		//normalize
		norm = 0.0;
		for(int j = 0; j < FVSIZE; j++) norm += pow(fv[j],2.0);
		norm = sqrt(norm);
		for(int j = 0; j < FVSIZE; j++){ 
			fv[j] /= norm;
			if(fv[j] > FV_THRESHOLD) fv[j] = FV_THRESHOLD;
		}
		//normalize again
		norm = 0.0;
		for(int j = 0; j < FVSIZE; j++) norm += pow(fv[j],2.0);
		norm = sqrt(norm);
		for(int j = 0; j < FVSIZE; j++)fv[j] /= norm;

		descriptors.push_back(Descriptor(desc_xi, desc_yi, fv));
	}
	delete weight;

	//cout<<"dd:"<<descriptors.size()<<endl;
	//release memory
	for(int i = 0; i < octaves; i++){
		for(int j=0; j < intervals; j++){
			delete interpolatedMagnitude[i][j];
			delete interpolatedOrientation[i][j];
		}
		delete [] interpolatedMagnitude[i];
		delete [] interpolatedOrientation[i];
	}
	delete [] interpolatedMagnitude;
	delete [] interpolatedOrientation;
}

Mat* Sift::build_interpolated_gaussian_table(unsigned size, double sigma){
	double half_kernel_size = size/2 - 0.5;

	double sog=0.0;
	Mat* ret = new Mat(size, size, CV_64FC1);

	double temp=0.0;
	for(int i = 0; i < size; i++){
		for(int j = 0; j < size; j++){
			temp = gaussian2D(i-half_kernel_size, j-half_kernel_size, sigma);
			ret->at<double>(j,i) = temp;
			sog+=temp;
		}
	}
	for(int i = 0; i < size; i++){
		for(int j = 0; j < size; j++){
			ret->at<double>(j,i) = 1.0/sog * ret->at<double>(j,i);
		}
	}
	return ret;
}

// gaussian2D
// Returns the value of the bell curve at a (x,y) for a given sigma
double Sift::gaussian2D(double x, double y, double sigma){
	return 1.0/(2*M_PI*sigma*sigma) * exp(-(x*x+y*y)/(2.0*sigma*sigma));
}

