#include <stdio.h>
#include "opencv2/opencv.hpp"
#include <iostream>
#include <algorithm>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "iostream"

using namespace cv;

int getPloarPeak(Mat &lineMat){
    std::vector<float> coefficient;
    if (lineMat.isContinuous()) {
        coefficient.assign((float*)lineMat.data, (float*)lineMat.data + lineMat.total()*lineMat.channels());
    } else {
        for (int i = 0; i < lineMat.rows; ++i) {
            coefficient.insert(coefficient.end(), lineMat.ptr<float>(i), lineMat.ptr<float>(i)+lineMat.cols*lineMat.channels());
        }
    }
    int s=coefficient.size();
    int maxElementIndex = std::max_element(coefficient.begin()+25,coefficient.end()-25) - coefficient.begin();
    return(maxElementIndex);
}


int my_fft(Mat & image,Mat & imgout)
{
    Mat I = image;
    if( I.empty())
        return -1;
    
    Mat padded;                            //expand input image to optimal size
    int m = getOptimalDFTSize( I.rows );
    int n = getOptimalDFTSize( I.cols ); // on the border add zero values
    copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));
 
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);         // Add to the expanded another plane with zeros
 
    dft(complexI, complexI);            // this way the result may fit in the source matrix
    split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude  
    Mat magI = planes[0];
    
    magI += Scalar::all(1);                    // switch to logarithmic scale
    log(magI, magI);
 
    // crop the spectrum, if it has an odd number of rows or columns
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
 
    // rearrange the quadrants of Fourier image  so that the origin is at the image center        
    int cx = magI.cols/2;
    int cy = magI.rows/2;
 
    Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant 
    Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right
 
    Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
 
    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
 
    normalize(magI, magI, 0, 1, NORM_MINMAX); // Transform the matrix with float values into a 
                                            // viewable image form (float between values 0 and 1).
    imgout = magI;
    return 0;
}
int imgWrapA(Mat& src, Mat& warp_dst, float A)
{
    Point2f srcTri[3];
    Point2f dstTri[3];

    Mat warp_mat(2, 3, CV_32FC1);
    int column=src.cols;
    int row=src.rows;

    warp_dst = Mat::zeros(src.rows, src.cols, src.type());
    srcTri[0] = Point2f(column/2.0,row/2.0);
    srcTri[1] = Point2f(column/2.0,row/4.0);
    srcTri[2] = Point2f(column/4.0,row/2.0-column/4.0*A);

    dstTri[0] = Point2f(column/2.0,row/2.0);
    dstTri[1] = Point2f(column/2.0,row/4.0);
    dstTri[2] = Point2f(column/4.0,row/2.0);
    warp_mat = getAffineTransform(srcTri, dstTri);

    warpAffine(src, warp_dst, warp_mat, warp_dst.size());
    return 0;
}
 
int estCorrect(Mat& orgImg0, Mat& correctImg, float cutoffF, float margin){
    Mat orgImg;
    fastNlMeansDenoisingColored(orgImg0,orgImg,9,9,7,21);
    Size s = orgImg.size();
    int w= s.width;
    int h= s.height;
    cv::Rect crop_region(int(margin*w), 0 ,int(w-2*margin*w), h-1);
    Mat crop_img=orgImg(crop_region);
    Mat img;
    cvtColor(crop_img, img, COLOR_BGR2GRAY);
    Mat y;
    Sobel(img, y, CV_16S, 0, 1);
    Mat absY;
    convertScaleAbs(y, absY);
    Mat magnitude_spectrum;
    my_fft(absY,magnitude_spectrum);
    float marginPlor = 0.9;// # Cut off the outer 10% of the image
    int size=min(img.rows, img.cols);
    Mat polar_img;
    warpPolar(magnitude_spectrum,polar_img, Size(int(size/2), 200), Point(img.cols/2,img.rows/2), 
                                  size*marginPlor*0.5, WARP_POLAR_LINEAR);
    float coreCut=0.01;
    Mat polar_img_lowF = polar_img(Range::all(), Range(int(coreCut*polar_img.cols),int(cutoffF*polar_img.cols))); 
    Mat polar_sum_200;

    reduce(polar_img_lowF, polar_sum_200, 1, cv::REDUCE_SUM);
    int polar_sum_200_rows = polar_sum_200.rows;
    int polar_sum_200_cols = polar_sum_200.cols;

    Mat polar_sum =polar_sum_200(Range(0,100),Range::all());
    polar_sum=polar_sum+polar_sum_200(Range(100,200),Range::all());
    int maxIndex=getPloarPeak(polar_sum);
    float offsetDegree=(maxIndex-49.0)/100.0*3.14;
    float aEst=sin(offsetDegree);
    imgWrapA(orgImg0, correctImg, aEst);
}

int estCorrect2D(Mat& orgImg,Mat& hvCorrectedImg, float cutoffF,float margin){

    Mat hCorrectedImg;
    Mat hCorrectedImg90;
    Mat vCorrectedImg;
    estCorrect(orgImg, hCorrectedImg, cutoffF, margin);
    rotate(hCorrectedImg, hCorrectedImg90, ROTATE_90_CLOCKWISE);
    estCorrect(hCorrectedImg90, vCorrectedImg, cutoffF, margin);
    rotate(vCorrectedImg, hvCorrectedImg, ROTATE_90_COUNTERCLOCKWISE);
}
int main(int argc, char** argv )
{
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }
    Mat orgImg = imread(argv[1]);
    Mat hvCorrectedImg; 
    estCorrect2D(orgImg, hvCorrectedImg, 0.8, 0.1);
    imwrite("test.png", hvCorrectedImg);
    waitKey(0);
    return 0;
}