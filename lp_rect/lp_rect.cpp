#include <stdio.h>
#include "opencv2/opencv.hpp"
#include <iostream>
#include <algorithm>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "iostream"


using namespace cv;

int getPloarPeak(Mat &lineMat){
    //Mat plate;
    //plotter p(plate,cv::Size(800,800));
    //p.plot_circle(plate, 7, 5, -7);
    //p.plot_line(plate, 2, 5);
    //std::vector<float> coefficient(lineMat.rows*lineMat.cols);
    //if (lineMat.isContinuous())
    //    coefficient = lineMat.data;
    std::vector<float> coefficient;
    if (lineMat.isContinuous()) {
        // array.assign((float*)mat.datastart, (float*)mat.dataend); // <- has problems for sub-matrix like mat = big_mat.row(i)
        coefficient.assign((float*)lineMat.data, (float*)lineMat.data + lineMat.total()*lineMat.channels());
    } else {
        for (int i = 0; i < lineMat.rows; ++i) {
            coefficient.insert(coefficient.end(), lineMat.ptr<float>(i), lineMat.ptr<float>(i)+lineMat.cols*lineMat.channels());
        }
    }
    int s=coefficient.size();
    for (int ii=0; ii<s; ii++){
        printf("coefficient[%d]=%f\n", coefficient[ii]);
    }
    //float maxIndex;
    int maxElementIndex = std::max_element(coefficient.begin()+25,coefficient.end()-25) - coefficient.begin();
    float maxElement = *std::max_element(coefficient.begin()+25, coefficient.end()-25);
    std::cout << "maxElementIndex:" << maxElementIndex << ", maxElement:" << maxElement << '\n';
    return(maxElementIndex);
    //p.plot_polynomial(plate,s,coefficient ,cv::Scalar(0,0,255));
    
    //imshow("plate  Image", plate);
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
 
    // compute the magnitude and switch to logarithmic scale
    // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
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
/*
def imgWrapA(orgImg,a):
    column=orgImg.shape[1]
    row=orgImg.shape[0]
    pts1 = np.float32([[column/2,row/2],[column/2,row/4],[column/4,row/2-column/4*a]])
    pts2 = np.float32([[column/2,row/2],[column/2,row/4],[column/4,row/2]])
    M = cv2.getAffineTransform(pts1,pts2)
    imgWarpAffine = cv2.warpAffine(orgImg,M,(column,row))
    return imgWarpAffine
*/
int imgWrapA(Mat& src, Mat& warp_dst, float A)
{
    Point2f srcTri[3];
    Point2f dstTri[3];

    Mat warp_mat(2, 3, CV_32FC1);
    int column=src.cols;
    int row=src.rows;

    warp_dst = Mat::zeros(src.rows, src.cols, src.type());

    /// 设置源图像和目标图像上的三组点以计算仿射变换
    srcTri[0] = Point2f(column/2.0,row/2.0);
    srcTri[1] = Point2f(column/2.0,row/4.0);
    srcTri[2] = Point2f(column/4.0,row/2.0-column/4.0*A);

    dstTri[0] = Point2f(column/2.0,row/2.0);
    dstTri[1] = Point2f(column/2.0,row/4.0);
    dstTri[2] = Point2f(column/4.0,row/2.0);

    /// 求得仿射变换
    warp_mat = getAffineTransform(srcTri, dstTri);

    /// 对源图像应用上面求得的仿射变换
    warpAffine(src, warp_dst, warp_mat, warp_dst.size());

    
    return 0;
}
 
//def estCorrect(orgImg0, cutoffF=0.8, margin=0.1):
int estCorrect(Mat& orgImg0, Mat& correctImg, float cutoffF, float margin){


    Mat orgImg;
    fastNlMeansDenoisingColored(orgImg0,orgImg,9,9,7,21);
    //imshow("Display Image", orgImg);
    
    Size s = orgImg.size();
    int w= s.width;
    int h= s.height;

    printf("int w= %d, int h= %d\n",w,h);
    //cv::Rect crop_region(int a,int b, int c, int d);
    //a,b : Coordinates of the top-left corner.
    //c,d : Rectangle width and height.
    printf("int(margin*w) = %d\n", int(margin*w));
    printf("int(w-margin*w) = %d\n", int(w-margin*w));

    cv::Rect crop_region(int(margin*w), 0 ,int(w-2*margin*w), h-1);
// specifies the region of interest in Rectangle form

    Mat crop_img=orgImg(crop_region);
    //imshow("crop_img Image", crop_img);

    
    //crop_img = orgImg[:, int(margin*w):int(w-margin*w)]
    //pix_color = np.array(crop_img)
    //full_pix_color = np.array(orgImg)
    //img = rgb2gray(pix_color)
    Mat img;
    cvtColor(crop_img, img, COLOR_BGR2GRAY);
    //imshow("gray img Image", img);
    Mat y;
    Sobel(img, y, CV_16S, 0, 1);
    Mat absY;
    convertScaleAbs(y, absY);
    //imshow("Sobel img Image", absY);
    
    //img=abs(img[0:-1,:]-img[1:,:])
    
    //print(img.shape)
    //f = np.fft.fft2(img)
    Mat magnitude_spectrum;
    my_fft(absY,magnitude_spectrum);
    //imshow("magnitude_spectrum img Image", magnitude_spectrum);



    
    //fshift = np.fft.fftshift(f)
    //magnitude_spectrum = 20*np.log(np.abs(fshift))
        
    //#plt.subplot(143),plt.imshow(np.abs(fshift), cmap = 'gray')
    //#plt.title('np.abs(fshift)'), plt.xticks([]), plt.yticks([])
    float marginPlor = 1.0;// # Cut off the outer 10% of the image
    //# Do the polar rotation along 100 angular steps with a radius of 256 pixels.
    int size=min(magnitude_spectrum.rows, magnitude_spectrum.cols);
    printf("size = %d\n", size);
    Mat polar_img;
    warpPolar(magnitude_spectrum,polar_img, Size(int(size/2), 200), Point(magnitude_spectrum.cols/2,magnitude_spectrum.rows/2), 
                                  size*marginPlor*0.5, WARP_POLAR_LINEAR);
    //imshow("polar_img  Image", polar_img);
    //imwrite("magnitude_spectrum.png", magnitude_spectrum);

    float coreCut=0.01;


    //warpPolar(magnitude_spectrum,polar_img, (int(size/2), 200), (img.rows/2,img.cols/2), 
    //                              size*margin*0.5, cv2.WARP_POLAR_LINEAR)
    
    //#print(polar_img.shape)
    Mat polar_img_lowF = polar_img(Range::all(), Range(int(coreCut*polar_img.cols),int(cutoffF*polar_img.cols))); 
    //imshow("polar_img_lowF  Image", polar_img_lowF);
    Mat polar_sum_200;

    reduce(polar_img_lowF, polar_sum_200, 1, cv::REDUCE_SUM);
    int polar_sum_200_rows = polar_sum_200.rows;
    int polar_sum_200_cols = polar_sum_200.cols;
    printf("polar_sum_200_rows =%d\n", polar_sum_200_rows);
    printf("polar_sum_200_cols =%d\n", polar_sum_200_cols);
    //imshow("polar_sum_200  Image", polar_sum_200);

    Mat polar_sum =polar_sum_200(Range(0,100),Range::all());
    polar_sum=polar_sum+polar_sum_200(Range(100,200),Range::all());
    int maxIndex=getPloarPeak(polar_sum);
    //1+polar_sum_200[ 100:200]

    //polar_img_lowF=polar_img[:,int(coreCut*polar_img.shape[1]):int(cutoffF*polar_img.shape[1])]
    
    //polar_sum_200=np.sum(polar_img_lowF,axis=1)
    //polar_sum=polar_sum_200[0:100]+polar_sum_200[ 100:200]
    //#polar_sum[50]=min(polar_sum) #matthew  do not count center line
    //#print(statistics.stdev(polar_sum[25:75]))
    //gainStdev=statistics.stdev(polar_sum[25:75])/10000
    //if isBias:
    //    polar_sum[45:56]=(polar_sum[45:56]*GAIN*gainStdev+polar_sum[45:56]*(1-gainStdev))
    //maxIndex=np.argmax(polar_sum[25:75])+25
    //print(maxIndex)
    float offsetDegree=(maxIndex-50.0)/100.0*3.14;
    printf("offsetDegree =%f\n", offsetDegree);
    float aEst=sin(offsetDegree);
    printf("aEst =%f\n", aEst);
    //Mat correctImg;
    imgWrapA(orgImg0, correctImg, aEst);
    //imshow("correctImg  Image", correctImg);
    
    //#correctImg=imgWrapA(pix_color,aEst)
    
    //full_pix_color0 = np.array(orgImg0)
    //correctImg=imgWrapA(full_pix_color0,aEst)
    //#full_pix_color
    //#polar_sum[25:75]=min(polar_sum) #matthew  do not count center line
    ///#maxIndex2=np.argmax(polar_sum)
    //#print("maxIndex2={}".format(maxIndex2))
    /*
    return correctImg
    */
}

//def estCorrect2D(orgImg, cutoffF=0.8, margin=0.1):
int estCorrect2D(Mat& orgImg,Mat& hvCorrectedImg, float cutoffF,float margin){

    Mat hCorrectedImg;
    Mat hCorrectedImg90;
    Mat vCorrectedImg;
    //Mat hvCorrectedImg;


    estCorrect(orgImg, hCorrectedImg, cutoffF, margin);

    //imshow("hCorrectedImg  Image", hCorrectedImg);

    rotate(hCorrectedImg, hCorrectedImg90, ROTATE_90_CLOCKWISE);
    estCorrect(hCorrectedImg90, vCorrectedImg, cutoffF, margin);
    rotate(vCorrectedImg, hvCorrectedImg, ROTATE_90_COUNTERCLOCKWISE);
    //imshow("hvCorrectedImg  Image", hvCorrectedImg);

    /*
    hCorrectedImg90=np.rot90(hCorrectedImg, )
    vCorrectedImg=estCorrect(hCorrectedImg90, cutoffF, margin)
    vCorrectedImg270=np.rot90(vCorrectedImg,  k=3)
    if isPlot:
        plt.subplots(1,3,figsize=(12,4))
        plt.subplot(131),plt.imshow(orgImg)
        plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(132),plt.imshow(hCorrectedImg)
        plt.title('Vertical rectify result'), plt.xticks([]), plt.yticks([])
        plt.subplot(133),plt.imshow(vCorrectedImg270)
        plt.title('Horizontal rectify result'), plt.xticks([]), plt.yticks([])
        plt.show()
        
    
    return hCorrectedImg, vCorrectedImg270
    */
}
/*
int fftplotWarp(char* imgPath){
    //
    Mat orgImg = imread(imgPath);
    //imshow("Display Image", orgImg);
    //pix_color = np.array(orgImg)
    //hCorrectedImg, CorrectedImg=estCorrect2D(pix_color, 0.8, 0.1)
    //CorrectedImg=estCorrect2D(orgImg, 0.8, 0.1);
    Mat hvCorrectedImg; 


    estCorrect2D(orgImg, hvCorrectedImg, 0.8, 0.1);

    imwrite("test.png", hvCorrectedImg); 
    //return(CorrectedImg);
}
*/

int main(int argc, char** argv )
{
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }

    //fftplotWarp(argv[1]);

    Mat orgImg = imread(argv[1]);
    //imshow("Display Image", orgImg);
    //pix_color = np.array(orgImg)
    //hCorrectedImg, CorrectedImg=estCorrect2D(pix_color, 0.8, 0.1)
    //CorrectedImg=estCorrect2D(orgImg, 0.8, 0.1);
    Mat hvCorrectedImg; 


    estCorrect2D(orgImg, hvCorrectedImg, 0.8, 0.1);

    imwrite("test.png", hvCorrectedImg);

    //test();

    /*

    Mat image;
    image = imread( argv[1], 1 );

    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", image);
    */

    waitKey(0);
    return 0;
}