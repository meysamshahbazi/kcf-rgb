#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp> 

#include <iostream>
#include <cstring>
//  #include "samples_utility.hpp"

using namespace std;
using namespace cv;

// prototype of the functino for feature extractor
void sobelExtractor(const Mat img, const Rect roi, Mat& feat);
void lbpExtractor(const Mat img,const Rect roi, Mat & feat);

template <typename _Tp>
void OLBP_(const Mat& src, Mat& dst);

int main( int argc, char** argv ){
    // show help
    if(argc<2){
        cout<<
        " Usage: tracker <video_name>\n"
        " examples:\n"
        " example_tracking_kcf Bolt/img/%04d.jpg\n"
        " example_tracking_kcf faceocc2.webm\n"
        << endl;
        return 0;
    }

    // declares all required variables
    Rect roi;
    Mat frame;

    TrackerKCF::Params param;
    param.resize = true;
    // float 
    param.detect_thresh = 0.25f;         //!<  detection confidence threshold
    param.sigma = 0.2f;                 //!<  gaussian kernel bandwidth
    param.lambda = 0.0001f; // 0.0001f                //!<  regularization
    param.interp_factor = 0.075f;         //!<  linear interpolation factor for adaptation
    param.output_sigma_factor = 1.0f / 16.0f;   //!<  spatial bandwidth (proportional to target)
    param.pca_learning_rate = 0.15f;     //!<  compression learning rate
    // bool 
    param.resize = true;                  //!<  activate the resize feature to improve the processing speed
    param.split_coeff=true;             //!<  split the training coefficients into two matrices
    param.wrap_kernel= false;             //!<  wrap around the kernel values
    param.compress_feature  = false;        //!<  activate the pca method to compress the features
    // int 
    param.max_patch_size = 40*40;  //80*80         //!<  threshold for the ROI size
    param.compressed_size = 1;          //!<  feature size after compression
    param.desc_pca  = TrackerKCF::MODE::GRAY;        //!<  compressed descriptors of TrackerKCF::MODE
    param.desc_npca = TrackerKCF::MODE::CUSTOM;       //!<  non-compressed descriptors of TrackerKCF::MODE

    // create a tracker object
    Ptr<TrackerKCF> tracker = TrackerKCF::create(param);

    tracker->setFeatureExtractor(lbpExtractor,false);

    // set input video
    std::string video = argv[1];
    VideoCapture cap(video);

    // get bounding box
    cap >> frame;
    roi=selectROI("tracker",frame);

    // Mat patch = frame(roi).clone();
    // cout<<"ROI "<<roi.size()<<endl;
    // cout<<"Patch "<<patch.size()<< endl;
    // imshow("imag",patch);
    // waitKey(0);

    //quit if ROI was not selected
    if(roi.width==0 || roi.height==0)
        return 0;

    // initialize the tracker
    int64 t1 = cv::getTickCount();
    tracker->init(frame,roi);
    int64 t2 = cv::getTickCount();
    int64 tick_counter = t2 - t1;

    // perform the tracking process
    printf("Start the tracking process, press ESC to quit.\n");
    int frame_idx = 1; 
    for ( ;; ){
        // get frame from the video
        cap >> frame;

        // stop the program if no more images
        if(frame.rows==0 || frame.cols==0)
        break;

        // update the tracking result
        t1 = cv::getTickCount();
        bool is_found = tracker->update(frame,roi);
        t2 = cv::getTickCount();
        tick_counter += t2 - t1;
        frame_idx++;
        // draw the tracked object
        if(is_found){
          rectangle( frame, roi, Scalar( 255, 0, 0 ), 2, 1 );
          // show image with the tracked object
          imshow("tracker",frame);
        }
        else
        {
          cout<< "TARGET has been lost"<<endl;
          break;

        }
        

        //quit on ESC button
        if(waitKey(1)==27)break;
    }

    cout << "Elapsed sec: " << static_cast<double>(tick_counter) / cv::getTickFrequency() << endl;
    cout << "FPS: " << ((double)(frame_idx)) / (static_cast<double>(tick_counter) / cv::getTickFrequency()) << endl;

    return 0;
}

void sobelExtractor(const Mat img, const Rect roi, Mat& feat){
    Mat sobel[2];
    Mat patch;
    Rect region=roi;

    // extract patch inside the image
    if(roi.x<0){region.x=0;region.width+=roi.x;}
    if(roi.y<0){region.y=0;region.height+=roi.y;}
    if(roi.x+roi.width>img.cols)region.width=img.cols-roi.x;
    if(roi.y+roi.height>img.rows)region.height=img.rows-roi.y;
    if(region.width>img.cols)region.width=img.cols;
    if(region.height>img.rows)region.height=img.rows;

    patch=img(region).clone();
    cvtColor(patch,patch, COLOR_BGR2GRAY);

    // add some padding to compensate when the patch is outside image border
    int addTop,addBottom, addLeft, addRight;
    addTop=region.y-roi.y;
    addBottom=(roi.height+roi.y>img.rows?roi.height+roi.y-img.rows:0);
    addLeft=region.x-roi.x;
    addRight=(roi.width+roi.x>img.cols?roi.width+roi.x-img.cols:0);

    copyMakeBorder(patch,patch,addTop,addBottom,addLeft,addRight,BORDER_REPLICATE);

    Sobel(patch, sobel[0], CV_32F,1,0,1);
    Sobel(patch, sobel[1], CV_32F,0,1,1);

    merge(sobel,2,feat);
    std::cout<<sobel[0].size()<<endl;
    feat=feat/255.0-0.5; // normalize to range -0.5 .. 0.5
}

void lbpExtractor(const Mat img,const Rect roi, Mat & feat)
{
    Mat patch;
    Rect region=roi;
    

    // extract patch inside the image
    if(roi.x<0){region.x=0;region.width+=roi.x;}
    if(roi.y<0){region.y=0;region.height+=roi.y;}
    if(roi.x+roi.width>img.cols)region.width=img.cols-roi.x;
    if(roi.y+roi.height>img.rows)region.height=img.rows-roi.y;
    if(region.width>img.cols)region.width=img.cols;
    if(region.height>img.rows)region.height=img.rows;

    patch=img(region).clone();

    // cout<<"ROI "<<roi.size()<<endl;
    // cout<<"Patch "<<patch.size()<< endl;
    // imshow("imag",patch);
    // waitKey(0);


    cvtColor(patch,patch, COLOR_BGR2HSV);

    

    // add some padding to compensate when the patch is outside image border
    int addTop,addBottom, addLeft, addRight;
    addTop=region.y-roi.y;
    addBottom=(roi.height+roi.y>img.rows?roi.height+roi.y-img.rows:0);
    addLeft=region.x-roi.x;
    addRight=(roi.width+roi.x>img.cols?roi.width+roi.x-img.cols:0);

    copyMakeBorder(patch,patch,addTop,addBottom,addLeft,addRight,BORDER_REPLICATE);
    vector<Mat> hsv_ch;
    split(patch,hsv_ch);
    std::vector<Mat> features;
    for(auto & c:hsv_ch)
    {
        Mat lbp;
        OLBP_<unsigned char>(c,lbp);
        lbp.convertTo(lbp, CV_32FC1, 1.0/255.0, -0.5);
        features.push_back(lbp);
        // add hsv channels!
        // Mat cc;
        // c.convertTo(cc, CV_32FC1, 1.0/255.0, -0.5);
        // features.push_back(cc);
    }

    merge(features,feat);
    // std::cout<<feat.size()<<endl;
    return;
}

template <typename _Tp>
void OLBP_(const Mat& src, Mat& dst) {
	dst = Mat::zeros(src.rows, src.cols, CV_8UC1);
	for(int i=1;i<src.rows-1;i++) {
		for(int j=1;j<src.cols-1;j++) {
			_Tp center = src.at<_Tp>(i,j);
			unsigned char code = 0;
			code |= (src.at<_Tp>(i-1,j-1) > center) << 7;
			code |= (src.at<_Tp>(i-1,j) > center) << 6;
			code |= (src.at<_Tp>(i-1,j+1) > center) << 5;
			code |= (src.at<_Tp>(i,j+1) > center) << 4;
			code |= (src.at<_Tp>(i+1,j+1) > center) << 3;
			code |= (src.at<_Tp>(i+1,j) > center) << 2;
			code |= (src.at<_Tp>(i+1,j-1) > center) << 1;
			code |= (src.at<_Tp>(i,j-1) > center) << 0;
			dst.at<unsigned char>(i,j) = code;
		}
	}
}