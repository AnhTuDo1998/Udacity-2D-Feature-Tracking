#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = cv::NORM_HAMMING;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        // ...
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)

        // ...
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
// BRIEF, ORB, FREAK, AKAZE, SIFT
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("BRIEF") == 0)
    {

        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
    else if (descriptorType.compare("ORB") == 0)
    {

        extractor = cv::ORB::create();
    }
    else if (descriptorType.compare("FREAK") == 0)
    {

        extractor = cv::xfeatures2d::FREAK::create();
    }
    else if (descriptorType.compare("AKAZE") == 0)
    {

        extractor = cv::AKAZE::create();
    }
    else if (descriptorType.compare("SIFT") == 0)
    {

        extractor = cv::xfeatures2d::SIFT::create();
    }
    else 
    {
        throw "Descriptor not implemented in this project!";
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

// Detect keypoints in image using the Harris Corner Detector
void detKeypointsHarris(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)
  
    // Detect Harris Corners and normalize output
    double t = (double)cv::getTickCount();
    cv::Mat dst, dst_norm, dst_norm_scaled;
  
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);
  
    if (bVis)
    {
        string windowName = "Harris Corner Detector Response Matrix";
        cv::namedWindow(windowName, 4);
        cv::imshow(windowName, dst_norm_scaled);
    }
  
    // Non-maxima suppresion and store result keypoints
    double overlap_percentage_tolerated = 0.0; // Total no overlap of keypoints
  
    // Loop through the pixels of the Harris Corner Detector Response Matrix.
    // Check if it is a keypoint (response is above the minReponse threshold).
    // If so, check if it overlap with any keypoints in our keypoints list (to return out).
    // If not overlap => Add, OTW: if the new keypoint has higher response replace, else leave as it is
    for (int row_idx = 0; row_idx < dst_norm.rows; row_idx++){
        for (int col_idx = 0; col_idx < dst_norm.cols; col_idx++){
        
            int harris_response = (int)dst_norm.at<float>(row_idx, col_idx);
        
            //Larger than the min response to qualify as a keypoint
            if (harris_response >= minResponse){
                cv::KeyPoint new_keypoint;
                new_keypoint.pt = cv::Point2f(col_idx, row_idx);
                new_keypoint.size = 2 * apertureSize;
                new_keypoint.response = harris_response;
          
                bool is_overlap = false;
          
                for (auto keypoint_iter = keypoints.begin(); keypoint_iter != keypoints.end(); keypoint_iter++){
                    // calculate overlap percentage
                    double overlapness = cv::KeyPoint::overlap(new_keypoint, *keypoint_iter);
                    if (overlapness > overlap_percentage_tolerated){
                        // Check if the response for the new keypoint is larger
                        // If that is the case, we will suppress the non-maximal response of the overlapped keypoint
                        is_overlap = true;
                        if (new_keypoint.response > keypoint_iter->response){
                            // Swap out the overlapped keypoint in such case
                            *keypoint_iter = new_keypoint;
                            break;
                        }// Otherwise, let it run all the way until all old_keypoints have been checked.
                    }
                }
              
                // Once check all keypoints, and none are overlap with the new one, then we insert it
                if(!is_overlap){
                    keypoints.push_back(new_keypoint);
                }
            }
        }
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Harris Corner detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
  
    if (bVis){
        // Visualize the keypoints
        cv::Mat visImg = dst_norm_scaled.clone();
        string window_name = "Harris Corner Keypoints Visualizer";
        cv::namedWindow(window_name, 5);
        cv::drawKeypoints(dst_norm_scaled, keypoints, visImg, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::imshow(window_name, visImg);
        cv::waitKey(0);
    }
}

// Detect keypoints in image using modern keypoint detectors
// FAST, BRISK, ORB, AKAZE, SIFT
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{
    // Construct a feature detector based on detectorType
    cv::Ptr<cv::FeatureDetector> detector;
    if (detectorType.compare("FAST")==0) detector = cv::FastFeatureDetector::create();
    else if (detectorType.compare("BRISK")==0) detector = cv::BRISK::create();
    else if (detectorType.compare("ORB")==0) detector = cv::ORB::create();
    else if (detectorType.compare("AKAZE")==0) detector = cv::AKAZE::create();
    else if (detectorType.compare("SIFT")==0) detector = cv::xfeatures2d::SIFT::create();
    else throw "Selected feature detector not implemented!";

    // detect the keypoints
    double t = (double)cv::getTickCount();
    detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << detectorType <<" detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // viz if needed
    if (bVis){
        cv::Mat visImg = img.clone();
        string windowName = detectorType + " Feature Detector Keypoints";
        cv::drawKeypoints(img, keypoints, visImg);
        cv::imshow(windowName, visImg);
    }
        
    
}
