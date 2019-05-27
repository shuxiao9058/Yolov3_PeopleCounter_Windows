#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <thread>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>



class PeopleCounter
{
public:
    PeopleCounter(cv::VideoCapture& capture,
					std::string cnf_path, std::string wts_path, std::string nms_path,
					float ct, float st,
					int iw, int ih, float zsf);
	PeopleCounter(cv::Mat& image,
					std::string cnf_path, std::string wts_path, std::string nms_path,
					float ct, float st,
					int iw, int ih, float zsf);
    
    void runThreads();
	void runDetectIamge();
    int getPeopleQty();
    
private:
	enum DetectSource {
		DETECT_PICTURE = 0, //!< status detect picture.
		DETECT_VIDEO = 1 //!< status detect video.
	};
    // Get the names of the output layers
    std::vector<std::string> getOutputsNames(const cv::dnn::Net& net);
    // Filter out low confidence objects with non-maxima suppression
    void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame);
    int countPeople(cv::Mat& frame, const std::vector<cv::Mat>& outs);
    void processFrame(cv::Mat& frame);	
    void updateFrameRegionToShow();
    void boundRegionToCaptureFrame(cv::Rect& region);
    void adjustFrameRegion(cv::Rect& region, cv::Rect& box);
    void adjustBlurMask(cv::Rect& region);
    void padAspectRatio(cv::Mat& img, float ratio);
    void boxToPoints(cv::Rect& box, int& leftTopX, int& leftTopY, int& rightBottomX, int& rightBottomY);
    void pointsToBox(cv::Rect& box, int& leftTopX, int& leftTopY, int& rightBottomX, int& rightBottomY);
    
    template <typename T>
    T clip(const T& n, const T& lower, const T& upper) {
        return std::max(lower, std::min(n, upper));
    }
    
    void producer();
    void processor();
    
    cv::VideoCapture _capture;
	cv::Mat _image;
    cv::Mat _lastCapturedFrame;
    cv::Mat _lastOverlayFrame;
    cv::Mat _lastOverlayedFrame;
    cv::Mat _lastProcessedFrame;
    cv::Mat _blurMask;
    cv::Rect _frameRegionToShow;
    cv::Rect _frameRegionToShowPrevious;
    cv::Rect _frameRegionToShowZoomed;
    float _zoomSpeedFactor;
    int _captureFrameWidth;
    int _captureFrameHeight;
    
    int _peopleQty;
    
    cv::dnn::Net _net;                        // object detection neural network
    
    std::string _modelConfigurationFile; // network configuration file
    std::string _modelWeightsFile;        // network weights file
    std::string _classesFile;            // network classes file
    
    float _confThreshold;            // Confidence threshold
    float _nmsThreshold;            // Non-maximum suppression threshold
    int _inpWidth;                    // Width of network's input image
    int _inpHeight;                    // Height of network's input image
    std::vector<std::string> _classes;
    
    bool _threadsEnabled;
    std::mutex _mutexFrameCapture;
    std::mutex _mutexFrameRegion;
    std::mutex _mutexFrameOverlay;
};

