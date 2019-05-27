#include "people_counter.h"

PeopleCounter::PeopleCounter(cv::VideoCapture& cap,
                             std::string cnf_path, std::string wts_path, std::string nms_path,
                             float ct, float st, int iw, int ih, float zsf) :
_capture(cap),
_frameRegionToShow({ 0, 0, 0, 0 }),
_frameRegionToShowPrevious({ 0, 0, 0, 0 }),
_zoomSpeedFactor(zsf),
_peopleQty(0),
_modelConfigurationFile(cnf_path),
_modelWeightsFile(wts_path),
_classesFile(nms_path),
_confThreshold(ct),
_nmsThreshold(st),
_inpWidth(iw),
_inpHeight(ih),
_threadsEnabled(true)
{
    _captureFrameWidth = static_cast<int>(_capture.get(cv::CAP_PROP_FRAME_WIDTH));
    _captureFrameHeight = static_cast<int>(_capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    
    _frameRegionToShow = { 0, 0, _captureFrameWidth, _captureFrameHeight };
    _frameRegionToShowPrevious = { 0, 0, _captureFrameWidth, _captureFrameHeight };
    _frameRegionToShowZoomed = { 0, 0, _captureFrameWidth, _captureFrameHeight };
    
    // Setup the model
    _net = cv::dnn::readNetFromDarknet(_modelConfigurationFile, _modelWeightsFile);
    _net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    _net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    
    std::ifstream ifs(_classesFile.c_str());
    std::string line;
    while (getline(ifs, line)) {
        _classes.push_back(line);
    }
}

PeopleCounter::PeopleCounter(cv::Mat& img,
	std::string cnf_path, std::string wts_path, std::string nms_path,
	float ct, float st, int iw, int ih, float zsf) :
	_image(img),
	_frameRegionToShow({ 0, 0, 0, 0 }),
	_frameRegionToShowPrevious({ 0, 0, 0, 0 }),
	_zoomSpeedFactor(zsf),
	_peopleQty(0),
	_modelConfigurationFile(cnf_path),
	_modelWeightsFile(wts_path),
	_classesFile(nms_path),
	_confThreshold(ct),
	_nmsThreshold(st),
	_inpWidth(iw),
	_inpHeight(ih),
	_threadsEnabled(true)
{
	_captureFrameWidth = static_cast<int>(_image.size().width);
	_captureFrameHeight = static_cast<int>(_image.size().height);

	_frameRegionToShow = { 0, 0, _captureFrameWidth, _captureFrameHeight };
	_frameRegionToShowPrevious = { 0, 0, _captureFrameWidth, _captureFrameHeight };
	_frameRegionToShowZoomed = { 0, 0, _captureFrameWidth, _captureFrameHeight };

	// Setup the model
	_net = cv::dnn::readNetFromDarknet(_modelConfigurationFile, _modelWeightsFile);
	_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

	std::ifstream ifs(_classesFile.c_str());
	std::string line;
	while (getline(ifs, line)) {
		_classes.push_back(line);
	}
}

void PeopleCounter::producer() {
    std::cout << "\nStarting Producer Thread\n";
    cv::Mat frame;
    
    while (_threadsEnabled) {
        _capture.read(frame);
        
        {
            std::lock_guard<std::mutex> lck(_mutexFrameCapture);
            _lastCapturedFrame = frame.clone();
        }
        
        // Stop the program if no video stream
        if (frame.empty()) {
            _threadsEnabled = false;
            break;
        }
    }
    if (_capture.isOpened()) {
        _capture.release();
    }
    std::cout << "\nStopping Producer Thread\n";
}

void PeopleCounter::processor() {
    std::cout << "\nStarting Processor Thread\n";
    cv::Mat frame;
    
    while (_threadsEnabled) {
        {
            std::lock_guard<std::mutex> lck(_mutexFrameCapture);
            frame = _lastCapturedFrame.clone();
        }
        
        if (!frame.empty()) {
            processFrame(frame);
            std::cout << "There are [ " << _peopleQty << " ] peoples\n";
        }
    }
    std::cout << "\nStopping Processor Thread\n";
}

void PeopleCounter::runThreads() {
    std::thread producer_t(&PeopleCounter::producer, this);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    std::thread processor_t(&PeopleCounter::processor, this);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    // Create a window
    static const std::string kWinName = "people counter";
    cv::namedWindow(kWinName, cv::WINDOW_NORMAL);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    while (_threadsEnabled) {
        if (cv::waitKey(1) >= 0) {
            _threadsEnabled = false;
            break;
        }
        
        {
            std::lock(_mutexFrameRegion, _mutexFrameOverlay, _mutexFrameCapture);
            std::lock_guard<std::mutex> lckRegion(_mutexFrameRegion, std::adopt_lock);
            std::lock_guard<std::mutex> lckOverlay(_mutexFrameOverlay, std::adopt_lock);
            std::lock_guard<std::mutex> lckCapture(_mutexFrameCapture, std::adopt_lock);
            
            updateFrameRegionToShow();
            
            if (!_lastCapturedFrame.empty() && !_lastOverlayFrame.empty()) {
                // Blur the background
                cv::Mat blurred = cv::Mat::zeros(_lastCapturedFrame.size(), _lastCapturedFrame.type());
                cv::GaussianBlur(_lastCapturedFrame, blurred, cv::Size(15, 15), 0.0);   
                blurred.copyTo(_lastCapturedFrame, _blurMask);                
                cv::bitwise_or(_lastCapturedFrame, _lastOverlayFrame, _lastOverlayedFrame);
            }
            
            if (!_lastOverlayedFrame.empty()) {
				cv::Mat frame = _lastOverlayedFrame(_frameRegionToShowZoomed);
                padAspectRatio(frame, (float)_inpWidth / (float)_inpHeight);                
                cv::resize(frame, frame, cv::Size(_inpWidth, _inpHeight));
                
                if (!frame.empty()) {
					cv::Mat _image = _lastOverlayedFrame;

				
                    cv::imshow(kWinName, frame);
                }
            }
        }
    }
    
    cv::destroyAllWindows();
    
    producer_t.join();
	processor_t.join();
}

void PeopleCounter::runDetectIamge()
{
	//std::lock_guard<std::mutex> lck(_mutexFrameCapture);
	_lastCapturedFrame = _image.clone();
	static const std::string kWinName = "people counter";
	cv::namedWindow(kWinName, cv::WINDOW_NORMAL);
	std::this_thread::sleep_for(std::chrono::milliseconds(10));
	processFrame(_image);

	std::lock(_mutexFrameRegion, _mutexFrameOverlay, _mutexFrameCapture);
	std::lock_guard<std::mutex> lckRegion(_mutexFrameRegion, std::adopt_lock);
	std::lock_guard<std::mutex> lckOverlay(_mutexFrameOverlay, std::adopt_lock);
	std::lock_guard<std::mutex> lckCapture(_mutexFrameCapture, std::adopt_lock);

	updateFrameRegionToShow();


	if (!_lastCapturedFrame.empty() && !_lastOverlayFrame.empty()) {
		// Blur the background
		cv::Mat blurred = cv::Mat::zeros(_lastCapturedFrame.size(), _lastCapturedFrame.type());
		cv::GaussianBlur(_lastCapturedFrame, blurred, cv::Size(15, 15), 0.0);		
		blurred.copyTo(_lastCapturedFrame, _blurMask);
		cv::bitwise_or(_lastCapturedFrame, _lastOverlayFrame, _lastOverlayedFrame);
	}

	if (!_lastOverlayedFrame.empty()) {
		cv::Mat frame = _lastOverlayedFrame(_frameRegionToShowZoomed);
		padAspectRatio(frame, (float)_inpWidth / (float)_inpHeight);
		cv::resize(frame, frame, cv::Size(_inpWidth, _inpHeight));
		if (!frame.empty()) {
			cv::imshow(kWinName, frame);
		}
	}
	std::cout << "There are [ " << _peopleQty << " ] peoples\n";
	
	cv::waitKey(0);
}

int PeopleCounter::getPeopleQty() {
    return _peopleQty;
}

void PeopleCounter::processFrame(cv::Mat& frame) {
    // Create a 4D blob from a frame.
    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1 / 255.0, cv::Size(_inpWidth, _inpHeight), cv::Scalar(0, 0, 0), true, false);
    
    // Nets forward pass
    std::vector<cv::Mat> outs;
    _net.setInput(blob);
    _net.forward(outs, getOutputsNames(_net));
    
    // Filter out low confidence objects
    _peopleQty = countPeople(frame, outs);
}

int PeopleCounter::countPeople(cv::Mat& frame, const std::vector<cv::Mat>& outs)
{
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    
    for (size_t i = 0; i < outs.size(); ++i) {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
            cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            cv::Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > _confThreshold) {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
    }
    
    // Perform non maximum suppression
    std::vector<int> indices;
    int peopleQty = 0;
    cv::dnn::NMSBoxes(boxes, confidences, _confThreshold, _nmsThreshold, indices);
    
    {
        std::lock(_mutexFrameRegion, _mutexFrameOverlay);
        std::lock_guard<std::mutex> lckRegion(_mutexFrameRegion, std::adopt_lock);
        std::lock_guard<std::mutex> lckOverlay(_mutexFrameOverlay, std::adopt_lock);
        
        _lastOverlayFrame = cv::Mat::zeros(frame.size(), frame.type());
        _blurMask = cv::Mat::ones(frame.size(), CV_8UC1);
        cv::Rect frameRegion = { _captureFrameWidth, _captureFrameHeight, (-1)*_captureFrameWidth, (-1)*_captureFrameHeight };
        
        for (size_t i = 0; i < indices.size(); ++i) {
            int idx = indices[i];
            cv::Rect box = boxes[idx];
            
            if (_classes[classIds[idx]] == "person") {
                peopleQty++;
                drawPred(classIds[idx], confidences[idx], box.x, box.y,
                         box.x + box.width, box.y + box.height, _lastOverlayFrame);
                
                adjustBlurMask(box);
                
                // Expand the frame region to show to contain all objects
                adjustFrameRegion(frameRegion, box);
            }
        }
        
        if ((peopleQty > 0) && (frameRegion.height > 0 && frameRegion.width > 0)) {
            _frameRegionToShow = frameRegion;
        }
        else {
            _frameRegionToShow = cv::Rect(0, 0, _captureFrameWidth, _captureFrameHeight);
        }
        
        // Put efficiency information.
        // The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        std::vector<double> layersTimes;
        double freq = cv::getTickFrequency() / 1000;
        double t = _net.getPerfProfile(layersTimes) / freq;
        std::string label = cv::format("Inference time for a frame : %.2f ms", t);
        cv::putText(_lastOverlayFrame, label, cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
    }
    
    return peopleQty;
}

void PeopleCounter::padAspectRatio(cv::Mat& img, float ratio) {
    int width = img.cols;
    int height = img.rows;
    
    if (width > height) {
        int padding = (width / ratio - height) / 2;
        padding = clip(padding, 0, height);
        cv::copyMakeBorder(img, img, padding, padding, 0, 0, cv::BORDER_ISOLATED, 0);
    }
    else {
        int padding = (height * ratio - width) / 2;
        padding = clip(padding, 0, width);
        cv::copyMakeBorder(img, img, 0, 0, padding, padding, cv::BORDER_ISOLATED, 0);
    }
}

void PeopleCounter::adjustBlurMask(cv::Rect& region) {
    boundRegionToCaptureFrame(region);
    _blurMask(region).setTo(cv::Scalar(0));
}

void PeopleCounter::adjustFrameRegion(cv::Rect& region, cv::Rect& box) {
    int leftTopX = std::min(box.x, region.x);
    int leftTopY = std::min(box.y, region.y);
    int rightBottomX = std::max(region.x + region.width, box.x + box.width);
    int rightBottomY = std::max(region.y + region.height, box.y + box.height);
    
    region.x = leftTopX;
    region.y = leftTopY;
    region.width = rightBottomX - leftTopX;
    region.height = rightBottomY - leftTopY;
}

void PeopleCounter::boundRegionToCaptureFrame(cv::Rect& region) {
    int leftTopX, leftTopY, rightBottomX, rightBottomY;
    
    boxToPoints(region, leftTopX, leftTopY, rightBottomX, rightBottomY);
    
    leftTopX = clip(leftTopX, 0, _captureFrameWidth);
    leftTopY = clip(leftTopY, 0, _captureFrameHeight);
    rightBottomX = clip(rightBottomX, 0, _captureFrameWidth);
    rightBottomY = clip(rightBottomY, 0, _captureFrameHeight);
    
    pointsToBox(region, leftTopX, leftTopY, rightBottomX, rightBottomY);
}

void PeopleCounter::boxToPoints(cv::Rect& box, int& leftTopX, int& leftTopY, int& rightBottomX, int& rightBottomY) {
    leftTopX = box.x;
    leftTopY = box.y;
    rightBottomX = box.x + box.width;
    rightBottomY = box.y + box.height;
}

void PeopleCounter::pointsToBox(cv::Rect& box, int& leftTopX, int& leftTopY, int& rightBottomX, int& rightBottomY) {
    box.x = leftTopX;
    box.y = leftTopY;
    box.width = rightBottomX - leftTopX;
    box.height = rightBottomY - leftTopY;
}

void PeopleCounter::updateFrameRegionToShow() {
    static int oldLeftTopX, oldLeftTopY, oldRightBottomX, oldRightBottomY;
    static int leftTopX, leftTopY, rightBottomX, rightBottomY;
    static int newLeftTopX, newLeftTopY, newRightBottomX, newRightBottomY;
    
	boxToPoints(_frameRegionToShowPrevious, oldLeftTopX, oldLeftTopY, oldRightBottomX, oldRightBottomY);
	boxToPoints(_frameRegionToShow, leftTopX, leftTopY, rightBottomX, rightBottomY);
    boxToPoints(_frameRegionToShowZoomed, newLeftTopX, newLeftTopY, newRightBottomX, newRightBottomY);
//     
//     newLeftTopX += 0;
//     newLeftTopY += 0;
//     newRightBottomX += 0;
//     newRightBottomY += 0;
//     
	newLeftTopX += std::ceil(_zoomSpeedFactor * (leftTopX - oldLeftTopX));
	newLeftTopY += std::ceil(_zoomSpeedFactor * (leftTopY - oldLeftTopY));
	newRightBottomX += std::ceil(_zoomSpeedFactor * (rightBottomX - oldRightBottomX));
	newRightBottomY += std::ceil(_zoomSpeedFactor * (rightBottomY - oldRightBottomY));
	
    pointsToBox(_frameRegionToShowZoomed, newLeftTopX, newLeftTopY, newRightBottomX, newRightBottomY);
    boundRegionToCaptureFrame(_frameRegionToShowZoomed);
    
    _frameRegionToShowPrevious = _frameRegionToShowZoomed;
}

// Draw the predicted bounding box
void PeopleCounter::drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame)
{
	//cv::namedWindow("show in Frame", cv::WINDOW_AUTOSIZE);
	//cv::imshow("show in Frame", frame);
    //Draw a rectangle displaying the bounding box
    rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(255, 178, 50), 3);
    
    //Get the label for the class name and its confidence
    std::string label = cv::format("%.2f", conf);
    if (!_classes.empty()) {
        CV_Assert(classId < (int)_classes.size());
        label = _classes[classId] + ":" + label;
    }
    
    //Display the label at the top of the bounding box
    int baseLine;
    cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = cv::max(top, labelSize.height);
    rectangle(frame, cv::Point(left, top - round(1.5*labelSize.height)), cv::Point(left + round(1.5*labelSize.width), top + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
    putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0), 1);
}

std::vector<std::string> PeopleCounter::getOutputsNames(const cv::dnn::Net& net)
{
    static std::vector<std::string> names;
    if (names.empty()) {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        std::vector<int> outLayers = net.getUnconnectedOutLayers();
        
        //get the names of all the layers in the network
        std::vector<std::string> layersNames = net.getLayerNames();
        
        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i) {
            names[i] = layersNames[outLayers[i] - 1];
        }
    }
    return names;
}
