#include "people_counter.h"
#include <windows.h>
#include <Shlwapi.h>
#pragma comment(lib, "shlwapi.lib")


const char* keys =
"{help h ?|| usage examples: peoplecounter.exe --dev=0 }"
"{mov     |mov.mp4| video file name                    }"
"{pic     |13.jpg| image file name                    }"
"{dev     |0| input device id                          }"
"{ct      |0.5| confidence threshold                   }"
"{st      |0.4| non-maximum suppression threshold      }"
"{iw      |320| width of network's input image         }"
"{ih      |320| height of network's input image        }"
"{cfg     |net.cfg| network configuration              }"
"{wts     |net.wts| network weights                    }"
"{nms     |net.nms| network object classes             }"
"{zsf     |0.01| zooming speed factor                   }"
;

int main(int argc, char** argv)
{
	char szEXEPath[2048];
	GetModuleFileName(NULL, szEXEPath, 2048);
	PathRemoveFileSpec(szEXEPath);
	
	std::string strExePath = szEXEPath;
	strExePath += "\\";	

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Use this application to count the number of people in a video stream.");
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }
    
    cv::VideoCapture cap;
	cv::Mat image;
    
    try {
//		 Open the image file
//   		if (parser.has("pic")) {
//   			std::string str_name = strExePath + parser.get<std::string>("pic");
//    			image = cv::imread(str_name, cv::IMREAD_COLOR);
//  		}
        // Open the video file
		if (parser.has("mov")) {
			std::string str_name = strExePath + parser.get<std::string>("mov");			
			cap.open(str_name);			
        } 
		else 
		{
        // Open the cam
            cap.open(parser.get<int>("dev"));
        }
    }
    catch(...) {   
        return 0;
    }
    
	
	if (cap.isOpened()) {
		std::string strCfg = parser.get<std::string>("cfg");
		PeopleCounter peopleCounter(cap,
			strExePath + parser.get<std::string>("cfg"), strExePath + parser.get<std::string>("wts"), strExePath + parser.get<std::string>("nms"),
            parser.get<float>("ct"), parser.get<float>("st"),
            parser.get<int>("iw"), parser.get<int>("ih"), parser.get<float>("zsf"));        
			peopleCounter.runThreads();
    }
	if (!image.empty())                      // Check for invalid input
	{
			PeopleCounter peopleCounter(image,
			strExePath + parser.get<std::string>("cfg"), strExePath + parser.get<std::string>("wts"), strExePath + parser.get<std::string>("nms"),
			parser.get<float>("ct"), parser.get<float>("st"),
			parser.get<int>("iw"), parser.get<int>("ih"), parser.get<float>("zsf"));
			peopleCounter.runDetectIamge();
	}
    cv::waitKey(1000);
    
    return 0;
}
