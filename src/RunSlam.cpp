
#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <string>
#include <fstream>
#include <iomanip>

using namespace std;

void LoadImages(const std::string &imgPath, std::vector<std::string> imgsLeft, std::vector<std::string> imgsRight);

int main(int argc, char** argv) {

	std::vector<std::string> imgsLeft;
	std::vector<std::string> imgsRight;
	cv::Mat imLeft, imRight;

	LoadImages(argv[1], imgsLeft, imgsRight);

	for(size_t i; i < imgsLeft.size(); i++)
	{
		imLeft = cv::imread();

	}

	return 0;
}


void LoadImages(const std::string &imgPath, std::vector<std::string> imgsLeft, std::vector<std::string> imgsRight)
{
	std::string timesPath = imgPath + "/times.txt";
	std::string imgsLeftPath = imgPath + "/image_0/";
	std::string imgsRightPath = imgPath + "/image_1/";
	std::ifstream inTimes;
	std::string time;
	std::vector<std::string> times;

	inTimes.open(timesPath);

	while(!inTimes.eof())
	{
		getline(inTimes, time);

		if(!time.empty())
			times.push_back(time);
	}
	cout << "times.size(): " << times.size() << endl;

	imgsLeft.resize(times.size());
	imgsRight.resize(times.size());

	for(size_t i = 0; i < times.size(); i++)
	{
		std::stringstream ss;
		ss << std::setfill('0') << std::setw(6) << i;

		imgsLeft[i] = imgsLeftPath + ss.str() + ".png";
		imgsRight[i] = imgsRightPath + ss.str() + ".png";

	}
}