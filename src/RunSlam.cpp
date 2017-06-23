
#include <iostream>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>
#include <opencv2/calib3d.hpp>
#include <string>
#include <fstream>
#include <iomanip>

using namespace std;

void LoadImages(const std::string &imgPath, std::vector<std::string> &imgsLeft, std::vector<std::string> &imgsRight);
void DetectFeatures(const cv::Mat &imLeft, const cv::Mat &imRight,
					std::vector<cv::KeyPoint> &keyPointsLeft, std::vector<cv::KeyPoint> &keyPointsRight);
void ComputeDepthMap(const cv::Mat &imLeft, const cv::Mat &imRight, cv::Mat &filtered_disp, cv::Mat &filtered_disp_vis);
void MakeDescriptors(const cv::Mat &imLeft, const cv::Mat &imRight,
					 std::vector<cv::KeyPoint> &keyPointsLeft, std::vector<cv::KeyPoint> &keyPointsRight,
					 cv::Mat &descriptorsLeft, cv::Mat &descriptorsRight);
void MatchKeyPoints(cv::Mat &descriptorsLeft, cv::Mat &descriptorsRight, std::vector<cv::DMatch> &matches);
void pose2d2d(const std::vector<cv::KeyPoint> &keyPointsLeft, const std::vector<cv::KeyPoint> &keyPointsRight,
		      const std::vector<cv::DMatch> &matches, cv::Mat &R, cv::Mat &t, const cv::Mat intrinsicParams);


int main(int argc, char** argv) {

	std::vector<std::string> imgsLeft;
	std::vector<std::string> imgsRight;
	cv::Mat imLeftPrev, imLeft, imRight;
	cv::Mat img_1, img_2;
	std::vector<cv::KeyPoint> keyPointsLeft, keyPointsRight;
	cv::Mat descriptorsLeft, descriptorsRight;
	cv::Mat filteredDisp, filteredDispVis;
	std::vector<cv::DMatch> matches;
	cv::Mat R, t;

	//Camera intrinsic params

	cv::Mat intrinsicParams  = (cv::Mat_<double>(3, 3) << 707.0912, 0, 601.8873, 0, 707.0912, 183.1104, 0, 0, 1);

	LoadImages(argv[1], imgsLeft, imgsRight);

	int NumImgs = imgsLeft.size();

	std::cout << "NumImgs: " << NumImgs << "\n";

	for(size_t i = 0; i < NumImgs; i++)
	{

		keyPointsLeft.clear();
		keyPointsRight.clear();
		descriptorsLeft.release();
		descriptorsRight.release();
		matches.clear();

		imLeft = cv::imread(imgsLeft[i], CV_LOAD_IMAGE_UNCHANGED);
		imRight = cv::imread(imgsRight[i], CV_LOAD_IMAGE_UNCHANGED);


		//cv::cvtColor( img_1, imLeft, cv::COLOR_BGR2GRAY);
		//cv::cvtColor( img_2, imRight, cv::COLOR_BGR2GRAY);

		DetectFeatures(imLeft, imRight, keyPointsLeft, keyPointsRight);
		MakeDescriptors(imLeft, imRight, keyPointsLeft, keyPointsRight, descriptorsLeft, descriptorsRight);
		MatchKeyPoints(descriptorsLeft, descriptorsRight, matches);
		pose2d2d(keyPointsLeft, keyPointsRight, matches, R, t, intrinsicParams);


		cv::Mat imgMatches;
		cv::drawMatches(imLeft, keyPointsLeft, imRight, keyPointsRight,
						matches, imgMatches, cv::Scalar::all(-1),
						cv::Scalar::all(-1), std::vector<char>(),
						cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		cv::imshow("Good Matches", imgMatches);

		cv::waitKey(1000);


		/*for( int i = 0; i < (int)matches.size(); i++ )
		  {
			  printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n",
					  i, matches[i].queryIdx, matches[i].trainIdx );
		  }
		 */
		ComputeDepthMap(imLeft, imRight, filteredDisp, filteredDispVis);
	}

	return 0;
}

void DetectFeatures(const cv::Mat &imLeft, const cv::Mat &imRight,
					std::vector<cv::KeyPoint> &keyPointsLeft, std::vector<cv::KeyPoint> &keyPointsRight)
{
	int minHessian = 400;
	cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(minHessian);

	detector->detect(imLeft, keyPointsLeft);
	detector->detect(imRight, keyPointsRight);

	cv::Mat imKeyPointLeft, imKeyPointsRight;

	cv::drawKeypoints(imLeft, keyPointsLeft, imKeyPointLeft, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
	cv::drawKeypoints(imRight, keyPointsRight, imKeyPointsRight, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);

	//cv::imshow("keypoints 1", imKeyPointLeft);
	//cv::imshow("keypoints 2", imKeyPointsRight);

	//cv::waitKey(1000);
}

void MakeDescriptors(const cv::Mat &imLeft, const cv::Mat &imRight,
					 std::vector<cv::KeyPoint> &keyPointsLeft, std::vector<cv::KeyPoint> &keyPointsRight,
					 cv::Mat &descriptorsLeft, cv::Mat &descriptorsRight)
{
	int minHessian = 400;
	cv::Ptr<cv::xfeatures2d::SURF> descriptor = cv::xfeatures2d::SURF::create(minHessian);

	descriptor->compute(imLeft, keyPointsLeft, descriptorsLeft);
	descriptor->compute(imRight, keyPointsRight, descriptorsRight);

}

void MatchKeyPoints(cv::Mat &descriptorsLeft, cv::Mat &descriptorsRight, std::vector<cv::DMatch> &matches)
{
	double minDistance = 100;
	double maxDistance = 0;
	std::vector<std::vector<cv::DMatch> > allMathes;
	cv::FlannBasedMatcher matcher;

	//matcher.match(descriptorsLeft, descriptorsRight, allMathes);
	matcher.knnMatch(descriptorsLeft, descriptorsRight, allMathes, 2);

	for(size_t i =0; i < descriptorsLeft.rows; i++)
	{
		if(allMathes[i][0].distance < 0.6 * (allMathes[i][1].distance))
			matches.push_back(allMathes[i][0]);

	}
	/*
	for(size_t i = 0; i < descriptorsLeft.rows; i++)
	{
		double dist = allMathes[i].distance;

		if(dist < minDistance)
			minDistance = dist;
		if(dist > maxDistance)
			maxDistance = dist;
	}

	std::cout << "minDistance: " << minDistance << "\n";
	std::cout << "maxDistance: " << maxDistance << "\n";
	std::cout << "allMatches size: " << allMathes.size() << "\n";

	for(size_t i = 0; i < descriptorsLeft.rows; i++)
	{
		if(allMathes[i].distance <= max(2 * minDistance, 0.03))
		{
			std::cout << "Match num: " << i << "\n";
			matches.push_back(allMathes[i]);
		}

	}
*/
	std::cout << "Matches size: " << matches.size() << "\n";

}

void pose2d2d(const std::vector<cv::KeyPoint> &keyPointsLeft, const std::vector<cv::KeyPoint> &keyPointsRight,
		      const std::vector<cv::DMatch> &matches, cv::Mat &R, cv::Mat &t, const cv::Mat intrinsicParams)
{
	std::vector<cv::Point2f> pointsLeft;
	std::vector<cv::Point2f> pointsRight;
	cv::Mat essentialMatrix;
	cv::Mat R1, R2;
	double focalLength = intrinsicParams.at<double>(0,0);
	cv::Point2d principalPoint = cv::Point2d(intrinsicParams.at<double>(0,2), intrinsicParams.at<double>(1,2));

	for(size_t i = 0; i < matches.size(); i++)
	{
		pointsLeft.push_back(keyPointsLeft[matches[i].queryIdx].pt);
		pointsRight.push_back(keyPointsRight[matches[i].trainIdx].pt);

	}

	essentialMatrix = cv::findEssentialMat(pointsLeft, pointsRight, focalLength, principalPoint);

	std::cout << "pointsLeft: " << pointsLeft.size() << std::endl;
	std::cout << "pointsRight: " << pointsRight.size() << std::endl;

	std::cout << "essentialMatrix is " << essentialMatrix << std::endl;

	//cv::decomposeEssentialMat(essentialMatrix, R1, R2, t);

	cv::recoverPose(essentialMatrix, pointsLeft, pointsRight, R, t, focalLength, principalPoint);

	std::cout << "R is " << R << std::endl;
	std::cout << "t is " << t << std::endl;
}

void ComputeDepthMap(const cv::Mat &imLeft, const cv::Mat &imRight, cv::Mat &filtered_disp, cv::Mat &filtered_disp_vis)
{
	cv::Mat left_for_matcher, right_for_matcher;
	cv::Mat left_disp, right_disp;
	//cv::Mat filtered_disp_in;
	cv::Mat conf_map = cv::Mat(imLeft.rows, imLeft.cols, CV_8U);
	conf_map = cv::Scalar(255);
	cv::Rect ROI;
	cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter;
	int wsize = 3;
	double wls_lambda = 8000.0;
	double wls_sigma = 1.5;

	left_for_matcher = imLeft.clone();
	right_for_matcher = imRight.clone();

	cv::Ptr<cv::StereoSGBM> left_matcher = cv::StereoSGBM::create(0, 160, wsize,  0, 0, 0,
            													  0,  0, 0, 0, cv::StereoSGBM::MODE_HH);

	left_matcher->setP1(24*wsize*wsize);
	left_matcher->setP2(96*wsize*wsize);
	left_matcher->setPreFilterCap(63);
	left_matcher->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);
	wls_filter = cv::ximgproc::createDisparityWLSFilter(left_matcher);

	cv::Ptr<cv::StereoMatcher> right_matcher = cv::ximgproc::createRightMatcher(left_matcher);

	left_matcher->compute(left_for_matcher, right_for_matcher, left_disp);
	right_matcher->compute(right_for_matcher, left_for_matcher, right_disp);


	wls_filter->setLambda(wls_lambda);
	wls_filter->setSigmaColor(wls_sigma);
	wls_filter->filter(left_disp, imLeft, filtered_disp, right_disp);

    //std::cout << "Disparity map 1 : " << filtered_disp.size() << std::endl;

	conf_map = wls_filter->getConfidenceMap();
	ROI = wls_filter->getROI();


	//cv::Mat raw_disp_vis, filtered_disp_vis;
	double vis_mult = 1.0;

	//cv::ximgproc::getDisparityVis(left_disp, raw_disp_vis, vis_mult);
	//cv::namedWindow("raw disparity", cv::WINDOW_AUTOSIZE);
	//cv::imshow("raw disparity", raw_disp_vis);

	cv::ximgproc::getDisparityVis(filtered_disp, filtered_disp_vis, vis_mult);
	//cv::namedWindow("filtered disparity", cv::WINDOW_AUTOSIZE);
	//cv::imshow("filtered disparity", raw_disp_vis);
	//cv::waitKey();

}


void LoadImages(const std::string &imgPath, std::vector<std::string> &imgsLeft, std::vector<std::string> &imgsRight)
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
