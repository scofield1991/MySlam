#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>
#include <opencv2/calib3d.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include "g2o/core/base_vertex.h"
#include "g2o/core/base_unary_edge.h"
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/sba/types_six_dof_expmap.h>


#include <string>
#include <fstream>
#include <iomanip>
#include <map>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <chrono>

#include "PnPProblem.h"
#include "Utils.h"
#include "RobustMatcher.h"

using namespace std;

struct ImagePose
{
	cv::Mat imgLeft;
	cv::Mat deptMap;
	cv::Mat descriptorsLeft, descriptorsRight;
	std::vector<cv::KeyPoint> keyPointsLeft, keyPointsRight;
	std::vector<cv::DMatch> matchesLeftRight;
	std::vector<cv::DMatch> matchesPreviousCur;
	cv::Mat points4D;
	std::vector<cv::Point3f> points3D;
	//Matched keypoints in left image
	std::vector<cv::KeyPoint> keyPointsLeftMatched;
	cv::Mat descriptorsLeftMatched;
	//keypoints to 3d points keyPointIndex, landmarkIndex
	std::map<int, int> keyPointLandMark;
	// keypoint matches in other images (keyPointIndex, ImgIndex, KeyPOintIndex)
	std::map<int, std::map<int, int> > keyPointMatches;

    // Camera pose.
    cv::Mat Tcw;
	cv::Mat T;
	cv::Mat PLeft;
	cv::Mat PRight;

    // Rotation, translation and camera center
	//position of the world origin in camera's coordinate system
    cv::Mat Rcw;
    cv::Mat tcw;
    //camera's position in world coordinate system
    cv::Mat Rwc;
    cv::Mat Ow; //==twc

	int& KeyPointMatchesIndeces(int keyPointIndex, int imgIndex)
	{
		return keyPointMatches[keyPointIndex][imgIndex];
	}

	bool KeyPointMatchesExist(int keyPointIndex, int imgIndex)
	{
		return keyPointMatches[keyPointIndex].count(imgIndex);
	}

	int& KeyPoint3D(int keyPointIndex) { return keyPointLandMark[keyPointIndex]; }
	bool KeyPoint3DExist(int keyPointIndex) { return keyPointLandMark.count(keyPointIndex) > 0; }
};

struct CameraParams
{
    // Stereo baseline multiplied by fx.
    float mbf;

    // Stereo baseline in meters.
    float mb;

    cv::Mat mK;
    float fx;
    float fy;
    float cx;
    float cy;
    float invfx;
    float invfy;
    cv::Mat mDistCoef;
    float mThDepth;
};

struct LandMark
{
	cv::Point3f pt;
	int seen = 0;
};

std::vector<ImagePose> imgPoses;
std::vector<LandMark> landMarks;

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
void triangulateKeyPoints(ImagePose &prev, ImagePose &cur, cv::Mat &points4D, std::vector<cv::Point2f> &pointsCur,
						  std::vector<cv::Point2f> &pointsPrev, std::vector<int> &pointsCurIdx,
						  std::vector<int> &pointsPrevIdx, int img_num, std::vector<size_t> &keyPointsUsed);
void triangulateKeyPointsLeftRight(ImagePose &cur, std::vector<cv::Point2f> &pointsLeft, std::vector<cv::Point2f> &pointsRight);
int rescaleKeyPoints(ImagePose &prev, ImagePose &cur, const std::vector<cv::Point2f> &pointsCurMatched,
					 const std::vector<cv::Point2f> &pointsPrevMatched, const cv::Mat &points4D,
					 std::vector<size_t> &keyPointsUsed, std::vector<LandMark> &landMarks,  int img_num);
void goodTriangulatedPoints(ImagePose &prev, ImagePose &cur, std::vector<size_t> &keyPointsUsed,
							cv::Mat &points4D, std::vector<LandMark> &landMarks,  int img_num);
void get3D2Dpoints(ImagePose &prev, ImagePose &cur, std::vector<cv::Point3f> &points3D, std::vector<cv::Point2f> &points2D);
void setPose(cv::Mat T, ImagePose &ip);
void updatePoseMatrices(ImagePose &ip);
void UnprojectStereo(ImagePose &cur, std::vector<cv::Point2f> &pointsLeft, std::vector<cv::Point2f> &pointsRight,
		                cv::Mat &filteredDisp, CameraParams &camPms);
void bundleAdjustment (const std::vector< cv::Point3f > points_3d, const std::vector< cv::Point2f > points_2d,
	                   const cv::Mat& K, cv::Mat& R, cv::Mat& t);
cv::Mat toCvMat(const Eigen::Matrix<double,4,4> &m);
void find_feature_matches ( const cv::Mat& img_1, const cv::Mat& img_2, std::vector<cv::KeyPoint>& keypoints_1,
                            std::vector<cv::KeyPoint>& keypoints_2, std::vector<cv::DMatch>& matches );
cv::Point2d pixel2cam ( const cv::Point2d& p, const cv::Mat& K );



int main(int argc, char** argv) {

	// For visualization
	cv::namedWindow("Trajectory", cv::WINDOW_AUTOSIZE);
	cv::Mat traj = cv::Mat::zeros(600, 600, CV_8UC3);
	char text[100];
	int fontFace = cv::FONT_HERSHEY_PLAIN;
	double fontScale = 1;
	int thickness = 1;
	cv::Point textOrg(10, 50);
	cv::Mat frameVis;
	cv::Scalar red(0, 0, 255);

	std::vector<std::string> imgsLeft;
	std::vector<std::string> imgsRight;
	cv::Mat imLeftPrev, imLeft, imRight;
	cv::Mat img_1, img_2;

	//Camera intrinsic params
	cv::Mat intrinsicParams  = (cv::Mat_<float>(3, 3) << 707.0912, 0, 601.8873, 0, 707.0912, 183.1104, 0, 0, 1);
	cv::Mat intrinsicParamsRight  = (cv::Mat_<float>(3, 4) << 707.0912, 0, 601.8873, 379,8145, \
																0, 707.0912, 183.1104, 0, \
																0, 0, 1, 0);

	//cv::Mat intrinsicParamsRight  = (cv::Mat_<double>(3, 4) << 0, 0, 0, -379.8145, \
															   0, 0, 0, 0, \
															   0, 0, 0, 0);

	CameraParams camPms;

	camPms.fx = 707.0912;
	camPms.fy = 707.0912;
	camPms.cx = 601.8873;
	camPms.cy = 183.1104;
	camPms.invfx = 1.0f / camPms.fx;
	camPms.invfy = 1.0f / camPms.fy;
	camPms.mbf = 379,8145;
	camPms.mb = camPms.mbf / camPms.fx;
	camPms.mThDepth = camPms.mbf * 45.0f / camPms.fx;

	cout << endl << "Depth Threshold (Close/Far Points): " << camPms.mThDepth << std::endl;

	//for PnP Solver
	double paramsCam[] = { camPms.fx, camPms.fy, camPms.cx, camPms.cy };
	PnPProblem pnpDetection(paramsCam);
	int pnpMethod = cv::SOLVEPNP_ITERATIVE;
	cv::Mat inliersIdx;

	//RANSAC parameters
	int iterationsCount = 500;
	float reprojectionError = 2.0;
	double confidence = 0.95;

	// Robust Matcher parameters
	int numKeyPoints = 2000;
	float ratioTest = 0.70f;
	RobustMatcher rmacher;


	LoadImages(argv[1], imgsLeft, imgsRight);

	int NumImgs = imgsLeft.size();

	std::cout << "NumImgs: " << NumImgs << "\n";

	for(size_t i = 0; i < NumImgs; i++)
	{

		std::vector<cv::Point2f> pointsLeftMatched, pointsRightMatched;
		std::vector<cv::Point3f> points3D;
		std::vector<cv::Point2f> points2D;
		// Rotation and tranclstion matrices
		cv::Mat R, t;
		// Projective matrix
		cv::Mat P(3, 4, CV_32F);

		std::vector<cv::KeyPoint> keyPointsLeft, keyPointsRight;
		std::vector<int> keyPointsUsed;
		cv::Mat descriptorsLeft, descriptorsRight;
		cv::Mat filteredDisp, filteredDispVis;
		std::vector<cv::DMatch> matches;

		// Matched keypoints from previous and current image
		std::vector<cv::Point2f> pointsCurMatched;
		std::vector<cv::Point2f> pointsPrevMatched;

		std::vector<int> pointsCurIdx, pointsPrevIdx;

		cv::Mat points4D;
		ImagePose cur;
		cv::Mat R_local, t_local;
		// Transformation matrix
		cv::Mat T = cv::Mat::eye(4, 4, CV_32F);

		if(i == 0)
		{
			cur.T = cv::Mat::eye(4, 4, CV_32F);
			setPose(cur.T, cur);
			cur.PLeft = intrinsicParams * cv::Mat::eye(3, 4, CV_32F);
			cur.PRight = cur.PLeft + intrinsicParamsRight;

			//std::cout << "cur.PLeft: " << cur.PLeft << "\n";
			//std::cout << "cur.PRight: " << cur.PRight << "\n";
			//cur.PRight = intrinsicParamsRight;
		}


		imLeft = cv::imread(imgsLeft[i], CV_LOAD_IMAGE_UNCHANGED);
		imRight = cv::imread(imgsRight[i], CV_LOAD_IMAGE_UNCHANGED);
		cur.imgLeft = imLeft;
		frameVis = imLeft.clone();

		//cv::cvtColor( img_1, imLeft, cv::COLOR_BGR2GRAY);
		//cv::cvtColor( img_2, imRight, cv::COLOR_BGR2GRAY);

		DetectFeatures(imLeft, imRight, cur.keyPointsLeft, cur.keyPointsRight);

		MakeDescriptors(imLeft, imRight, cur.keyPointsLeft, cur.keyPointsRight, cur.descriptorsLeft, cur.descriptorsRight);

		MatchKeyPoints(cur.descriptorsLeft, cur.descriptorsRight, cur.matchesLeftRight);

		ComputeDepthMap(imLeft, imRight, filteredDisp, filteredDispVis);
		filteredDisp.copyTo(cur.deptMap);

		if(i > 0)
		{
			ImagePose &prev = imgPoses[i-1];
			std::cout << "imgPoses[i-1].T:  " << imgPoses[i-1].T << "\n";

			cv::Mat imLeftPrev = cv::imread(imgsLeft[i-1], CV_LOAD_IMAGE_UNCHANGED);
			std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
    	    std::vector<cv::DMatch> matchesNew;
    	    find_feature_matches ( imLeftPrev, imLeft, keypoints_1, keypoints_2, matchesNew );
    		std::cout<<"matchesNew.size: "<< matchesNew.size() << std::endl;

    		std::vector<cv::Point3f> pts_3d;
    		std::vector<cv::Point2f> pts_2d;
    		for ( cv::DMatch m:matchesNew )
    		{
        		//ushort d = d1.ptr<unsigned short> (int ( keypoints_1[m.queryIdx].pt.y )) [ int ( keypoints_1[m.queryIdx].pt.x ) ];
        		ushort d = prev.deptMap.ptr<unsigned short> (int ( keypoints_1[m.queryIdx].pt.y )) [ int ( keypoints_1[m.queryIdx].pt.x ) ];
        		if ( d == 0 )   // bad depth
        		{
        			std::cout << "d = 0 \n";
            		continue;
        		}
        		float dd = d/16.0;
        		cv::Point2d p1 = pixel2cam ( keypoints_1[m.queryIdx].pt, intrinsicParams );
        		pts_3d.push_back ( cv::Point3f ( p1.x*dd, p1.y*dd, dd ) );
        		pts_2d.push_back ( keypoints_2[m.trainIdx].pt );
    		}

    		    cout<<"3d-2d pairs: "<<pts_3d.size() <<endl;

    			//Mat r, t;
    			//solvePnP ( pts_3d, pts_2d, K, Mat(), r, t, false ); // 调用OpenCV 的 PnP 求解，可选择EPNP，DLS等方法
    			//Mat R;
    			//cv::Rodrigues ( r, R ); // r为旋转向量形式，用Rodrigues公式转换为矩阵

    			//cout<<"R="<<endl<<R<<endl;
    			//cout<<"t="<<endl<<t<<endl;

   			    //cout<<"calling bundle adjustment"<<endl;

    			//bundleAdjustment ( pts_3d, pts_2d, K, R, t );



			MatchKeyPoints(prev.descriptorsLeftMatched, cur.descriptorsLeft, cur.matchesPreviousCur);

			std::cout << "Matches size: " << cur.matchesPreviousCur.size() << "\n";

			cv::Mat imgMatches;
			cv::drawMatches( prev.imgLeft, prev.keyPointsLeftMatched, imLeft, cur.keyPointsLeft,
					        cur.matchesPreviousCur, imgMatches, cv::Scalar::all(-1),
							cv::Scalar::all(-1), std::vector<char>(),
							cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
			cv::imshow("Good Matches", imgMatches);

			cv::waitKey(1);

			get3D2Dpoints(prev, cur, points3D, points2D);

			std::cout << "points3D.size(): " << points3D.size() << "\n";
			std::cout << "points3D.size(): " << points3D.size() << "\n";

			cv::Mat rvec(3,1,cv::DataType<float>::type);
			cv::Mat tvec(3,1,cv::DataType<float>::type);
			cv::Mat distCoeffs = cv::Mat::zeros(4, 1, cv::DataType<float>::type);
			cv::Mat cameraMatrix = intrinsicParams;

			pnpDetection.estimatePoseRANSAC(points3D, points2D, pnpMethod, inliersIdx,
											iterationsCount, reprojectionError, confidence);

			std::vector<cv::Point2f> points2DInliers;
			for(int inliersIndex = 0; inliersIndex < inliersIdx.rows; inliersIndex++)
			{
				int n = inliersIdx.at<int>(inliersIndex);
				cv::Point2f point2D = points2D[n];
				points2DInliers.push_back(point2D);
			}

			std::cout << "pnpDetection.get_t_matrix(): " << pnpDetection.get_t_matrix() << "\n";

			std::cout << "points2DInliers.size" << points2DInliers.size() << "\n";
			draw2DPoints(frameVis, points2DInliers, red);
			cv::imshow("Show inliers", frameVis);

			//cv::solvePnPRansac(points3D, points2D, intrinsicParams, distCoeffs, rvec, tvec);
			//cv::Rodrigues(rvec, R_local);
			//t_local = tvec;
			R_local = pnpDetection.get_R_matrix();
			t_local = pnpDetection.get_t_matrix();
			std::cout << "t_local: " << t_local << "\n";

			cout<<"calling bundle adjustment"<<endl;

            bundleAdjustment ( points3D, points2D, intrinsicParams, R_local, t_local );

			R_local.copyTo(T(cv::Range(0, 3), cv::Range(0, 3)));
			t_local.copyTo(T(cv::Range(0, 3), cv::Range(3, 4)));

			std::cout << "prev.Tcw: " << prev.Tcw << "\n";
			cur.T = prev.Tcw * T;
			

			setPose(cur.T, cur);
			std::cout << "cur.Tcw: " << cur.Tcw << "\n";
/*
			R_local = R_local.t();
			std::cout << "tvec: " << tvec;
			t_local = -R_local * tvec;
			//t_local = tvec;
			std::cout << "t_local: " << t_local;

			//pose2d2d(cur.keyPointsLeft, prev.keyPointsLeft, matches, R_local, t_local, intrinsicParams);



			cur.T = prev.T * T;

			std::cout << "cur.T:  " << cur.T << "\n";
			*/
			//R = cur.T(cv::Range(0, 3), cv::Range(0, 3));
			//t = cur.T(cv::Range(0, 3), cv::Range(3, 4));

			//P(cv::Range(0, 3), cv::Range(0, 3)) = cur.Rwc;
			//P(cv::Range(0, 3), cv::Range(3, 4)) = cur.Ow;
			//P = intrinsicParams * P;
			//cur.PLeft = P;
			//cur.PRight = cur.PLeft + intrinsicParamsRight;

			float x = cur.tcw.at<float>(0) + 300.0;
			float y = cur.tcw.at<float>(2) + 300.0;
			cv::circle(traj, cv::Point(x, y), 1, CV_RGB(255, 0, 0), 2);
			cv::rectangle(traj, cv::Point(10, 30), cv::Point(550, 50), CV_RGB(0, 0, 0), CV_FILLED);
			std::cout << "x, y: " << x << " " << y << "\n";
			sprintf(text, "Coordinates: x=%02fm, y=%02fm, z=%02fm", cur.tcw.at<float>(0), cur.tcw.at<float>(1), cur.tcw.at<float>(2));
			cv::putText(traj, text, textOrg, fontFace, fontScale, cv::Scalar::all(255), thickness, 8);

			cv::imshow("Trajectory", traj);
			cv::waitKey(1);


			/*

			triangulateKeyPoints(prev, cur, points4D, pointsCurMatched, pointsPrevMatched, pointsCurIdx,
								 pointsPrevIdx, i, keyPointsUsed);
			std::cout << points4D << "\n";

			if(i > 1)
			{
				double scale;
				scale = rescaleKeyPoints(prev, cur, pointsCurMatched, pointsPrevMatched,
						                 points4D, keyPointsUsed, landMarks, i);

				t_local *= scale;

				t_local.copyTo(T(cv::Range(0, 3), cv::Range(3, 4)));

				cur.T = prev.T * T;

				R = cur.T(cv::Range(0, 3), cv::Range(0, 3));
				t = cur.T(cv::Range(0, 3), cv::Range(3, 4));

				P(cv::Range(0, 3), cv::Range(0, 3)) = R.t();
				P(cv::Range(0, 3), cv::Range(3,4 )) = -R.t() * t;
				P = intrinsicParams * P;
				cur.PLeft = P;


				//triangulateKeyPoints(prev, cur, points4D, pointsCurMatched, pointsPrevMatched, pointsCurIdx,
				//								 pointsPrevIdx, i, keyPointsUsed);

				cv::triangulatePoints(prev.PLeft, cur.PLeft, pointsPrevMatched, pointsCurMatched, points4D);

			}
			goodTriangulatedPoints(prev, cur, keyPointsUsed, points4D, landMarks, i);
			*/
		}

		UnprojectStereo(cur, pointsLeftMatched, pointsRightMatched, filteredDisp, camPms);
		std::cout << "points3D.size: " << cur.points3D.size() << "\n";
		std::cout << "cur.keyPointsLeftMatched.size: " << cur.keyPointsLeftMatched.size() << "\n";

		//triangulateKeyPointsLeftRight(cur, pointsLeftMatched, pointsRightMatched);
		imgPoses.push_back(cur);



		/*for( int i = 0; i < (int)matches.size(); i++ )
		  {
			  printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n",
					  i, matches[i].queryIdx, matches[i].trainIdx );
		  }
		 */

	}

	return 0;
}

/*
void triangulateKeyPoints(const cv::Mat &ProjMatrixPrev, const cv::Mat &ProjMatrixCur,
						  const std::vector<cv::KeyPoint> &keyPointsPrev, const std::vector<cv::KeyPoint> &keyPointsCur,
	                      const std::vector<cv::DMatch> &matches, cv::Mat &points4D, std::vector<cv::Point2f> &pointsCur,
						  std::vector<cv::Point2f> &pointsPrev, std::vector<int> &pointsCurIdx,
						  std::vector<int> &pointsPrevIdx, int img_num)
{

	for(size_t i = 0; i < matches.size(); i++)
	{
			pointsCur.push_back(keyPointsCur[matches[i].queryIdx].pt);
			pointsPrev.push_back(keyPointsPrev[matches[i].trainIdx].pt);

			pointsCurIdx.push_back(matches[i].queryIdx);
			pointsPrevIdx.push_back(matches[i].trainIdx);
	}


	cv::triangulatePoints(ProjMatrixPrev, ProjMatrixCur, pointsPrev, pointsPrev, points4D);
}
*/

cv::Point2d pixel2cam ( const cv::Point2d& p, const cv::Mat& K )
{
    return cv::Point2d
           (
               ( p.x - K.at<double> ( 0,2 ) ) / K.at<double> ( 0,0 ),
               ( p.y - K.at<double> ( 1,2 ) ) / K.at<double> ( 1,1 )
           );
}

void find_feature_matches ( const cv::Mat& img_1, const cv::Mat& img_2,
                            std::vector<cv::KeyPoint>& keypoints_1,
                            std::vector<cv::KeyPoint>& keypoints_2,
                            std::vector<cv::DMatch>& matches )
{
    //-- 初始化
    cv::Mat descriptors_1, descriptors_2;
    // used in OpenCV3
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    // use this if you are in OpenCV2
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
    cv::Ptr<cv::DescriptorMatcher> matcher  = cv::DescriptorMatcher::create ( "BruteForce-Hamming" );
    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect ( img_1,keypoints_1 );
    detector->detect ( img_2,keypoints_2 );

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute ( img_1, keypoints_1, descriptors_1 );
    descriptor->compute ( img_2, keypoints_2, descriptors_2 );

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    std::vector<cv::DMatch> match;
    // BFMatcher matcher ( NORM_HAMMING );
    matcher->match ( descriptors_1, descriptors_2, match );

    //-- 第四步:匹配点对筛选
    double min_dist=10000, max_dist=0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = match[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        if ( match[i].distance <= max ( 2*min_dist, 30.0 ) )
        {
            matches.push_back ( match[i] );
        }
    }
}


void bundleAdjustment (
    const std::vector< cv::Point3f > points_3d,
    const std::vector< cv::Point2f > points_2d,
    const cv::Mat& K, cv::Mat& R, cv::Mat& t )
{
    // 初始化g2o
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<6,3> > Block;  // pose 维度为 6, landmark 维度为 3
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverCSparse<Block::PoseMatrixType>(); // 线性方程求解器
    Block* solver_ptr = new Block ( linearSolver );     // 矩阵块求解器
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( solver_ptr );
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm ( solver );

    // vertex
    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap(); // camera pose
    Eigen::Matrix3d R_mat;
    R_mat <<
          R.at<double> ( 0,0 ), R.at<double> ( 0,1 ), R.at<double> ( 0,2 ),
               R.at<double> ( 1,0 ), R.at<double> ( 1,1 ), R.at<double> ( 1,2 ),
               R.at<double> ( 2,0 ), R.at<double> ( 2,1 ), R.at<double> ( 2,2 );
    pose->setId ( 0 );
    pose->setEstimate ( g2o::SE3Quat (
                            R_mat,
                            Eigen::Vector3d ( t.at<double> ( 0,0 ), t.at<double> ( 1,0 ), t.at<double> ( 2,0 ) )
                        ) );
    optimizer.addVertex ( pose );

    int index = 1;
    for ( const cv::Point3f p:points_3d )   // landmarks
    {
        g2o::VertexSBAPointXYZ* point = new g2o::VertexSBAPointXYZ();
        point->setId ( index++ );
        point->setEstimate ( Eigen::Vector3d ( p.x, p.y, p.z ) );
        point->setMarginalized ( true ); // g2o 中必须设置 marg 参见第十讲内容
        optimizer.addVertex ( point );
    }

    // parameter: camera intrinsics
    g2o::CameraParameters* camera = new g2o::CameraParameters (
        K.at<double> ( 0,0 ), Eigen::Vector2d ( K.at<double> ( 0,2 ), K.at<double> ( 1,2 ) ), 0
    );
    camera->setId ( 0 );
    optimizer.addParameter ( camera );

    // edges
    index = 1;
    for ( const cv::Point2f p:points_2d )
    {
        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
        edge->setId ( index );
        edge->setVertex ( 0, dynamic_cast<g2o::VertexSBAPointXYZ*> ( optimizer.vertex ( index ) ) );
        edge->setVertex ( 1, pose );
        edge->setMeasurement ( Eigen::Vector2d ( p.x, p.y ) );
        edge->setParameterId ( 0,0 );
        edge->setInformation ( Eigen::Matrix2d::Identity() );
        optimizer.addEdge ( edge );
        index++;
    }

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.setVerbose ( false );
    optimizer.initializeOptimization();
    optimizer.optimize ( 100 );
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> ( t2-t1 );
    cout<<"optimization costs time: "<<time_used.count() <<" seconds."<<endl;

    cout<<endl<<"after optimization:"<<endl;
    cout<<"T="<<endl<<Eigen::Isometry3d ( pose->estimate() ).matrix() <<endl;
    cv::Mat T = toCvMat(Eigen::Isometry3d ( pose->estimate() ).matrix());

    T(cv::Range(0, 3), cv::Range(0, 3)).copyTo(R);
    T(cv::Range(0, 3), cv::Range(3, 4)).copyTo(t);

    			//R_local.copyTo(T(cv::Range(0, 3), cv::Range(0, 3)));
			//t_local.copyTo(T(cv::Range(0, 3), cv::Range(3, 4)));
}

cv::Mat toCvMat(const Eigen::Matrix<double,4,4> &m)
{
    cv::Mat cvMat(4,4,CV_32F);
    for(int i=0;i<4;i++)
        for(int j=0; j<4; j++)
            cvMat.at<float>(i,j)=m(i,j);

    return cvMat.clone();
}

void setPose(cv::Mat T, ImagePose &ip)
{
	ip.Tcw = T.clone();
	updatePoseMatrices(ip);

}

void updatePoseMatrices(ImagePose &ip)
{
    ip.Rcw = ip.Tcw.rowRange(0,3).colRange(0,3);
    ip.Rwc = ip.Rcw.t();
    ip.tcw = ip.Tcw.rowRange(0,3).col(3);
    ip.Ow = -ip.Rcw.t()* ip.tcw;

}

void UnprojectStereo(ImagePose &cur, std::vector<cv::Point2f> &pointsLeft, std::vector<cv::Point2f> &pointsRight,
		                cv::Mat &filteredDisp, CameraParams &camPms)
{

	for(size_t i = 0; i < cur.matchesLeftRight.size(); i++)
	{
		cv::KeyPoint kpLeft = cur.keyPointsLeft[cur.matchesLeftRight[i].queryIdx];
		const float u = kpLeft.pt.x;
		const float v = kpLeft.pt.y;

        short pixVal = filteredDisp.at<short>(v, u);
        float disparity = pixVal / 16.0f;
        //std::cout << "disparity: " << disparity << "\n";
        const float z  = camPms.mbf / disparity;
        //std::cout << "z: " << z << "\n";

        if(z > 0 && z < camPms.mThDepth)
        {

        	const float x = (u - camPms.cx) * z * camPms.invfx;
        	const float y = (v - camPms.cy) * z * camPms.invfy;
        	cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);
        	//x3Dc = cur.Rwc * x3Dc + cur.Ow;
        	cv::Point3f x3D;
        	x3D.x = x3Dc.at<float>(0, 0);
        	x3D.y = x3Dc.at<float>(1, 0);
        	x3D.z = x3Dc.at<float>(2, 0);
        	cur.points3D.push_back(x3D);

    		pointsLeft.push_back(cur.keyPointsLeft[cur.matchesLeftRight[i].queryIdx].pt);
    		pointsRight.push_back(cur.keyPointsRight[cur.matchesLeftRight[i].trainIdx].pt);

    		cur.keyPointsLeftMatched.push_back(cur.keyPointsLeft[cur.matchesLeftRight[i].queryIdx]);
    		cur.descriptorsLeftMatched.push_back(cur.descriptorsLeft.row(cur.matchesLeftRight[i].queryIdx));

        }




	}
/*
	for(size_t i = 0; i < cur.keyPointsLeftMatched.size(); i++)
	{
		const float u = cur.keyPointsLeftMatched[i].pt.x;
		const float v = cur.keyPointsLeftMatched[i].pt.y;

        short pixVal = filteredDisp.at<short>(v, u);
        float disparity = pixVal / 16.0f;
        std::cout << "disparity: " << disparity << "\n";

        const float z  = camPms.mbf / disparity;
        std::cout << "z: " << z << "\n";

        //else
        //{
        //	std::cout << "dosparity is  < 0: " << disparity << "\n";
        //	const float z  = camPms.mbf / 0.01;
        //}
        if(z > 0)
        {
        	const float x = (u - camPms.cx) * z * camPms.invfx;
        	const float y = (v - camPms.cy) * z * camPms.invfy;
        	cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);
        	x3Dc = cur.Rwc * x3Dc + cur.Ow;
        	cv::Point3f x3D;
        	x3D.x = x3Dc.at<float>(0, 0);
        	x3D.y = x3Dc.at<float>(1, 0);
        	x3D.z = x3Dc.at<float>(2, 0);
        	points3D.push_back(x3D);
        }
        else
        {
        	std::cout << "z is  < 0: " << z << "\n";
        }
	}
*/
}

void get3D2Dpoints(ImagePose &prev, ImagePose &cur, std::vector<cv::Point3f> &points3D, std::vector<cv::Point2f> &points2D)
{
	for(int i = 0; i < cur.matchesPreviousCur.size(); i++)
	{
		int matchQueryIdx = cur.matchesPreviousCur[i].queryIdx;
		int matchTrainIdx = cur.matchesPreviousCur[i].trainIdx;

		//cv::Point3f pt3d;
		//pt3d.x = prev.points4D.at<float>(0, matchQueryIdx) / prev.points4D.at<float>(3, matchQueryIdx);
		//pt3d.y = prev.points4D.at<float>(1, matchQueryIdx) / prev.points4D.at<float>(3, matchQueryIdx);
		//pt3d.z = prev.points4D.at<float>(2, matchQueryIdx) / prev.points4D.at<float>(3, matchQueryIdx);

		//std::cout << "pt3d.x: " << pt3d.x <<  "\n";
		//std::cout << "pt3d.y: " << pt3d.y <<  "\n";
		//std::cout << "pt3d.z: " << pt3d.z <<  "\n";

		points3D.push_back(prev.points3D[matchQueryIdx]);
		points2D.push_back(cur.keyPointsLeft[matchTrainIdx].pt);

	}
}

void goodTriangulatedPoints(ImagePose &prev, ImagePose &cur, std::vector<size_t> &keyPointsUsed,
							cv::Mat &points4D, std::vector<LandMark> &landMarks,  int img_num)
{
	for(size_t i = 0; i< keyPointsUsed.size(); i++)
	{
		int k = keyPointsUsed[i];
		int matchIndex = prev.KeyPointMatchesIndeces(k, img_num);

		cv::Point3f pt3d;
		pt3d.x = points4D.at<float>(0, i) / points4D.at<float>(3, i);
		pt3d.y = points4D.at<float>(1, i) / points4D.at<float>(3, i);
		pt3d.z = points4D.at<float>(2, i) / points4D.at<float>(3, i);

		if(prev.KeyPoint3DExist(k))
		{
			cur.KeyPoint3D(matchIndex) = prev.KeyPoint3D(k);
			landMarks[prev.KeyPoint3D(k)].pt += pt3d;
			landMarks[cur.KeyPoint3D(matchIndex)].seen++;
		}
		else
		{
			LandMark landMark;
			landMark.pt = pt3d;
			landMark.seen = 2;
			landMarks.push_back(landMark);

			//std::cout << "landmark: " << landMark.pt << "\n";

			prev.KeyPoint3D(k) = landMarks.size() - 1;
			cur.KeyPoint3D(matchIndex) = landMarks.size() - 1;
		}
	}
}

void triangulateKeyPoints(ImagePose &prev, ImagePose &cur, cv::Mat &points4D, std::vector<cv::Point2f> &pointsCur,
						  std::vector<cv::Point2f> &pointsPrev, std::vector<int> &pointsCurIdx,
						  std::vector<int> &pointsPrevIdx, int img_num, std::vector<size_t> &keyPointsUsed)
{

	// find keyPoints that matched in current and previous images
	for(size_t i = 0; i < cur.matchesPreviousCur.size(); i++)
	{
			pointsCur.push_back(cur.keyPointsLeft[cur.matchesPreviousCur[i].queryIdx].pt);
			pointsPrev.push_back(prev.keyPointsLeft[cur.matchesPreviousCur[i].trainIdx].pt);

			pointsCurIdx.push_back(cur.matchesPreviousCur[i].queryIdx);
			pointsPrevIdx.push_back(cur.matchesPreviousCur[i].trainIdx);

	}

	for(size_t i = 0; i < cur.matchesPreviousCur.size(); i++)
	{
		cur.KeyPointMatchesIndeces(pointsCurIdx[i], img_num-1) = pointsPrevIdx[i];
		prev.KeyPointMatchesIndeces(pointsPrevIdx[i], img_num) = pointsCurIdx[i];

	}

	for(size_t i = 0; i < prev.keyPointsLeft.size(); i++)
	{
		if(prev.KeyPointMatchesExist(i, img_num))
			keyPointsUsed.push_back(i);
	}

	std::cout << "keyPointsUsed: " << keyPointsUsed.size() << "\n";
	std::cout << "prev.keyPointsLeft: " << prev.keyPointsLeft.size() << "\n";
	std::cout << "prev.P: " << prev.PLeft << "\n";
	std::cout << "cur.P: " << cur.PLeft << "\n";

	cv::triangulatePoints(prev.PLeft, cur.PLeft, pointsPrev, pointsCur, points4D);

    // Scale the new 3d points to be similar to the existing 3d points (landmark)
    // Use ratio of distance between pairing 3d points

}

void triangulateKeyPointsLeftRight(ImagePose &cur, std::vector<cv::Point2f> &pointsLeft, std::vector<cv::Point2f> &pointsRight)
{
	// find keyPoints that matched in left and right images
	for(size_t i = 0; i < cur.matchesLeftRight.size(); i++)
	{
		pointsLeft.push_back(cur.keyPointsLeft[cur.matchesLeftRight[i].queryIdx].pt);
		pointsRight.push_back(cur.keyPointsRight[cur.matchesLeftRight[i].trainIdx].pt);

		cur.keyPointsLeftMatched.push_back(cur.keyPointsLeft[cur.matchesLeftRight[i].queryIdx]);
		cur.descriptorsLeftMatched.push_back(cur.descriptorsLeft.row(cur.matchesLeftRight[i].queryIdx));

	}

	std::cout << "cur.keyPointsLeftMatched.size: " << cur.keyPointsLeftMatched.size() << "\n";
	//cv::Mat descs(cur.descriptorsLeftMatched[0]);
	std::cout << "cur.descriptorsLeftMatched.size: " << cur.descriptorsLeftMatched.size() << "\n";


	cv::triangulatePoints(cur.PLeft, cur.PRight, pointsLeft, pointsRight, cur.points4D);
	std::cout << "cur.points4D.size: " << cur.points4D.size() << "\n";

}

int rescaleKeyPoints(ImagePose &prev, ImagePose &cur, const std::vector<cv::Point2f> &pointsCurMatched,
					 const std::vector<cv::Point2f> &pointsPrevMatched, const cv::Mat &points4D,
					 std::vector<size_t> &keyPointsUsed, std::vector<LandMark> &landMarks,  int img_num)
{
	double scale = 0;
	int count = 0;

	cv::Point3f prevCamera;

	prevCamera.x = prev.T.at<double>(0, 3);
	prevCamera.y = prev.T.at<double>(1, 3);
	prevCamera.z = prev.T.at<double>(2, 3);

	std::vector<cv::Point3f> newPts;
	std::vector<cv::Point3f> existingPts;

	for(int i = 0; i < keyPointsUsed.size(); i++)
	{
	   int k = keyPointsUsed[i];

		if(prev.KeyPointMatchesExist(k, img_num) && prev.KeyPoint3DExist(k))
		{
			cv::Point3f pt3d;

			std::cout << "Matches exist!" << "\n";
			pt3d.x = points4D.at<float>(0, i) / points4D.at<float>(3, i);
			pt3d.y = points4D.at<float>(1, i) / points4D.at<float>(3, i);
			pt3d.z = points4D.at<float>(2, i) / points4D.at<float>(3, i);

			int index = prev.KeyPoint3D(k);

			cv::Point3f avgLandMark = landMarks[index].pt / (landMarks[index].seen - 1);

			std::cout << "pt3d: " << pt3d << "\n";
			std::cout << "avgLandMark: " << avgLandMark << "\n";

			newPts.push_back(pt3d);
			existingPts.push_back(avgLandMark);

		}
		//int index = prev.KeyPoint3D();
	}

	for(int i = 0; i < newPts.size() - 1; i++)
	{
		for(int k = i+1; k < newPts.size(); k++)
		{
			double s = cv::norm(existingPts[i] - existingPts[k]) / cv::norm(newPts[i] - newPts[k]);

			scale += s;
			count++;
		}
	}

	assert(count > 0);

	scale /= count;

	std::cout << "image: " << img_num << "scale: " << scale << "\n";

	return scale;

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
	std::cout << "keyPointsLeft.size: " << keyPointsLeft.size() << "\n";
	cv::Mat descT = descriptorsLeft.t();
	std::cout << "descriptorsLeft.size: " << descriptorsLeft.size() << "\n";
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
		if(allMathes[i][0].distance < 0.7 * (allMathes[i][1].distance))
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
	std::string timesPath = imgPath + "/timestamps.txt";
	std::string imgsLeftPath = imgPath + "/image_00/data/";
	std::string imgsRightPath = imgPath + "/image_01/data/";
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
		ss << std::setfill('0') << std::setw(10) << i;

		imgsLeft[i] = imgsLeftPath + ss.str() + ".png";
		imgsRight[i] = imgsRightPath + ss.str() + ".png";
		//std::cout << imgsLeft[i] << "\n";

	}
}
