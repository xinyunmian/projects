#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

#include <vector>
#include <cstddef>
#include <Eigen/Core>

//#include "trackMain/kalmanfilter.h"
//#include "trackMain/track.h"
//
//#include "trackMain/model2.h"

typedef unsigned char uintb8;

#ifndef __deepSortDll_H__
#define __deepSortDll_H__
typedef struct _DeepSortParams
{
	char MetaPath[256];        //模型结构文件
	char CkptPath[256];        //模型参数文件
	tensorflow::string nameO;   //网络输出层名称
	tensorflow::string nameI;  //数据输入层名称
}DeepSortParams;

typedef Eigen::Matrix<float, 1, 4, Eigen::RowMajor> DETECTBOX;
typedef Eigen::Matrix<float, 1, 128, Eigen::RowMajor> FEATURE;


class DETECTION_ROW {
public:
	DETECTBOX tlwh; //np.float  (x, y, w, h)
	float confidence; //float
	char *objName;
	FEATURE feature; //np.float32
	//DETECTBOX to_xyah() const;   //(center x, center y, aspect ratio, height)
	//DETECTBOX to_tlbr() const;   //(top left, bottom right)
};

typedef std::vector<DETECTION_ROW> DETECTIONS;



bool Modelinit(DeepSortParams* pSortParams);
void tobuffer(const std::vector<cv::Mat> &imgs, uintb8 *buf);

bool getRectsFeature(const cv::Mat& img, DETECTIONS& d);

int feature_dim;
tensorflow::Session* session;
std::vector<tensorflow::Tensor> output_tensors;
std::vector<tensorflow::string> outnames;
tensorflow::string input_layer;


//typedef struct _TrackerParams
//{
//	float max_iou_distance;   //IOU比较相似性的阈值
//	int max_age;              //最大允许连续未成功匹配帧数
//	int n_init;               //连续成功匹配n_init帧数，则确定为成功匹配
//	int _next_idx;            //目标跟踪ID
//	int n_feature;            //保存最近帧数下成功匹配的box特征，进行下一帧跟踪匹配
//	float cosd;               //box匹配时的cosin距离
//}TrackerParams;
//
//KalmanFiltertb* kf;
//std::vector<Track> tracks;
//void predict();
//void update(const DETECTIONS& detections);
//void Initracker(TrackerParams* pTrackParams);
//typedef DYNAMICM(tracker::* GATED_METRIC_FUNC)(
//	std::vector<Track>& tracks,
//	const DETECTIONS& dets,
//	const std::vector<int>& track_indices,
//	const std::vector<int>& detection_indices);
//void _match(const DETECTIONS& detections, TRACHER_MATCHD& res);
//void _initiate_track(const DETECTION_ROW& detection);
//DYNAMICM gated_matric(
//	std::vector<Track>& tracks,
//	const DETECTIONS& dets,
//	const std::vector<int>& track_indices,
//	const std::vector<int>& detection_indices);
//DYNAMICM iou_cost(
//	std::vector<Track>& tracks,
//	const DETECTIONS& dets,
//	const std::vector<int>& track_indices,
//	const std::vector<int>& detection_indices);
//Eigen::VectorXf iou(DETECTBOX& bbox,
//	DETECTBOXSS &candidates);

#endif
