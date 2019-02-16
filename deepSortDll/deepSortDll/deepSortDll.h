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
	char MetaPath[256];        //ģ�ͽṹ�ļ�
	char CkptPath[256];        //ģ�Ͳ����ļ�
	tensorflow::string nameO;   //�������������
	tensorflow::string nameI;  //�������������
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
//	float max_iou_distance;   //IOU�Ƚ������Ե���ֵ
//	int max_age;              //�����������δ�ɹ�ƥ��֡��
//	int n_init;               //�����ɹ�ƥ��n_init֡������ȷ��Ϊ�ɹ�ƥ��
//	int _next_idx;            //Ŀ�����ID
//	int n_feature;            //�������֡���³ɹ�ƥ���box������������һ֡����ƥ��
//	float cosd;               //boxƥ��ʱ��cosin����
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
