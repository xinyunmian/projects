#include "deepSortDll.h"
#pragma comment(lib,"tensorflow.lib")
#pragma comment(lib,"libprotobuf.lib")
//#include "trackMain/tracker.h"
using namespace tensorflow;
using namespace std;

//#define args_nn_budget 100
//#define args_max_cosine_distance 0.2


bool Modelinit(DeepSortParams* pSortParams)
{
	tensorflow::SessionOptions sessOptions;
	sessOptions.config.mutable_gpu_options()->set_allow_growth(true);
	session = NewSession(sessOptions);
	if (session == nullptr) return false;

	const tensorflow::string pathToGraph = pSortParams->MetaPath;
	Status status;
	MetaGraphDef graph_def;
	status = ReadBinaryProto(tensorflow::Env::Default(), pathToGraph, &graph_def);
	if (status.ok() == false) return false;

	status = session->Create(graph_def.graph_def());
	if (status.ok() == false) return false;

	const tensorflow::string checkpointPath = pSortParams->CkptPath;
	Tensor checkpointTensor(DT_STRING, TensorShape());
	checkpointTensor.scalar<std::string>()() = checkpointPath;
	status = session->Run(
	{ { graph_def.saver_def().filename_tensor_name(), checkpointTensor }, },
	{}, { graph_def.saver_def().restore_op_name() }, nullptr);
	if (status.ok() == false) return false;

	input_layer = pSortParams->nameI;
	outnames.push_back(pSortParams->nameO);
	feature_dim = 128;
	return true;
}

bool getRectsFeature(const cv::Mat& img, DETECTIONS& d) 
{
	std::vector<cv::Mat> mats;
	for (DETECTION_ROW& dbox : d) {
		cv::Rect rc = cv::Rect(int(dbox.tlwh(0)), int(dbox.tlwh(1)),
			int(dbox.tlwh(2)), int(dbox.tlwh(3)));
		rc.x -= (rc.height * 0.5 - rc.width) * 0.5;
		rc.width = rc.height * 0.5;
		rc.x = (rc.x >= 0 ? rc.x : 0);
		rc.y = (rc.y >= 0 ? rc.y : 0);
		rc.width = (rc.x + rc.width <= img.cols ? rc.width : (img.cols - rc.x));
		rc.height = (rc.y + rc.height <= img.rows ? rc.height : (img.rows - rc.y));

		cv::Mat mattmp = img(rc).clone();
		cv::resize(mattmp, mattmp, cv::Size(64, 128));
		mats.push_back(mattmp);
	}
	int count = mats.size();

	Tensor input_tensor(DT_UINT8, TensorShape({ count, 128, 64, 3 }));
	tobuffer(mats, input_tensor.flat<uintb8>().data());
	std::vector<std::pair<tensorflow::string, Tensor>> feed_dict = {
		{ input_layer, input_tensor },
	};

	Status status = session->Run(feed_dict, outnames, {}, &output_tensors);
	if (!status.ok()) {
		std::cout << "ERROR: RUN failed..." << std::endl;
		std::cout << status.ToString() << "\n";
	}
	float* tensor_buffer = output_tensors[0].flat<float>().data();
	int i = 0;
	for (DETECTION_ROW& dbox : d) {
		for (int j = 0; j < feature_dim; j++)
			dbox.feature[j] = tensor_buffer[i*feature_dim + j];
		i++;
	}
	return true;
}

void tobuffer(const std::vector<cv::Mat> &imgs, uintb8 *buf) 
{
	int pos = 0;
	for (const cv::Mat& img : imgs) {
		int Lenth = img.rows * img.cols * 3;
		int nr = img.rows;
		int nc = img.cols;
		if (img.isContinuous()) {
			nr = 1;
			nc = Lenth;
		}
		for (int i = 0; i < nr; i++) {
			const uchar* inData = img.ptr<uchar>(i);
			for (int j = 0; j < nc; j++) {
				buf[pos] = *inData++;
				pos++;
			}
		}//end for
	}//end imgs;
}


//enum DETECTBOX_IDX { IDX_X = 0, IDX_Y, IDX_W, IDX_H };
//DETECTBOX DETECTION_ROW::to_xyah() const
//{//(centerx, centery, ration, h)
//	DETECTBOX ret = tlwh;
//	ret(0, IDX_X) += (ret(0, IDX_W)*0.5);
//	ret(0, IDX_Y) += (ret(0, IDX_H)*0.5);
//	ret(0, IDX_W) /= ret(0, IDX_H);
//	return ret;
//}
//
//DETECTBOX DETECTION_ROW::to_tlbr() const
//{//(x,y,xx,yy)
//	DETECTBOX ret = tlwh;
//	ret(0, IDX_X) += ret(0, IDX_W);
//	ret(0, IDX_Y) += ret(0, IDX_H);
//	return ret;
//}

//tracker mytracker(args_max_cosine_distance, args_nn_budget);
//DETECTIONS detections;
//mytracker.predict();
//mytracker.update(detections);