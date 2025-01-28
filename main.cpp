#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>
#include <vector>
#include <string>
#include <iostream>

class DeskObjectDetector {
private:
    cv::dnn::Net net;
    std::vector<std::string> classes;
    const cv::Size input_size;
    const float conf_threshold;
    const float nms_threshold;

public:
    struct Detection {
        cv::Rect box;
        float confidence;
        int class_id;
    };

    DeskObjectDetector(const std::string& model_path, 
                      const std::vector<std::string>& class_names,
                      float confidence_threshold = 0.5,
                      float nms_thresh = 0.4) 
        : classes(class_names), 
          input_size(cv::Size(640, 640)),
          conf_threshold(confidence_threshold),
          nms_threshold(nms_thresh) {
        
        try {
            // ONNX 모델 로드
            net = cv::dnn::readNetFromONNX(model_path);
            
            // CPU 사용 설정 (OpenCL 비활성화)
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            
            std::cout << "Model loaded successfully: " << model_path << std::endl;
            
            // 초기 더미 추론으로 모델 웜업
            cv::Mat dummy(input_size, CV_8UC3, cv::Scalar(0, 0, 0));
            cv::Mat blob;
            cv::dnn::blobFromImage(dummy, blob, 1./255., input_size, 
                                 cv::Scalar(0,0,0), true, false, CV_32F);
            
            net.setInput(blob);
            std::vector<cv::String> outNames = net.getUnconnectedOutLayersNames();
            std::vector<cv::Mat> outputs;
            net.forward(outputs, outNames);
            
            std::cout << "Model warmup completed." << std::endl;
        } catch (const cv::Exception& e) {
            std::cerr << "Error initializing model: " << e.what() << std::endl;
            throw;
        }
    }

    std::vector<Detection> detect(cv::Mat& frame) {
        try {
            cv::Mat blob;
            cv::dnn::blobFromImage(frame, blob, 1./255., input_size, 
                                cv::Scalar(0,0,0), true, false, CV_32F);
            
            net.setInput(blob);
            std::vector<cv::Mat> outputs;
            net.forward(outputs, net.getUnconnectedOutLayersNames());

            cv::Mat output = outputs[0];
            std::cout << "Output tensor dimensions: " << output.dims << std::endl;
            for (int i = 0; i < output.dims; i++) {
                std::cout << "Dim " << i << ": " << output.size[i] << std::endl;
            }

            // 결과를 저장할 벡터들을 미리 예약
            std::vector<Detection> detections;
            std::vector<float> confidences;
            std::vector<cv::Rect> boxes;
            detections.reserve(100);  // 적절한 크기로 예약
            confidences.reserve(100);
            boxes.reserve(100);

            // 출력 데이터에 직접 접근
            float* data = (float*)output.data;
            int dimensions = 14;  // 4(bbox) + 10(클래스 수)
            int rows = output.size[2];  // 8400

            // 각 detection 처리
            for (int i = 0; i < rows; ++i) {
                float* row = data + i * dimensions;
                
                // 클래스 신뢰도 계산
                float max_conf = 0.0f;
                int max_class_id = -1;
                
                // 클래스별 신뢰도 확인
                for (int j = 4; j < dimensions; ++j) {
                    if (row[j] > max_conf) {
                        max_conf = row[j];
                        max_class_id = j - 4;
                    }
                }

                if (max_conf > conf_threshold) {
                    float x = row[0];
                    float y = row[1];
                    float w = row[2];
                    float h = row[3];

                    // 픽셀 좌표로 변환
                    int left = static_cast<int>((x - 0.5f * w) * frame.cols);
                    int top = static_cast<int>((y - 0.5f * h) * frame.rows);
                    int width = static_cast<int>(w * frame.cols);
                    int height = static_cast<int>(h * frame.rows);

                    // 경계값 검사
                    left = std::max(0, std::min(left, frame.cols - 1));
                    top = std::max(0, std::min(top, frame.rows - 1));
                    width = std::min(width, frame.cols - left);
                    height = std::min(height, frame.rows - top);

                    Detection det;
                    det.box = cv::Rect(left, top, width, height);
                    det.confidence = max_conf;
                    det.class_id = max_class_id;

                    detections.push_back(det);
                    confidences.push_back(max_conf);
                    boxes.push_back(det.box);
                }
            }

            // NMS 적용
            std::vector<int> indices;
            cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indices);

            // 최종 결과 필터링
            std::vector<Detection> final_detections;
            final_detections.reserve(indices.size());
            for (int idx : indices) {
                final_detections.push_back(detections[idx]);
            }

            return final_detections;
        }
        catch (const std::bad_alloc& e) {
            std::cerr << "Memory allocation error: " << e.what() << std::endl;
            return std::vector<Detection>();
        }
        catch (const cv::Exception& e) {
            std::cerr << "OpenCV error in detect(): " << e.what() << std::endl;
            return std::vector<Detection>();
        }
        catch (const std::exception& e) {
            std::cerr << "Error in detect(): " << e.what() << std::endl;
            return std::vector<Detection>();
        }
    }

    void drawDetections(cv::Mat& frame, const std::vector<Detection>& detections) {
        for (const auto& det : detections) {
            cv::rectangle(frame, det.box, cv::Scalar(0, 255, 0), 2);

            std::string label = classes[det.class_id] + " " + 
                              std::to_string(det.confidence).substr(0, 4);

            int baseLine;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 
                                               0.5, 1, &baseLine);
            int top = std::max(det.box.y, labelSize.height);
            cv::rectangle(frame, 
                         cv::Point(det.box.x, top - labelSize.height),
                         cv::Point(det.box.x + labelSize.width, top + baseLine),
                         cv::Scalar(255, 255, 255), cv::FILLED);
            cv::putText(frame, label, cv::Point(det.box.x, top), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        }
    }
};

int main() {
    std::vector<std::string> class_names = {
        "laptop", "book", "cell phone", "cup", "bottle", 
        "keyboard", "mouse", "remote", "scissors", "clock"
    };

    // Raspberry Pi Camera 초기화
    cv::VideoCapture cap;
    cap.open(0, cv::CAP_V4L2);  // libcamera-apps와 호환되는 V4L2 백엔드 사용

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open Raspberry Pi Camera." << std::endl;
        return -1;
    }

    // 카메라 설정
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FPS, 30);
    
    // 카메라 설정 확인
    std::cout << "Camera initialized with:" << std::endl;
    std::cout << "Width: " << cap.get(cv::CAP_PROP_FRAME_WIDTH) << std::endl;
    std::cout << "Height: " << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;
    std::cout << "FPS: " << cap.get(cv::CAP_PROP_FPS) << std::endl;

    const std::string model_path = "/home/taewonkim/Desktop/desk_detection/best.onnx";

    try {
        DeskObjectDetector detector(model_path, class_names);
        
        cv::Mat frame;
        while (true) {
            cap >> frame;
            if (frame.empty()) {
                std::cerr << "Error: Blank frame grabbed" << std::endl;
                break;
            }

            auto detections = detector.detect(frame);
            detector.drawDetections(frame, detections);
            
            cv::imshow("Object Detection", frame);
            
            if (cv::waitKey(1) == 'q') {
                break;
            }
        }
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error: " << e.what() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
