#pragma once

#include <common/config.hpp>
#include <pipeline/tasks.hpp>

#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

namespace veilsight {
    class IRecognizer {
    public:
        virtual ~IRecognizer() = default;
        virtual RecognitionResult recognize(const RecognitionTask& task) = 0;
    };

    class IRecognizerFactory {
    public:
        virtual ~IRecognizerFactory() = default;
        virtual std::unique_ptr<IRecognizer> create() const = 0;
        virtual int backend_threads() const = 0;
        virtual bool reload_gallery(std::string* error) {
            if (error) *error = "recognizer does not support gallery reload";
            return false;
        }
    };

    struct EnrollmentAnalysisCandidate {
        FaceObservation face;
        bool usable = false;
        std::vector<std::string> reject_reasons;
        std::vector<float> embedding;
    };

    struct EnrollmentAnalysisResult {
        bool ok = false;
        std::string message;
        int image_width = 0;
        int image_height = 0;
        std::vector<EnrollmentAnalysisCandidate> candidates;
    };

    EnrollmentAnalysisResult analyze_mobilefacenet_enrollment_image(
        const FaceDetectorModuleConfig& face_cfg,
        const RecognizerModuleConfig& recognizer_cfg,
        const cv::Mat& bgr);

    std::unique_ptr<IRecognizerFactory> create_recognizer_factory(const RecognizerModuleConfig& cfg);
    std::unique_ptr<IRecognizer> create_recognizer(const RecognizerModuleConfig& cfg);
}
