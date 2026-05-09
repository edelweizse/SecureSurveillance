#pragma once

#include <common/config.hpp>
#include <pipeline/tasks.hpp>

#include <memory>

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
    };

    std::unique_ptr<IRecognizerFactory> create_recognizer_factory(const RecognizerModuleConfig& cfg);
    std::unique_ptr<IRecognizer> create_recognizer(const RecognizerModuleConfig& cfg);
}
