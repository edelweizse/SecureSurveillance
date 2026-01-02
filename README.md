# SecureSurveillance: ARM-Targeted PPFR Framework

## Abstract

This project implements PPFR framework designed for deployment on resource-constrained embedded systems, specifically targeting Raspberry Pi. The system serves as the foundational infrastructure for a Privacy-Preserving Face Recognition (PPFR) research project, developed as part of a Bachelor of Science diploma thesis.

## Introduction

The framework provides a modular architecture for real-time video frame ingestion, models inference and streaming the results. The current implementation focuses on efficient frame ingestion and MJPEG streaming, with planned extensions for detection, tracking, recognition, and analytics modules.
## System Architecture

### Core Components

**Frame Ingestion Layer**
- Multi-source frame acquisition (webcam, file, RTSP)
- GStreamer-based pipeline implementation
- Configurable resolution, frame rate, and encoding parameters

**Encoding and Streaming Layer**
- MJPEG encoding with configurable quality parameters
- HTTP-based streaming server (port 8080)
- Thread-safe frame buffering and metadata management

**Web Interface**
- Streamlit-based monitoring dashboard
- Real-time video stream visualization
- Metadata display and system status monitoring

### Planned Extensions

The following modules are planned for future implementation:
- **Detection Module**: Object and face detection capabilities
- **Tracking Module**: Multi-object tracking across temporal sequences
- **Recognition Module**: Privacy-preserving face recognition algorithms
- **Analytics Module**: Behavioral analysis and statistical processing

## Target Platform Specifications

- **Primary Platform**: Raspberry Pi
- **Compatible Platforms**: ARM-based single-board computers
- **Architecture Support**: aarch64
- **Operating System**: Linux-based distributions (Raspberry Pi OS, Ubuntu ARM, etc.)

## Requirements

### Build Dependencies

- CMake 3.16 or higher
- C++17 compatible compiler (GCC/Clang)
- OpenCV
- GStreamer 1.0 and development libraries
- yaml-cpp library
- [httplib](https://github.com/yhirose/cpp-httplib)

### Runtime Dependencies

- Python 3.x
- Streamlit
- requests library

## Configuration

System configuration is managed through YAML files located in the `configs/` directory. The framework supports three frame source types:

### Webcam Source

```yaml
ingest:
  type: webcam
  src_id: cam0
  webcam:
    device: "/dev/video0"
    width: 1280
    height: 720
    fps: 30
    mjpg: true
```

**Performance Considerations**: For CPU-only ARM platforms, resolutions of 640x480 or 800x600 are recommended to maintain real-time performance.

### File Source

```yaml
ingest:
  type: file
  src_id: file0
  file:
    path: "assets/test_video.mp4"
    loop: false
```

### RTSP Source

```yaml
ingest:
  type: rtsp
  src_id: rtsp0
  rtsp:
    url: "rtsp://example.com/stream"
    latency_ms: 100
    tcp: true
```

## Usage

### Core Service Execution

```bash
./build/apps/core_service/core_service [config_path]
```

If no configuration path is specified, the system defaults to `configs/webcam.yaml`.

The service initializes the following endpoints:
- MJPEG video stream: `http://0.0.0.0:8080/video`
- Metadata endpoint: `http://0.0.0.0:8080/meta`

### Web Interface

```bash
streamlit run ui/app.py
```

The interface is accessible at `http://localhost:8501` 

## API Specification

### Endpoints

**GET /video**
- **Content-Type**: `multipart/x-mixed-replace; boundary=--boundary`
- **Description**: Continuous MJPEG video stream

**GET /meta**
- **Content-Type**: `application/json`
- **Description**: Frame metadata in JSON format

### Metadata Response Format

```json
{
  "source": "cam0",
  "frame_id": 12345
}
```

## Project Structure

```
.
├── apps/
│   └── core_service/          # Main application entry point
├── core/                      # Core library implementation
│   ├── include/               # Header files
│   └── src/                   # Implementation files
├── configs/                   # Configuration files
├── tests/                     # Unit and integration tests
└── ui/                        # Streamlit web interface
```

## Performance Optimization

For optimal performance on ARM platforms:

1. **Resolution**: Use 640x480 or 800x600 for real-time processing
2. **Frame Rate**: Limit to 15-20 FPS if frame drops occur
4. **Memory**: Monitor memory usage; consider frame buffer size reduction if needed

## Development Status

| Component | Status |
|-----------|--------|
| Frame Ingestion | Implemented |
| MJPEG Encoding | Implemented |
| HTTP Streaming | Implemented |
| Web Interface | Implemented |
| Detection Module | Planned |
| Tracking Module | Planned |
| Recognition Module (PPFR) | Planned |
| Analytics Module | Planned |
