#include <chrono>
#include <cublas_v2.h>
#include <unordered_map>
#include "NvInferPlugin.h"
#include "NvUffParser.h"
#include "ros/ros.h"
#include "cv_bridge/cv_bridge.h"
#include "object_msgs/ObjectsInBoxes.h"
#include "ssd_ros/common.h"
#include "ssd_ros/utils.h"

using namespace std;
using namespace nvinfer1;
using namespace nvuffparser;

static constexpr int INPUT_C = 3;
static constexpr int INPUT_H = 300;
static constexpr int INPUT_W = 300;

const char* INPUT_BLOB_NAME = "Input";
const char* OUTPUT_BLOB_NAME0 = "NMS";
int OUTPUT_CLS_SIZE;

DetectionOutputParameters detectionOutputParam{ true, false, 0,   OUTPUT_CLS_SIZE,        100,
                                                100,  0.5,   0.6, CodeTypeSSD::TF_CENTER, { 0, 2, 1 },
                                                true, true };

// Visualization
float visualizeThreshold;

class Logger : public ILogger
{
  void log(Severity severity, const char* msg) override
  {
    if (severity != Severity::kINFO)
      ROS_INFO("[[trt_infer.cu]] %s", msg);
  }
} gLogger;

namespace
{
const char* FLATTENCONCAT_PLUGIN_VERSION{"1"};
const char* FLATTENCONCAT_PLUGIN_NAME{"FlattenConcat_TRT"};
} // namespace

class FlattenConcat : public IPluginV2
{
public:
    // Ordinary ctor, plugin not yet configured for particular inputs/output
    FlattenConcat() {}

    // Ctor for clone()
    FlattenConcat(const int* flattenedInputSize, int numInputs, int flattenedOutputSize)
        : mFlattenedOutputSize(flattenedOutputSize)
    {
        for (int i = 0; i < numInputs; ++i)
            mFlattenedInputSize.push_back(flattenedInputSize[i]);
    }

    // Ctor for loading from serialized byte array
    FlattenConcat(const void* data, size_t length)
    {
        const char* d = reinterpret_cast<const char*>(data);
        const char* a = d;

        size_t numInputs = read<size_t>(d);
        for (size_t i = 0; i < numInputs; ++i)
        {
            mFlattenedInputSize.push_back(read<int>(d));
        }
        mFlattenedOutputSize = read<int>(d);

        assert(d == a + length);
    }

    int getNbOutputs() const override
    {
        // We always return one output
        return 1;
    }

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
        // At least one input
        assert(nbInputDims >= 1);
        // We only have one output, so it doesn't
        // make sense to check index != 0
        assert(index == 0);

        size_t flattenedOutputSize = 0;
        int inputVolume = 0;

        for (int i = 0; i < nbInputDims; ++i)
        {
            // We only support NCHW. And inputs Dims are without batch num.
            assert(inputs[i].nbDims == 3);

            inputVolume = inputs[i].d[0] * inputs[i].d[1] * inputs[i].d[2];
            flattenedOutputSize += inputVolume;
        }

        return DimsCHW(flattenedOutputSize, 1, 1);
    }

    int initialize() override
    {
        // Called on engine initialization, we initialize cuBLAS library here,
        // since we'll be using it for inference
        CHECK(cublasCreate(&mCublas));
        return 0;
    }

    void terminate() override
    {
        // Called on engine destruction, we destroy cuBLAS data structures,
        // which were created in initialize()
        CHECK(cublasDestroy(mCublas));
    }

    size_t getWorkspaceSize(int maxBatchSize) const override
    {
        // The operation is done in place, it doesn't use GPU memory
        return 0;
    }

    int enqueue(int batchSize, const void* const* inputs, void** outputs, void*, cudaStream_t stream) override
    {
        // Does the actual concat of inputs, which is just
        // copying all inputs bytes to output byte array
        size_t inputOffset = 0;
        float* output = reinterpret_cast<float*>(outputs[0]);
        cublasSetStream(mCublas, stream);

        for (size_t i = 0; i < mFlattenedInputSize.size(); ++i)
        {
            const float* input = reinterpret_cast<const float*>(inputs[i]);
            for (int batchIdx = 0; batchIdx < batchSize; ++batchIdx)
            {
                CHECK(cublasScopy(mCublas, mFlattenedInputSize[i],
                                  input + batchIdx * mFlattenedInputSize[i], 1,
                                  output + (batchIdx * mFlattenedOutputSize + inputOffset), 1));
            }
            inputOffset += mFlattenedInputSize[i];
        }

        return 0;
    }

    size_t getSerializationSize() const override
    {
        // Returns FlattenConcat plugin serialization size
        size_t size = sizeof(mFlattenedInputSize[0]) * mFlattenedInputSize.size()
            + sizeof(mFlattenedOutputSize)
            + sizeof(size_t); // For serializing mFlattenedInputSize vector size
        return size;
    }

    void serialize(void* buffer) const override
    {
        // Serializes FlattenConcat plugin into byte array

        // Cast buffer to char* and save its beginning to a,
        // (since value of d will be changed during write)
        char* d = reinterpret_cast<char*>(buffer);
        char* a = d;

        size_t numInputs = mFlattenedInputSize.size();

        // Write FlattenConcat fields into buffer
        write(d, numInputs);
        for (size_t i = 0; i < numInputs; ++i)
        {
            write(d, mFlattenedInputSize[i]);
        }
        write(d, mFlattenedOutputSize);

        // Sanity check - checks if d is offset
        // from a by exactly the size of serialized plugin
        assert(d == a + getSerializationSize());
    }

    void configureWithFormat(const Dims* inputs, int nbInputs, const Dims* outputDims, int nbOutputs, nvinfer1::DataType type, nvinfer1::PluginFormat format, int maxBatchSize) override
    {
        // We only support one output
        assert(nbOutputs == 1);

        // Reset plugin private data structures
        mFlattenedInputSize.clear();
        mFlattenedOutputSize = 0;

        // For each input we save its size, we also validate it
        for (int i = 0; i < nbInputs; ++i)
        {
            int inputVolume = 0;

            // We only support NCHW. And inputs Dims are without batch num.
            assert(inputs[i].nbDims == 3);

            // All inputs dimensions along non concat axis should be same
            for (size_t dim = 1; dim < 3; dim++)
            {
                assert(inputs[i].d[dim] == inputs[0].d[dim]);
            }

            // Size of flattened input
            inputVolume = inputs[i].d[0] * inputs[i].d[1] * inputs[i].d[2];
            mFlattenedInputSize.push_back(inputVolume);
            mFlattenedOutputSize += mFlattenedInputSize[i];
        }
    }

    bool supportsFormat(DataType type, PluginFormat format) const override
    {
        return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
    }

    const char* getPluginType() const override { return FLATTENCONCAT_PLUGIN_NAME; }

    const char* getPluginVersion() const override { return FLATTENCONCAT_PLUGIN_VERSION; }

    void destroy() override {}

    IPluginV2* clone() const override
    {
        return new FlattenConcat(mFlattenedInputSize.data(), mFlattenedInputSize.size(), mFlattenedOutputSize);
    }

    void setPluginNamespace(const char* pluginNamespace) override
    {
        mPluginNamespace = pluginNamespace;
    }

    const char* getPluginNamespace() const override
    {
        return mPluginNamespace.c_str();
    }

private:
    template <typename T>
    void write(char*& buffer, const T& val) const
    {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }

    template <typename T>
    T read(const char*& buffer)
    {
        T val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
        return val;
    }

    // Number of elements in each plugin input, flattened
    std::vector<int> mFlattenedInputSize;
    // Number of elements in output, flattened
    int mFlattenedOutputSize{0};
    // cuBLAS library handle
    cublasHandle_t mCublas;
    // We're not using TensorRT namespaces in
    // this sample, so it's just an empty string
    std::string mPluginNamespace = "";
};


class FlattenConcatPluginCreator : public IPluginCreator
{
public:
    FlattenConcatPluginCreator()
    {
        mFC.nbFields = 0;
        mFC.fields = 0;
    }

    ~FlattenConcatPluginCreator() {}

    const char* getPluginName() const override { return FLATTENCONCAT_PLUGIN_NAME; }

    const char* getPluginVersion() const override { return FLATTENCONCAT_PLUGIN_VERSION; }

    const PluginFieldCollection* getFieldNames() override { return &mFC; }

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) override
    {
        return new FlattenConcat();
    }

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override
    {

        return new FlattenConcat(serialData, serialLength);
    }

    void setPluginNamespace(const char* pluginNamespace) override
    {
        mPluginNamespace = pluginNamespace;
    }

    const char* getPluginNamespace() const override
    {
        return mPluginNamespace.c_str();
    }

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mPluginNamespace = "";
};

PluginFieldCollection FlattenConcatPluginCreator::mFC{};
std::vector<PluginField> FlattenConcatPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(FlattenConcatPluginCreator);

IRuntime* runtime;
ICudaEngine* engine;
IExecutionContext* context;
cudaStream_t stream;

int nbBindings, inputIndex, outputIndex0, outputIndex1;
vector<void*> buffers;
Dims inputDims;

bool is_initialized = false;

vector<string> CLASSES;

void setup(std::string labelFilename, std::string planFilename, int numClasses, float th)
{
  OUTPUT_CLS_SIZE = numClasses;
  visualizeThreshold = th;

  ifstream labelFile(labelFilename.c_str());
  ifstream planFile(planFilename.c_str());

  if (!labelFile.is_open())
  {
    ROS_INFO("Label Not Found!!!");
    is_initialized = false;
  }
  else if (!planFile.is_open())
  {
    ROS_INFO("Plan Not Found!!!");
    is_initialized = false;
  }
  else
  {
    string line;
    while (getline(labelFile, line))
    {
      CLASSES.push_back(line);
    }

    initLibNvInferPlugins(&gLogger, "");

    ROS_INFO("Begin loading plan...");
    stringstream planBuffer;
    planBuffer << planFile.rdbuf();
    string plan = planBuffer.str();

    ROS_INFO("*** deserializing");
    runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine((void*)plan.data(), plan.size(), nullptr);
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);
    CHECK(cudaStreamCreate(&stream));
    ROS_INFO("End loading plan...");

    // Input and output buffer pointers that we pass to the engine - the engine requires exactly
    // IEngine::getNbBindings(),
    // of these, but in this case we know that there is exactly 1 input and 2 output.
    nbBindings = engine->getNbBindings();
    vector<pair<int64_t, DataType>> buffersSizes;
    for (int i = 0; i < nbBindings; ++i)
    {
      Dims dims = engine->getBindingDimensions(i);
      DataType dtype = engine->getBindingDataType(i);

      int64_t eltCount = samplesCommon::volume(dims);
      buffersSizes.push_back(make_pair(eltCount, dtype));
    }

    for (int i = 0; i < nbBindings; ++i)
    {
      auto bufferSizesOutput = buffersSizes[i];
      buffers.push_back(samplesCommon::safeCudaMalloc(bufferSizesOutput.first *
                                                       samplesCommon::getElementSize(bufferSizesOutput.second)));
    }

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings().
    inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    outputIndex0 = engine->getBindingIndex(OUTPUT_BLOB_NAME0);
    outputIndex1 = outputIndex0 + 1;  // engine.getBindingIndex(OUTPUT_BLOB_NAME1);

    inputDims = engine->getBindingDimensions(inputIndex);
    is_initialized = true;
  }
}

void destroy(void)
{
  if (is_initialized)
  {
    runtime->destroy();
    engine->destroy();
    context->destroy();
    // Release the stream and the buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex0]));
    CHECK(cudaFree(buffers[outputIndex1]));
  }
  is_initialized = false;
}

object_msgs::ObjectsInBoxes infer(const sensor_msgs::ImageConstPtr& color_msg)
{
  object_msgs::ObjectsInBoxes bboxes;

  // preprocessing
  cv::Mat image = cv_bridge::toCvShare(color_msg, "rgb8")->image;
  cv::Size imsize = image.size();
  cv::resize(image, image, cv::Size(INPUT_W, INPUT_H));
  vector<float> inputData(INPUT_C * INPUT_H * INPUT_W);
  cvImageToTensor(image, &inputData[0], inputDims);
  preprocessInception(&inputData[0], inputDims);

  // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
  CHECK(cudaMemcpyAsync(buffers[inputIndex], &inputData[0], INPUT_C * INPUT_H * INPUT_W * sizeof(float),
                        cudaMemcpyHostToDevice, stream));

  auto t_start = chrono::high_resolution_clock::now();
  context->execute(1, &buffers[0]);
  auto t_end = chrono::high_resolution_clock::now();
  float total = chrono::duration<float, milli>(t_end - t_start).count();
  bboxes.inference_time_ms = total;
  
  // Host memory for outputs.
  vector<float> detectionOut(detectionOutputParam.keepTopK * 7);
  vector<int> keepCount(1);

  CHECK(cudaMemcpyAsync(&detectionOut[0], buffers[outputIndex0], detectionOutputParam.keepTopK * 7 * sizeof(float),
                        cudaMemcpyDeviceToHost, stream));
  CHECK(cudaMemcpyAsync(&keepCount[0], buffers[outputIndex1], sizeof(int), cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);

  for (int i = 0; i < keepCount[0]; ++i)
  {
    float* det = &detectionOut[0] + i * 7;
    if (det[2] < visualizeThreshold)
      continue;

    // Output format for each detection is stored in the below order
    // [image_id, label, confidence, xmin, ymin, xmax, ymax]
    assert((int)det[1] < OUTPUT_CLS_SIZE);
    object_msgs::ObjectInBox bbox;
    bbox.object.object_name = CLASSES[(int)det[1]].c_str();
    bbox.object.probability = det[2];
    bbox.roi.x_offset = det[3] * imsize.width;
    bbox.roi.y_offset = det[4] * imsize.height;
    bbox.roi.width = (det[5] - det[3]) * imsize.width;
    bbox.roi.height = (det[6] - det[4]) * imsize.height;
    bbox.roi.do_rectify = false;
    bboxes.objects_vector.push_back(bbox);
  }

  auto f_end = chrono::high_resolution_clock::now();
  float f_total = chrono::duration<float, milli>(f_end - t_start).count();
  bboxes.all_time_ms = f_total;

  bboxes.header = color_msg->header;

  return bboxes;
}
