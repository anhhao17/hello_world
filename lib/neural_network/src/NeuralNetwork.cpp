#include "model.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "Arduino.h"
#include "NeuralNetwork.h"



  // Period (microseconds)

// TFLite globals, used for compatibility with Arduino-style sketches
namespace {
	tflite::ErrorReporter* error_reporter = nullptr;
	const tflite::Model* model = nullptr;
	tflite::MicroInterpreter* interpreter = nullptr;
	TfLiteTensor* model_input = nullptr;
	TfLiteTensor* model_output = nullptr;

	// Create an area of memory to use for input, output, and other TensorFlow
	// arrays. You'll need to adjust this by combiling, running, and looking
	// for errors.
	int inference_count = 0;
    constexpr int kTensorArenaSize = 2000;
	uint8_t tensor_arena[kTensorArenaSize];
} // namespace

void setup1() {

  // Wait for Serial to connect
#if DEBUG
  while(!Serial);
#endif

  // Let's make an LED vary in brightness
  

  // Set up logging (will report to Serial, even within TFLite functions)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

	static tflite::AllOpsResolver resolver;

	static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  	interpreter = &static_interpreter;	
	TfLiteStatus allocate_status = interpreter->AllocateTensors();
  	if (allocate_status != kTfLiteOk) {
    	TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    	return;
  	}
  // Assign model input and output buffers (tensors) to pointers
	model_input = interpreter->input(0);
	model_output = interpreter->output(0);
	inference_count = 0;
  // Get information about the memory area to use for the model's input
  // Supported data types:
  // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/common.h#L226
#if DEBUG
  Serial.print("Number of dimensions: ");
  Serial.println(model_input->dims->size);
  Serial.print("Dim 1 size: ");
  Serial.println(model_input->dims->data[0]);
  Serial.print("Dim 2 size: ");
  Serial.println(model_input->dims->data[1]);
  Serial.print("Input type: ");
  Serial.println(model_input->type);
#endif
}

void loop1() {

#if DEBUG
  unsigned long start_timestamp = micros();
#endif

	unsigned long timestamp=micros();
	timestamp = timestamp % (unsigned long)period;
	float x_val = ((float)timestamp * 2 * pi) / period;
	model_input->data.f[0] = x_val;
	TfLiteStatus invoke_status = interpreter->Invoke();
  	if (invoke_status != kTfLiteOk) {
    	error_reporter->Report("Invoke failed on input: %f\n", x_val);
  	}
	float y_val = model_output->data.f[0];
	Serial.println(y_val);

#if DEBUG
  Serial.print("Time for inference (us): ");
  Serial.println(micros() - start_timestamp);
#endif
}   