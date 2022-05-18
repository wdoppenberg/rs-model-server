use onnxruntime::environment::{Environment};
use onnxruntime::error::{OrtError};
use onnxruntime::{LoggingLevel, GraphOptimizationLevel};
use onnxruntime::tensor::{OrtOwnedTensor};
use ndarray::Array4;


fn main() -> Result<(), OrtError> {
    let environment = Environment::builder()
        .with_name("test")
        .with_log_level(LoggingLevel::Verbose)
        .build()?;

    let mut session = environment
        .new_session_builder()?
        .with_optimization_level(GraphOptimizationLevel::Extended)?
        .with_number_threads(1)?
        .with_model_from_file("resnet50-v1-12.onnx")?;

    let test_array = Array4::<f32>::zeros((1, 3, 224, 224));
    let input_tensor_values = vec![test_array];

    let outputs: Vec<OrtOwnedTensor<f32, _>> = session.run(input_tensor_values)?;
    println!("{:?}", outputs[0].shape());

    let num_classes = outputs[0].shape()[1];

    for i in 0..num_classes {
        println!("Score for class [{}] =  {}", i, outputs[0][[0, i]]);
    }

    Ok(())
}
