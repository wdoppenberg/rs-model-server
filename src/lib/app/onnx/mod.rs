pub mod init;

pub use init::{get_environment, get_session};


#[cfg(test)]
mod integration {
    use std::error::Error;
    use ndarray::Array4;
    use onnxruntime::tensor::OrtOwnedTensor;
    use super::init::{get_environment, get_session};

    #[test]
    fn run_forward_pass() -> Result<(), Box<dyn Error>> {
        let model_path= "blobs/resnet18-v1-7.onnx";

        // If model_path is not a valid path, print error message and return.
        if !std::path::Path::new(model_path).exists() {
            println!("Model file not found, please check the path: {}", model_path);
            print!(
                "Please download the model file from: \
            https://github.com/onnx/models/blob/main/vision/classification/resnet/model/resnet18-v1-7.onnx?raw=true"
            );
        }

        let environment = get_environment("ONNX Runtime Rust Example")?;

        let mut session = get_session(&environment, model_path)?;

        let test_array: Array4<f32> = Array4::zeros((1, 3, 224, 224));
        let input_tensor_values = vec![test_array];

        let outputs: Vec<OrtOwnedTensor<f32, _>> = session.run(input_tensor_values)?;
        assert_eq!(outputs[0].shape(), &[1, 1000]);

        let num_classes = outputs[0].shape()[1];

        for i in 0..num_classes {
            assert!(outputs[0][[0, i]].abs() < 100.);
        }
        Ok(())
    }
}