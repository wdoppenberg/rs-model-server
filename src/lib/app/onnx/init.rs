use onnxruntime::environment::Environment;
pub use onnxruntime::session::Session;
use onnxruntime::{GraphOptimizationLevel, LoggingLevel, OrtError};

pub fn get_environment(name: &str) -> Result<Environment, OrtError> {
    Environment::builder()
        .with_name(name)
        .with_log_level(LoggingLevel::Verbose)
        .build()
}

pub fn get_session<'env, 'a> (environment: &'env Environment, model_path: String) -> Result<Session<'env>, OrtError> {
    let num_threads = std::thread::available_parallelism().unwrap().get() as i16;

    let session = environment.new_session_builder()?
        .with_optimization_level(GraphOptimizationLevel::All)?
        .with_number_threads(num_threads)?
        .with_model_from_file(model_path)?;
    Ok(session)
}
