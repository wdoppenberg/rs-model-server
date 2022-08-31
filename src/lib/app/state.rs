use onnxruntime::environment::Environment;

pub struct AppState {
    pub env: Environment,
    // pub onnx_session:  Arc<Mutex<&'a mut Session<'a>>>,
}

impl AppState {
    pub fn new(
        env: Environment,
        // onnx_session: &'a mut Session<'a>,
    ) -> Self {
        
        AppState {
            env,
            // onnx_session: Arc::new(Mutex::new(onnx_session)),
        }
    }
}
