pub mod lib;

use actix_web::HttpResponse;
use actix_web::Responder;
use ndarray::Array;
use ndarray::IxDynImpl;
use ndarray::Dim;
use ndarray::Array4;
use std::io::Error;
use actix_web::{web, App, HttpServer, get};
use onnxruntime::OrtError;
use std::convert::From;
use lib::app::{state::AppState, onnx::init::*};


#[get("/forward")]
async fn run_forward_pass(data: web::Data<AppState>) -> impl Responder {
    let test_array: Array4<f32> = Array4::zeros((1, 3, 224, 224));
    let input_tensor_values = vec![test_array];

    let mut onnx_session = get_session(&data.env, "blobs/resnet18-v1-7.onnx".to_string()).unwrap();
    let outputs: Array<f32, Dim<IxDynImpl>> = onnx_session.run(input_tensor_values).unwrap()[0].to_owned();
    HttpResponse::Ok().body(format!("{:?}", outputs))
}

 
#[derive(Debug)]
enum GeneralError {
    OrtError(String),
    ActixError(String),
}

impl From<OrtError> for GeneralError {
    fn from(error: OrtError) -> Self {
        GeneralError::OrtError(error.to_string())
    }
}

impl From<Error> for GeneralError {
    fn from(error: Error) -> Self {
        GeneralError::ActixError(error.to_string())
    }
}

// This struct represents state

#[actix_web::main] // or #[tokio::main]
async fn main() -> std::result::Result<(), GeneralError> {

    HttpServer::new(|| { 
        let environment = get_environment("onnxruntime").unwrap();
        
        App::new()
            .app_data(web::Data::new(AppState::new(environment)))
            .service(run_forward_pass)
        }) 
        .bind(("127.0.0.1", 8081))?
        .run()
        .await?;

    Ok(())
}
