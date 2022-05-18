use actix_web::{get, web, App, HttpServer, Responder};
use rand;

#[get("/hello/{name}")]
async fn greet(name: web::Path<String>) -> impl Responder {
    format!("Hello {name}!")
}

#[get("/rand")]
async fn random() -> impl Responder {
    format!("{}", rand::random::<u32>())
}


#[actix_web::main] // or #[tokio::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| {
        App::new()
            .route("/hello", web::get().to(|| async { "Hello World!" }))
            .service(greet)
            .service(random)
    })
        .bind(("127.0.0.1", 8081))?
        .run()
        .await
}