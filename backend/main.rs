use dotenv::var;
use rocket::http::Method;
use rocket_cors::{AllowedOrigins, CorsOptions};

mod controllers;
mod sessions;
mod utils;

#[macro_use]
extern crate rocket;

#[launch]
fn rocket() -> _ {
    dotenv::dotenv().ok();
    simple_logger::init_with_env().unwrap();

    let cors = CorsOptions::default()
        .allowed_origins(AllowedOrigins::some_exact(&[
            &var("CORS_ALLOW_ORIGIN").unwrap_or("http://localhost:5173".to_owned()), // Allow requests from the frontend
        ]))
        .allowed_methods(
            vec![Method::Get, Method::Post, Method::Patch, Method::Options]
                .into_iter()
                .map(From::from)
                .collect(),
        )
        .allow_credentials(true);

    let app = rocket::build().mount("/", controllers::rembg::routes());
    app.attach(cors.to_cors().unwrap())
}
