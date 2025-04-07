mod backend;
mod frontend;
mod utils;

pub fn main() {
    let backend_thread = backend::run();
    let frontend_thread = frontend::run();

    backend_thread.join().unwrap();
    frontend_thread.join().unwrap();
}
