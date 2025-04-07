use std::thread::{self, JoinHandle};
use std::{path::PathBuf, process::Command};

pub fn run() -> JoinHandle<()> {
    thread::spawn(|| {
        dotenv::dotenv().ok();
        let dir = env!("CARGO_MANIFEST_DIR");

        Command::new("cargo")
            .envs(std::env::vars())
            .args(["watch", "-x", "run", "-w", "backend"])
            .current_dir(PathBuf::from(dir))
            .spawn()
            .unwrap()
            .wait_with_output()
            .unwrap();
    })
}
