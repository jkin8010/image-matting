use std::fs;
use std::thread;
use std::thread::JoinHandle;
use std::{path::PathBuf, process::Command};

#[cfg(windows)]
pub const NPM: &'static str = "npm.cmd";

#[cfg(not(windows))]
pub const NPM: &'static str = "npm";

pub const VITEJS_PORT: u16 = 5173;

pub fn run() -> JoinHandle<()> {
    if !crate::utils::net::is_port_free(VITEJS_PORT) {
        println!("========================================================");
        println!(" ViteJS (the frontend compiler/bundler) needs to run on");
        println!(" port 5580 but it seems to be in use.");
        println!("========================================================");
        panic!("Port 5580 is taken but is required for development!")
    }

    thread::spawn(|| {
        let dir = env!("CARGO_MANIFEST_DIR");

        if fs::metadata(PathBuf::from_iter([dir, "frontend", "node_modules"])).is_err() {
            Command::new(NPM)
                .arg("install")
                .spawn()
                .unwrap()
                .wait()
                .unwrap();
        }

        Command::new(NPM)
            .args([
                "run",
                "dev",
                "--",
                format!("--port={}", VITEJS_PORT).as_str(),
            ])
            .current_dir(PathBuf::from_iter([dir, "frontend"]))
            .spawn()
            .unwrap()
            .wait_with_output()
            .unwrap();
    })
}
