use image::DynamicImage;
use ort::{
    CPUExecutionProvider, CUDAExecutionProvider, CoreMLExecutionProvider,
    ExecutionProviderDispatch, GraphOptimizationLevel, Session,
};

#[derive(Debug, Clone)]
pub enum OptLevel {
    Disable,
    Level1,
    Level2,
    Level3,
}

#[derive(Debug, Clone)]
pub struct SessionOptions {
    opt_level: Option<OptLevel>,
    num_threads: usize,
    parallel_execution: bool,
    memory_pattern: bool,
    providers: Option<Vec<String>>,
}

impl Default for SessionOptions {
    fn default() -> Self {
        Self {
            opt_level: Some(OptLevel::Level3),
            num_threads: 32,
            parallel_execution: true,
            memory_pattern: true,
            providers: Some(vec!["coreml".to_owned()]),
        }
    }
}

impl SessionOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_opt_level(&mut self, opt_level: OptLevel) -> &mut Self {
        self.opt_level = Some(opt_level);
        self
    }

    pub fn with_num_threads(&mut self, num_threads: usize) -> &mut Self {
        self.num_threads = num_threads;
        self
    }

    pub fn with_parallel_execution(&mut self, parallel_execution: bool) -> &mut Self {
        self.parallel_execution = parallel_execution;
        self
    }

    pub fn with_memory_pattern(&mut self, memory_pattern: bool) -> &mut Self {
        self.memory_pattern = memory_pattern;
        self
    }

    pub fn with_providers(&mut self, providers: Vec<String>) -> &mut Self {
        self.providers = Some(providers);
        self
    }

    pub fn build(&self) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(self.clone())
    }
}

pub struct BaseSession {
    pub(crate) debug: bool,
    pub(crate) inner_session: Option<ort::Session>,
    pub session_options: SessionOptions,
    pub model_path: String,
}

impl BaseSession {
    pub fn new(
        debug: bool,
        session_options: SessionOptions,
        model_name: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let session_opts = session_options.clone();
        log::debug!("Session created with options: {:?}", &session_opts);

        let project_dir = std::env!("CARGO_MANIFEST_DIR");
        let model_path = format!("{}/models/onnx/{}.onnx", project_dir, model_name);

        log::debug!("Model path: {}", &model_path);
        let mut session_builder = Session::builder()?;

        if let Some(opt_level) = session_options.opt_level {
            session_builder = session_builder.with_optimization_level(match opt_level {
                OptLevel::Disable => GraphOptimizationLevel::Disable,
                OptLevel::Level1 => GraphOptimizationLevel::Level1,
                OptLevel::Level2 => GraphOptimizationLevel::Level2,
                OptLevel::Level3 => GraphOptimizationLevel::Level3,
            })?;
        }

        session_builder =
            session_builder.with_intra_threads(if session_options.num_threads <= 0 {
                1
            } else {
                session_options.num_threads
            })?;

        let prividers = session_options
            .providers
            .unwrap_or(vec!["coreml".to_owned()])
            .iter()
            .map(|provider| match provider.as_str() {
                "coreml" => CoreMLExecutionProvider::default().build(),
                "cuda" => CUDAExecutionProvider::default().build(),
                _ => CPUExecutionProvider::default().build(),
            })
            .collect::<Vec<ExecutionProviderDispatch>>();
        session_builder = session_builder.with_execution_providers(prividers)?;

        log::debug!("Session builder: starting to build session");
        let session = session_builder.commit_from_file(model_path.clone())?;

        log::debug!("Session: {:?}", &session);

        Ok(Self {
            debug,
            session_options: session_opts,
            inner_session: Some(session),
            model_path: model_path.to_owned(),
        })
    }
}

pub trait BaseSessionTrait {
    fn get_session(&self) -> Option<&ort::Session>;

    fn run(
        &self,
        original_image: DynamicImage,
    ) -> Result<ndarray::Array3<u8>, Box<dyn std::error::Error>>;

    fn post_process(
        &self,
        output: ndarray::Array3<u8>,
        original_image: DynamicImage,
    ) -> Result<ndarray::Array3<u8>, Box<dyn std::error::Error>>;

    fn get_model_name(&self) -> String;
}

#[derive(thiserror::Error, Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum SessionError {
    #[error("Session not initialized")]
    PredictError,
    #[error("Model no output")]
    NoOutput,
    #[error("Image processing error")]
    ImageProcessingError,
    #[error("Model loading error")]
    ModelLoadError,
    #[error("Model not implemented")]
    NotImplemented,
}
