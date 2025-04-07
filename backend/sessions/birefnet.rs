use image::{imageops, DynamicImage, ImageBuffer};
use ndarray::{Array3, Axis};
use ort::inputs;

use super::base::{BaseSession, BaseSessionTrait, SessionError, SessionOptions};

pub struct BirefnetSession {
    pub(crate) input_size: u32,
    pub(crate) mean: [f32; 3],
    pub(crate) std: [f32; 3],
    pub model_name: String,
    pub(crate) base_session: Option<BaseSession>,
}

impl BirefnetSession {
    pub fn new(debug: bool, session_options: SessionOptions) -> Self {
        let model_name = "BiRefNet-general-bb_swin_v1_tiny-epoch_232";
        let base_session = BaseSession::new(debug, session_options, model_name);

        if let Err(error) = base_session {
            panic!("Failed to create session: {}", error);
        }

        Self {
            input_size: 1024,
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
            model_name: model_name.to_string(),
            base_session: base_session.ok(),
        }
    }
}

impl BaseSessionTrait for BirefnetSession {
    fn get_model_name(&self) -> String {
        self.model_name.clone()
    }

    fn run(
        &self,
        original_image: DynamicImage,
    ) -> Result<ndarray::Array3<u8>, Box<dyn std::error::Error>> {
        let mask_size = self.input_size;
        let original_width = original_image.width();
        let original_height = original_image.height();
        log::info!(
            "Original image size: {}x{}",
            original_width,
            original_height
        );

        let resized_img = original_image.resize_exact(mask_size, mask_size, imageops::Lanczos3);
        let image_buffer_array = Array3::<u8>::from_shape_vec(
            (mask_size as usize, mask_size as usize, 3_usize),
            resized_img.to_rgb8().to_vec(),
        );

        if let Err(error) = image_buffer_array {
            log::error!("Error: {:?}", error);
            return Err(Box::new(SessionError::ImageProcessingError));
        }

        let mut input_array = image_buffer_array.unwrap().mapv(|x| x as f32 / 255.0);
        for i in 0..3 {
            input_array.index_axis_mut(Axis(i), i).map_mut(|pixel| {
                *pixel = (*pixel - self.mean[i]) / self.std[i];
            });
        }

        let input_tensor = input_array.permuted_axes([2, 0, 1]).insert_axis(Axis(0));

        let model = self.get_session();

        if model.is_none() {
            return Err(Box::new(SessionError::ModelLoadError));
        }

        let model = model.unwrap();
        let ort_inputs = inputs![input_tensor]?;

        let ort_outputs = model.run(ort_inputs)?;

        let output_tensor = ort_outputs[0].try_extract_tensor::<f32>()?;

        let alpha_mask_raw = output_tensor
            .to_shape((1, 1, mask_size as usize, mask_size as usize))?
            .remove_axis(Axis(0));

        let alpha_mask = alpha_mask_raw.permuted_axes([1, 2, 0]);

        let alpha_mask = alpha_mask.mapv(|x| (x * 255.0).round() as u8);
        let (mask_width, mask_height, _) = alpha_mask.dim();

        let alpha_image = DynamicImage::ImageLuma8(
            ImageBuffer::from_vec(
                mask_width as u32,
                mask_height as u32,
                alpha_mask.into_raw_vec(),
            )
            .unwrap_or_default(),
        );
        let output_image =
            alpha_image.resize_exact(original_width, original_height, imageops::Lanczos3);
        let output_array = Array3::from_shape_vec(
            (original_height as usize, original_width as usize, 1_usize),
            output_image.to_luma8().into_raw(),
        )?;

        Ok(output_array)
    }

    fn get_session(&self) -> Option<&ort::Session> {
        self.base_session
            .as_ref()
            .and_then(|s| s.inner_session.as_ref())
    }

    fn post_process(
        &self,
        output: ndarray::Array3<u8>,
        original_image: DynamicImage,
    ) -> Result<Array3<u8>, Box<dyn std::error::Error>> {
        let shape = [
            original_image.height() as usize,
            original_image.width() as usize,
            4,
        ];
        let mut output_img = Array3::from_shape_vec(shape, original_image.to_rgba8().to_vec())?;

        output_img
            .index_axis_mut(Axis(2), 3)
            .assign(&output.remove_axis(Axis(2)));

        Ok(output_img)
    }
}
