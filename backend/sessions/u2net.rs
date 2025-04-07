use image::{imageops, DynamicImage};
use ndarray::Array3;
use ort::inputs;

use crate::utils::image_helper::{
    apply_mask_image, rgbau8_to_array3, tensor_f32_to_u8, tensor_hwc_to_bchw,
    tensor_resize_bilinear, MaskType,
};

use super::base::{BaseSession, BaseSessionTrait, SessionError, SessionOptions};

pub struct U2netSession {
    pub(crate) input_size: u32,
    pub(crate) mean: [f32; 3],
    pub(crate) std: [f32; 3],
    pub model_name: String,
    pub(crate) base_session: Option<BaseSession>,
}

impl U2netSession {
    pub fn new(debug: bool, session_options: SessionOptions) -> Self {
        let model_name = "u2net";
        let base_session = BaseSession::new(debug, session_options, model_name);

        if let Err(error) = base_session {
            panic!("Failed to create session: {}", error);
        }

        Self {
            input_size: 320,
            mean: [123.675, 116.28, 103.53],
            std: [58.395, 57.120, 57.375],
            model_name: model_name.to_string(),
            base_session: base_session.ok(),
        }
    }
}

impl BaseSessionTrait for U2netSession {
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
            resized_img.to_rgb8().into_raw(),
        );

        if let Err(error) = image_buffer_array {
            log::error!("Error: {:?}", error);
            return Err(Box::new(SessionError::ImageProcessingError));
        }

        let input_tensor = tensor_hwc_to_bchw(image_buffer_array.unwrap(), self.mean, self.std);

        let model = self.get_session();

        if model.is_none() {
            return Err(Box::new(SessionError::ModelLoadError));
        }

        let model = model.unwrap();
        let ort_inputs = inputs![
            "input.1" => input_tensor.view()
        ]?;

        let ort_outputs = model.run(ort_inputs)?;

        let output_tensor = ort_outputs[0].try_extract_tensor::<f32>()?;

        let alpha_mask_raw =
            output_tensor.to_shape((1, 1, mask_size as usize, mask_size as usize))?;

        let alpha_mask = alpha_mask_raw
            .permuted_axes([2, 3, 1, 0])
            .to_shape([mask_size as usize, mask_size as usize, 1])?
            .to_owned();

        let alpha_mask = tensor_resize_bilinear(
            alpha_mask,
            original_width as usize,
            original_height as usize,
            false,
        );

        Ok(tensor_f32_to_u8(alpha_mask))
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
        let output_img = rgbau8_to_array3(&original_image.to_rgba8())?;

        let output_img_tensor = apply_mask_image(output_img, output, MaskType::Background);

        Ok(output_img_tensor)
    }
}
