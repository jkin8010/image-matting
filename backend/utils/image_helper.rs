use base64::prelude::*;
use image::{DynamicImage, ImageBuffer, Luma, Rgba};
use ndarray::{Array3, Array4, ErrorKind, ShapeError};

pub fn tensor_resize_bilinear(
    image_tensor: Array3<f32>,
    new_width: usize,
    new_height: usize,
    proportional: bool,
) -> Array3<f32> {
    let shape = image_tensor.shape();
    let src_height = shape[0] as usize;
    let src_width = shape[1] as usize;
    let src_channels = shape[2] as usize;

    let mut scale_x = src_width as f64 / new_width as f64;
    let mut scale_y = src_height as f64 / new_height as f64;

    if proportional {
        let downscaling = scale_x.max(scale_y) > 1.0;
        scale_y = if downscaling {
            scale_x.max(scale_y)
        } else {
            scale_x.min(scale_y)
        };
        scale_x = scale_y;
    }

    // Create a new NdArray to store the resized image
    let mut resized_image_data = Array3::<f32>::zeros([new_height, new_width, src_channels]);

    // Perform interpolation to fill the resized NdArray
    for y in 0..new_height {
        for x in 0..new_width {
            let src_x = x as f64 * scale_x;
            let src_y = y as f64 * scale_y;
            let x1 = f64::max(src_x.floor(), 0.0);
            let x2 = f64::min(src_x.ceil(), src_width as f64 - 1.0);
            let y1 = f64::max(src_y.floor(), 0.0);
            let y2 = f64::min(src_y.ceil(), src_height as f64 - 1.0);

            let dx = src_x - x1 as f64;
            let dy = src_y - y1 as f64;

            for c in 0..src_channels {
                let p1 = image_tensor
                    .get((y1 as usize, x1 as usize, c))
                    .map(|n| *n)
                    .unwrap_or_default();
                let p2 = image_tensor
                    .get((y1 as usize, x2 as usize, c))
                    .map(|n| *n)
                    .unwrap_or_default();
                let p3 = image_tensor
                    .get((y2 as usize, x1 as usize, c))
                    .map(|n| *n)
                    .unwrap_or_default();
                let p4 = image_tensor
                    .get((y2 as usize, x2 as usize, c))
                    .map(|n| *n)
                    .unwrap_or_default();

                // Perform bilinear interpolation
                let interpolated_value = (1.0 - dx) * (1.0 - dy) * p1 as f64
                    + dx * (1.0 - dy) * p2 as f64
                    + (1.0 - dx) * dy * p3 as f64
                    + dx * dy * p4 as f64;

                resized_image_data
                    .get_mut((y, x, c))
                    .map(|n| *n = interpolated_value as f32);
            }
        }
    }

    resized_image_data
}

pub fn tensor_u8_to_f32(input_tensor: Array3<u8>) -> Array3<f32> {
    input_tensor.map(|&n| n as f32 / 255.0)
}

pub fn tensor_f32_to_u8(input_tensor: Array3<f32>) -> Array3<u8> {
    input_tensor.map(|&n| (n * 255.0) as u8)
}

pub fn tensor_hwc_to_bchw(image_tensor: Array3<u8>, mean: [f32; 3], std: [f32; 3]) -> Array4<f32> {
    let shape = image_tensor.shape();
    let src_height = shape[0];
    let src_width = shape[1];
    let channels = shape[2];

    let mut output_tensor = Array4::zeros((1, channels, src_height, src_width));

    for (dim, value) in image_tensor.indexed_iter() {
        let (y, x, c) = dim;
        let new_value = *value as f32; // / 255.0;
        output_tensor[[0, c, y, x]] = (new_value - mean[c]) / std[c];
    }

    output_tensor
}

pub fn hwc_to_chw(image_tensor: Array3<u8>) -> Array3<u8> {
    image_tensor.permuted_axes([2, 0, 1])
}

pub fn chw_to_hwc(image_tensor: Array3<u8>) -> Array3<u8> {
    image_tensor.permuted_axes([1, 2, 0])
}

pub enum MaskType {
    Object,
    Background,
}

pub fn apply_mask_image(
    image_tensor: Array3<u8>,
    mask_tensor: Array3<u8>,
    mask_type: MaskType,
) -> Array3<u8> {
    let mut masked_image = image_tensor.clone();

    for ((y, x, channels), value) in masked_image.indexed_iter_mut() {
        if channels != 3 {
            continue;
        }

        let mask_value = mask_tensor.get((y, x, 0)).map(|n| *n).unwrap_or(0);

        *value = match mask_type {
            MaskType::Object => {
                if mask_value != 0 {
                    0
                } else {
                    255_u8 - mask_value
                }
            }
            MaskType::Background => {
                if mask_value != 0 {
                    mask_value
                } else {
                    0
                }
            }
        };
    }

    masked_image
}

pub fn array3_to_image(
    image_tensor: Array3<u8>,
) -> Result<ImageBuffer<Luma<u8>, Vec<u8>>, ShapeError> {
    let (height, width, _) = image_tensor.dim();
    let image_buffer = image_tensor.into_raw_vec();
    let image = ImageBuffer::from_raw(width as u32, height as u32, image_buffer);

    if image.is_none() {
        return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
    }

    Ok(image.unwrap())
}

pub fn rgbau8_to_array3(image: &ImageBuffer<Rgba<u8>, Vec<u8>>) -> Result<Array3<u8>, ShapeError> {
    let width = image.width();
    let height = image.height();
    let channels = 4;

    Array3::<u8>::from_shape_vec(
        (height as usize, width as usize, channels),
        image.as_raw().to_vec(),
    )
}

pub fn base64_to_image(base64_string: &str) -> Result<DynamicImage, Box<dyn std::error::Error>> {
    let buffer = BASE64_STANDARD.decode(base64_string)?;
    let image = image::load_from_memory(&buffer)?;
    Ok(image)
}
