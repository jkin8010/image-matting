use crate::sessions::base::{BaseSessionTrait, SessionOptions};
use crate::sessions::birefnet::BirefnetSession;
use image::codecs::jpeg::JpegEncoder;
use image::codecs::png::PngEncoder;
use image::{ExtendedColorType, ImageEncoder, ImageReader};
use once_cell::sync::OnceCell;
use rocket::{post, routes};
use std::path::PathBuf;
use tempfile::NamedTempFile;

use rocket::data::Data;
use rocket::http::{ContentType, Status};
use rocket::tokio::io;
use rocket_multipart_form_data::{
    mime, MultipartFormData, MultipartFormDataField, MultipartFormDataOptions,
};

static BIREFNET_SESSION: OnceCell<BirefnetSession> = OnceCell::new();

/// Helper function to parse multipart form data and retrieve the uploaded file.
async fn parse_uploaded_file(
    content_type: &ContentType,
    data: Data<'_>,
) -> io::Result<NamedTempFile> {
    let options = MultipartFormDataOptions::with_multipart_form_data_fields(vec![
        MultipartFormDataField::file("file")
            .size_limit(20 * 1024 * 1024) // 20MB
            .content_type_by_string(Some(mime::IMAGE_STAR))
            .unwrap(),
    ]);

    let multipart_form_data = MultipartFormData::parse(content_type, data, options)
        .await
        .map_err(|_| io::Error::new(io::ErrorKind::Other, "Failed to parse multipart form data"))?;

    let file_field = multipart_form_data
        .files
        .get("file")
        .and_then(|files| files.first())
        .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "No file found"))?;

    // 创建一个临时文件
    let temp_file = NamedTempFile::new().map_err(|e| {
        log::error!("Error creating temp file: {:?}", e);
        io::Error::new(io::ErrorKind::Other, "Error creating temp file")
    })?;

    // 将上传的文件内容复制到临时文件
    tokio::fs::copy(&file_field.path, temp_file.path())
        .await
        .map_err(|e| {
            log::error!("Error copying file: {:?}", e);
            io::Error::new(io::ErrorKind::Other, "Error copying file")
        })?;

    Ok(temp_file)
}

/// Helper function to decode an image from a file path.
fn decode_image(file_path: &std::path::Path) -> io::Result<image::DynamicImage> {
    let image_reader = ImageReader::open(file_path)?.with_guessed_format()?;
    image_reader.decode().map_err(|error| {
        log::error!("Error decoding image: {:?}", error);
        io::Error::new(io::ErrorKind::Other, "Error decoding image")
    })
}

/// Helper function to initialize or retrieve the Birefnet session.
fn get_birefnet_session() -> io::Result<&'static BirefnetSession> {
    let session_options = SessionOptions::new()
        .with_providers(vec!["cpu".to_owned()])
        .build()
        .map_err(|error| {
            log::error!("Error creating session options: {:?}", error);
            io::Error::new(io::ErrorKind::Other, error.to_string())
        })?;

    Ok(BIREFNET_SESSION.get_or_init(|| BirefnetSession::new(true, session_options)))
}

/// Helper function to encode an image buffer into the desired format.
fn encode_image(
    buffer: Vec<u8>,
    width: u32,
    height: u32,
    color_type: ExtendedColorType,
    format: &str,
) -> io::Result<Vec<u8>> {
    let mut output_buffer = Vec::new();
    match format {
        "png" => {
            let encoder = PngEncoder::new_with_quality(
                &mut output_buffer,
                image::codecs::png::CompressionType::Best,
                image::codecs::png::FilterType::Sub,
            );
            encoder
                .write_image(&buffer, width, height, color_type)
                .map_err(|error| {
                    log::error!("Error encoding PNG: {:?}", error);
                    io::Error::new(io::ErrorKind::Other, error.to_string())
                })?;
        }
        "jpeg" => {
            let encoder = JpegEncoder::new_with_quality(&mut output_buffer, 80);
            encoder
                .write_image(&buffer, width, height, color_type)
                .map_err(|error| {
                    log::error!("Error encoding JPEG: {:?}", error);
                    io::Error::new(io::ErrorKind::Other, error.to_string())
                })?;
        }
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Unsupported image format",
            ));
        }
    }
    Ok(output_buffer)
}

#[post("/rembg/image", format = "multipart/form-data", data = "<data>")]
pub async fn rembg(
    content_type: &ContentType,
    data: Data<'_>,
) -> io::Result<(Status, (ContentType, Vec<u8>))> {
    let temp_file = parse_uploaded_file(content_type, data).await?;
    let original_img = decode_image(temp_file.path())?;
    let session = get_birefnet_session()?;

    let alpha_mask = session.run(original_img.clone()).map_err(|error| {
        log::error!("Error running session: {:?}", error);
        io::Error::new(io::ErrorKind::Other, error.to_string())
    })?;

    let output_img_tensor = session
        .post_process(alpha_mask, original_img)
        .map_err(|error| {
            log::error!("Error post-processing image: {:?}", error);
            io::Error::new(io::ErrorKind::Other, error.to_string())
        })?;

    let (height, width, _) = output_img_tensor.dim();
    let img_buffer = output_img_tensor.into_raw_vec();

    let output_buffer = encode_image(
        img_buffer,
        width as u32,
        height as u32,
        ExtendedColorType::Rgba8,
        "png",
    )?;
    Ok((Status::Ok, (ContentType::PNG, output_buffer)))
}

#[post("/rembg/mask", format = "multipart/form-data", data = "<data>")]
pub async fn mask(
    content_type: &ContentType,
    data: Data<'_>,
) -> io::Result<(Status, (ContentType, Vec<u8>))> {
    let temp_file = parse_uploaded_file(content_type, data).await?;
    let original_img = decode_image(temp_file.path())?;
    let session = get_birefnet_session()?;

    let alpha_mask = session.run(original_img).map_err(|error| {
        log::error!("Error running session: {:?}", error);
        io::Error::new(io::ErrorKind::Other, error.to_string())
    })?;

    let (height, width, _) = alpha_mask.dim();
    let img_buffer = alpha_mask.into_raw_vec();

    let output_buffer = encode_image(
        img_buffer,
        width as u32,
        height as u32,
        ExtendedColorType::L8,
        "jpeg",
    )?;
    Ok((Status::Ok, (ContentType::JPEG, output_buffer)))
}

pub fn routes() -> Vec<rocket::Route> {
    routes![rembg, mask]
}
