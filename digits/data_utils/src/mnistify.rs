use std::{
    fs::File,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

use glob::glob;
use image::{DynamicImage, GenericImageView};

const TRAIN_IMAGES_FILE_NAME: &str = "train-images-idx3-ubyte";
const TRAIN_LABELS_FILE_NAME: &str = "train-labels-idx1-ubyte";
const TEST_IMAGES_FILE_NAME: &str = "t10k-images-idx3-ubyte";
const TEST_LABELS_FILE_NAME: &str = "t10k-labels-idx1-ubyte";

const IMAGES_FILE_MAGIC_NUMBER: u32 = 2051;
const LABELS_FILE_MAGIC_NUMBER: u32 = 2049;

const IMAGE_SIZE: u32 = 28;
const IMAGE_PIXELS: u32 = IMAGE_SIZE.pow(2);

// packs image files into MNIST dataset format
pub fn mnistify(train_dir_path: &str, test_dir_path: &str, output_dir_path: &str) {
    write_dataset(
        train_dir_path,
        output_dir_path,
        TRAIN_IMAGES_FILE_NAME,
        TRAIN_LABELS_FILE_NAME,
    );

    write_dataset(
        test_dir_path,
        output_dir_path,
        TEST_IMAGES_FILE_NAME,
        TEST_LABELS_FILE_NAME,
    );
}

pub fn write_dataset(
    images_dir_path: &str,
    output_dir_path: &str,
    output_images_filename: &str,
    output_labels_filename: &str,
) {
    let mut output_images_file = BufWriter::new(
        File::create(Path::new(output_dir_path).join(output_images_filename))
            .expect("failed to create file"),
    );

    let mut output_labels_file = BufWriter::new(
        File::create(Path::new(output_dir_path).join(output_labels_filename))
            .expect("failed to create file"),
    );

    let images_path_pattern = Path::new(images_dir_path).join("*.png");
    let images_path_pattern = images_path_pattern.to_str().unwrap();

    let images_paths = glob(images_path_pattern).expect("failed to read pattern");
    let images_paths = images_paths.collect::<Vec<_>>();
    let images_paths: Vec<&PathBuf> = images_paths
        .iter()
        .map(|r| r.as_ref().expect("failed to read path"))
        .collect();

    // write magic numbers
    output_images_file
        .write_all(&IMAGES_FILE_MAGIC_NUMBER.to_be_bytes())
        .expect("failed to write to file");

    output_labels_file
        .write_all(&LABELS_FILE_MAGIC_NUMBER.to_be_bytes())
        .expect("failed to write to file");

    // write image counts
    output_images_file
        .write_all(&(images_paths.len() as u32).to_be_bytes())
        .expect("failed to write to file");

    output_labels_file
        .write_all(&(images_paths.len() as u32).to_be_bytes())
        .expect("failed to write to file");

    // write image size (height / width)
    output_images_file
        .write_all(&IMAGE_SIZE.to_be_bytes())
        .expect("failed to write to file");

    output_images_file
        .write_all(&IMAGE_SIZE.to_be_bytes())
        .expect("failed to write to file");

    write_digits(
        images_paths,
        &mut output_images_file,
        &mut output_labels_file,
    );
}

pub fn write_digits(
    images_paths: Vec<&PathBuf>,
    output_images: &mut impl Write,
    output_labels: &mut impl Write,
) {
    for path in images_paths {
        let input_image = image::open(path).expect("failed to open file");
        let (width, height) = input_image.dimensions();

        assert_eq!(width, IMAGE_SIZE as u32, "invalid image size");
        assert_eq!(height, IMAGE_SIZE as u32, "invalid image size");

        output_images
            .write_all(&convert_rgb_image_to_mnist(&input_image))
            .expect("failed to write to file");

        output_labels
            .write_all(&[get_label_from_filename(path)])
            .expect("failed to write to file");
    }
}

fn get_label_from_filename(path: &Path) -> u8 {
    let name = path.file_name().unwrap().to_str().unwrap();
    let parts: Vec<&str> = name.split('-').collect();
    let parts: Vec<&str> = parts[1].split('.').collect();
    let label = parts[0];
    label.parse().expect("failed to parse label from filename")
}

fn convert_rgb_image_to_mnist(image: &DynamicImage) -> [u8; IMAGE_PIXELS as usize] {
    let input_buffer = image.as_rgba8().expect("failed to read image");
    let mut output_buffer: [u8; IMAGE_SIZE.pow(2) as usize] = [0; IMAGE_SIZE.pow(2) as usize];
    let mut output_idx = 0;

    for y in 0..IMAGE_SIZE {
        for x in 0..IMAGE_SIZE {
            let px = input_buffer.get_pixel(x, y);

            // empty pixel is opaque white     - rgba(255, 255, 255, 255)
            // non-empty pixel is opaque black - rgba(0, 0, 0, 255)
            let mnist_px = 255 - px[0];

            output_buffer[output_idx] = mnist_px;
            output_idx += 1;
        }
    }

    output_buffer
}
