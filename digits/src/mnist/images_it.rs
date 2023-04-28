use std::{fs::File, io::BufReader};

use super::utils;

pub struct ImagesIt {
    reader: BufReader<File>,
    images_count: u32,
    image_height: u32,
    image_width: u32,
    current_image: u32,
}

impl ImagesIt {
    pub fn new(path: &str) -> Self {
        let file = File::open(path).expect("failed to open file");
        let mut reader = BufReader::new(file);

        let magic_number = utils::read_u32(&mut reader);
        let images_count = utils::read_u32(&mut reader);
        let image_height = utils::read_u32(&mut reader);
        let image_width = utils::read_u32(&mut reader);

        assert!(magic_number == 2051, "invalid input file type");

        Self {
            reader,
            images_count,
            current_image: 0,
            image_height,
            image_width,
        }
    }

    pub fn image_size(&self) -> (u32, u32) {
        (self.image_width, self.image_height)
    }

    pub fn image_width(&self) -> u32 {
        self.image_width
    }

    pub fn image_height(&self) -> u32 {
        self.image_height
    }

    pub fn images_count(&self) -> u32 {
        self.images_count
    }
}

impl Iterator for ImagesIt {
    type Item = Vec<f64>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_image >= self.images_count {
            None
        } else {
            let image_size = self.image_size();
            let sz = image_size.0 * image_size.1;

            let res = utils::read_vec_u8(&mut self.reader, sz);
            let res = res.iter().map(|v| *v as f64 / 127.5 - 1.0).collect();

            self.current_image += 1;
            Some(res)
        }
    }
}
