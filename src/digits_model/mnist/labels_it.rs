use std::{fs::File, io::BufReader};

use super::utils;

pub struct LabelsIt {
    reader: BufReader<File>,
    labels_count: u32,
    current_label: u32,
}

impl LabelsIt {
    pub fn new(path: &str) -> Self {
        let file = File::open(path).expect("failed to open file");
        let mut reader = BufReader::new(file);

        let magic_number = utils::read_u32(&mut reader);
        let labels_count = utils::read_u32(&mut reader);

        assert!(magic_number == 2049, "invalid input file type");

        Self {
            reader,
            labels_count,
            current_label: 0,
        }
    }
}

impl Iterator for LabelsIt {
    type Item = u8;
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_label >= self.labels_count {
            None
        } else {
            let res = utils::read_u8(&mut self.reader);
            self.current_label += 1;
            Some(res)
        }
    }
}
