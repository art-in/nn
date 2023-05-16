use dfdx::data::ExactSizeDataset;

use crate::mnist::{Mnist, MnistBuilder};

pub enum MnistDataSetKind {
    Train,
    Test,
}

pub struct MnistDataSet(Mnist, MnistDataSetKind);

impl MnistDataSet {
    pub fn new(path: &str, kind: MnistDataSetKind) -> Self {
        Self(MnistBuilder::new().base_path(path).finalize(), kind)
    }

    pub fn get_images(&self) -> &Vec<u8> {
        match self.1 {
            MnistDataSetKind::Train => &self.0.trn_img,
            MnistDataSetKind::Test => &self.0.tst_img,
        }
    }

    pub fn get_labels(&self) -> &Vec<u8> {
        match self.1 {
            MnistDataSetKind::Train => &self.0.trn_lbl,
            MnistDataSetKind::Test => &self.0.tst_lbl,
        }
    }
}

const IMAGE_SIZE: usize = 28;
const IMAGE_PIXELS_COUNT: usize = IMAGE_SIZE.pow(2);

impl ExactSizeDataset for MnistDataSet {
    type Item<'a> = (Vec<f32>, usize) where Self: 'a;

    fn get(&self, index: usize) -> Self::Item<'_> {
        let images = self.get_images();
        let labels = self.get_labels();

        let mut image: Vec<f32> = Vec::with_capacity(IMAGE_PIXELS_COUNT);
        let start = IMAGE_PIXELS_COUNT * index;
        image.extend(
            images[start..start + IMAGE_PIXELS_COUNT]
                .iter()
                .map(|x| *x as f32 / 255.0),
        );

        (image, labels[index] as usize)
    }

    fn len(&self) -> usize {
        self.get_labels().len()
    }
}
