use dfdx::{
    prelude::Module,
    shapes::Rank3,
    tensor::{Cpu, Tensor, TensorFrom},
};
use mnist::{Mnist, MnistBuilder};

use crate::model_type::ModelBuild;

struct MnistTestSet {
    data: Mnist,
    index: usize,
    max_index: usize,
}

impl MnistTestSet {
    fn new(path: &str) -> Self {
        let data = MnistBuilder::new().base_path(path).finalize();
        let max_index = data.tst_lbl.len();
        Self {
            data,
            index: 0,
            max_index,
        }
    }
}

impl Iterator for MnistTestSet {
    type Item = (Vec<f32>, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.max_index {
            return None;
        }

        let start = self.index * 784;

        let mut img_data: Vec<f32> = Vec::with_capacity(784);
        img_data.extend(
            self.data.tst_img[start..start + 784]
                .iter()
                .map(|x| *x as f32 / 127.5 - 1.0),
        );

        let res = Some((img_data, self.data.tst_lbl[self.index] as usize));

        self.index += 1;

        res
    }
}

fn argmax(output: &[f32]) -> usize {
    let mut max_idx = 0usize;
    let mut max_out = output[0];

    for (idx, out) in output.iter().enumerate() {
        if *out > max_out {
            max_idx = idx;
            max_out = *out;
        }
    }

    max_idx
}

pub fn test(model: &ModelBuild) -> f32 {
    let device = Cpu::default();

    let mnist_path = "data/";
    let dataset = MnistTestSet::new(mnist_path);

    let mut errors = 0;
    let mut count = 0;

    for (img, lbl) in dataset {
        let data: Tensor<Rank3<1, 28, 28>, f32, _> = device.tensor(img);

        let output = model.forward(data);

        let output = output.as_vec();

        let predicted_label = argmax(&output);

        if predicted_label != lbl {
            errors += 1;
        }

        count += 1;
    }

    let errors_percent = (errors as f32 / count as f32) * 100.0;

    println!(
        "images: {}, errors: {errors}, error_percent: {:.2}%",
        count, errors_percent
    );

    errors_percent
}
