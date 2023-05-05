use std::{
    fs::{self, File},
    io::{BufReader, BufWriter, Read},
};

use autograd::val::BVal;

use crate::{layer::Layer, utils};

pub struct Network {
    pub layers: Vec<Layer>,
    parameters: Vec<BVal>,
}

impl Network {
    pub fn new(layers_sizes: Vec<usize>) -> Network {
        let mut layers = Vec::new();

        for i in 0..(layers_sizes.len() - 1) {
            layers.push(Layer::new(layers_sizes[i], layers_sizes[i + 1]))
        }

        let mut parameters = Vec::new();
        for layer in &layers {
            for param in layer.parameters() {
                parameters.push(param);
            }
        }

        Network { layers, parameters }
    }

    pub fn forward(&self, inputs: &[f64]) -> Vec<BVal> {
        let mut res: Vec<BVal> = inputs.iter().map(|v| BVal::new(*v)).collect();

        for layer in &self.layers {
            res = layer.forward(res);
        }

        res
    }

    pub fn parameters(&self) -> &Vec<BVal> {
        &self.parameters
    }

    pub fn reset_grad(&self) {
        for param in self.parameters() {
            param.borrow_mut().grad = 0.0;
        }
    }

    fn get_layer_sizes(&self) -> Vec<usize> {
        let mut layers_sizes: Vec<usize> = Vec::new();

        // input size
        layers_sizes.push(self.layers[0].neurons[0].weights.len());

        for layer in &self.layers {
            layers_sizes.push(layer.neurons.len());
        }

        layers_sizes
    }

    fn serialize_to_file_path(&self, path: &str) {
        let file = File::create(path).expect("failed to create file");
        let mut writer = BufWriter::new(file);

        // write network structure
        utils::write_u32(&mut writer, self.layers.len() as u32);
        utils::write_u32(&mut writer, self.layers[0].neurons[0].weights.len() as u32);

        for layer in &self.layers {
            utils::write_u32(&mut writer, layer.neurons.len() as u32);
        }

        // write params
        for param in self.parameters() {
            let d = param.borrow().d;
            utils::write_f64(&mut writer, d);
        }
    }

    pub fn serialize_to_file(&self, dir: &str, file_name_prefix: &str) {
        let path = utils::get_model_file_path(dir, file_name_prefix, &self.get_layer_sizes());
        println!("serializing network to file: {}", path);
        self.serialize_to_file_path(&path);
    }

    pub fn deserialize_from_reader(mut reader: impl Read) -> Self {
        // read network structure
        let layers_count = utils::read_u32(&mut reader);
        let mut layers_sizes: Vec<usize> = Vec::new();

        let inputs_size = utils::read_u32(&mut reader);
        layers_sizes.push(inputs_size as usize);

        for _ in 0..layers_count {
            layers_sizes.push(utils::read_u32(&mut reader) as usize);
        }

        // init network
        let net = Network::new(layers_sizes);

        // read params
        for param in net.parameters() {
            let d = utils::read_f64(&mut reader);
            param.borrow_mut().d = d;
        }

        net
    }

    fn deserialize_from_file_path(path: &str) -> Self {
        let file = File::open(path).expect("failed to open file");
        let reader = BufReader::new(file);

        Self::deserialize_from_reader(reader)
    }

    pub fn new_or_deserialize_from_file(
        layers_sizes: Vec<usize>,
        dir: &str,
        file_name_prefix: &str,
    ) -> Network {
        let path = utils::get_model_file_path(dir, file_name_prefix, &layers_sizes);

        if fs::metadata(&path).is_ok() {
            println!("deserializing network from file: {}", path);
            Network::deserialize_from_file_path(&path)
        } else {
            println!("initializing network: {:?}", layers_sizes);
            Network::new(layers_sizes)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    #[test]
    fn forward() {
        let net = Network::new(vec![3, 4, 2]);
        let outputs = net.forward(&vec![1.0, 2.0, 3.0]);

        assert_eq!(outputs.len(), 2);
    }

    #[test]
    fn parameters() {
        let net = Network::new(vec![3, 4, 2]);
        let params = net.parameters();

        assert_eq!(params.len(), 26);
    }

    #[test]
    fn classification() {
        let inputs: Vec<Vec<f64>> = vec![
            vec![2.0, 3.0, -1.0],
            vec![3.0, -1.0, 0.5],
            vec![0.5, 1.0, 1.0],
            vec![1.0, 1.0, -1.0],
        ];
        let expecteds: Vec<f64> = vec![1.0, -1.0, -1.0, 1.0];

        let net = Network::new(vec![3, 4, 4, 1]);

        let mut last_total_loss = BVal::new(0.0);

        for _ in 0..100 {
            // forward
            let mut total_loss = BVal::new(0.0);
            for (input, expected) in inputs.iter().zip(expecteds.iter()) {
                let output = &net.forward(input)[0];
                let loss = (*expected - output).pow(2.0);
                total_loss = &total_loss + &loss;
                last_total_loss = total_loss.clone();
            }

            // backward
            net.reset_grad();

            total_loss.borrow_mut().grad = 1.0;
            total_loss.backward();

            // update
            for param in net.parameters() {
                let grad = param.borrow().grad;
                param.borrow_mut().d -= 0.05 * grad;
            }
        }

        assert!(last_total_loss.borrow().d < 0.1);
    }

    #[test]
    fn serialization() {
        const FILE_PATH: &str = "test.nm";

        let net1 = Network::new(vec![3, 4, 4, 1]);
        net1.serialize_to_file_path(FILE_PATH);

        let net2 = Network::deserialize_from_file_path(FILE_PATH);

        fs::remove_file(FILE_PATH).expect("failed to remove file");

        let layers1 = &net1.layers;
        let layers2 = &net1.layers;

        let layers = layers1.iter().zip(layers2.iter());

        for (layer1, layer2) in layers {
            assert_eq!(layer1.neurons.len(), layer2.neurons.len());
        }

        let params1 = net1.parameters();
        let params2 = net2.parameters();

        assert_eq!(params1.len(), params2.len());

        let params_it = params1.iter().zip(params2.iter());

        for (param1, param2) in params_it {
            assert_eq!(param1.borrow().d, param2.borrow().d);
            assert_eq!(param1.borrow().op, param2.borrow().op);
            assert_eq!(param1.borrow().grad, param2.borrow().grad);
        }
    }
}
