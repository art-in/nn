use crate::{layer::Layer, val::BVal};

// Multi-Layer Perceptron
pub struct MLP {
    pub layers: Vec<Layer>,
}

impl MLP {
    pub fn new(layers_sizes: Vec<usize>) -> MLP {
        let mut layers = Vec::new();

        for i in 0..(layers_sizes.len() - 1) {
            layers.push(Layer::new(layers_sizes[i], layers_sizes[i + 1]))
        }

        MLP { layers }
    }

    pub fn forward(&self, inputs: &Vec<f64>) -> Vec<BVal> {
        let mut res: Vec<BVal> = inputs.iter().map(|v| BVal::new(*v)).collect();

        for layer in &self.layers {
            res = layer.forward(res);
        }

        res
    }

    pub fn parameters(&self) -> Vec<BVal> {
        let mut res = Vec::new();

        for layer in &self.layers {
            for param in layer.parameters() {
                res.push(param);
            }
        }

        res
    }

    pub fn reset_grad(&self) {
        for param in &self.parameters() {
            param.borrow_mut().grad = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward() {
        let net = MLP::new(vec![3, 4, 2]);
        let outputs = net.forward(&vec![1.0, 2.0, 3.0]);

        assert_eq!(outputs.len(), 2);
    }

    #[test]
    fn parameters() {
        let net = MLP::new(vec![3, 4, 2]);
        let params = net.parameters();

        assert_eq!(params.len(), 26);
    }

    #[test]
    fn classifier() {
        let inputs: Vec<Vec<f64>> = vec![
            vec![2.0, 3.0, -1.0],
            vec![3.0, -1.0, 0.5],
            vec![0.5, 1.0, 1.0],
            vec![1.0, 1.0, -1.0],
        ];
        let expecteds: Vec<f64> = vec![1.0, -1.0, -1.0, 1.0];

        let net = MLP::new(vec![3, 4, 4, 1]);

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
            for param in &net.parameters() {
                let grad = param.borrow().grad;
                param.borrow_mut().d -= 0.05 * grad;
            }
        }

        assert!(last_total_loss.borrow().d < 0.1);
    }
}
