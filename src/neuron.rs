use crate::{utils::gen_rand_normal, val::BVal};

pub struct Neuron {
    pub weights: Vec<BVal>,
    pub bias: BVal,
}

impl Neuron {
    pub fn new(inputs_count: usize) -> Self {
        let mut weights = Vec::new();
        weights.resize_with(inputs_count as usize, || BVal::new(gen_rand_normal()));

        Neuron {
            weights,
            bias: BVal::new(gen_rand_normal()),
        }
    }

    pub fn forward(&self, inputs: &Vec<BVal>) -> BVal {
        assert_eq!(
            inputs.len(),
            self.weights.len(),
            "inputs supplied to neuron should have same size as its internal input weights"
        );

        let mut sum = self.bias.clone();
        for (input, weight) in inputs.iter().zip(self.weights.iter()) {
            let mul = input * weight;
            sum = &sum + &mul;
        }

        sum.tanh()
    }

    pub fn parameters(&self) -> Vec<BVal> {
        let mut res = self.weights.clone();
        res.push(self.bias.clone());
        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward() {
        let n = Neuron::new(3);

        let out = n.forward(&vec![BVal::new(1.0), BVal::new(2.0), BVal::new(3.0)]);

        assert!((out.borrow().d > -1.0) && (out.borrow().d < 1.0));
    }
}
