use autograd::val::BVal;

use crate::neuron::Neuron;

pub struct Layer {
    pub neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(inputs_count: usize, outputs_count: usize) -> Self {
        let mut neurons = Vec::new();
        neurons.resize_with(outputs_count, || Neuron::new(inputs_count));

        Layer { neurons }
    }

    pub fn forward(&self, inputs: Vec<BVal>) -> Vec<BVal> {
        let mut outputs = Vec::new();

        for n in &self.neurons {
            outputs.push(n.forward(&inputs))
        }

        outputs
    }

    pub fn parameters(&self) -> Vec<BVal> {
        let mut res = Vec::new();

        for neuron in &self.neurons {
            for param in neuron.parameters() {
                res.push(param);
            }
        }

        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward() {
        let l = Layer::new(3, 3);

        let outputs = l.forward(vec![BVal::new(1.0), BVal::new(2.0), BVal::new(3.0)]);

        for out in &outputs {
            assert!((out.borrow().d > -1.0) && (out.borrow().d < 1.0));
        }
    }
}
