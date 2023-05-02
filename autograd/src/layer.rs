use crate::{neuron::Neuron, pool::BValPool, val::BVal};

pub struct Layer {
    pub neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(inputs_count: usize, outputs_count: usize, pool: &BValPool) -> Self {
        let mut neurons = Vec::new();
        neurons.resize_with(outputs_count, || Neuron::new(inputs_count, pool));

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
        let pool = BValPool::default();
        let layer = Layer::new(3, 3, &pool);

        let pool = BValPool::default();
        let outputs = layer.forward(vec![pool.pull(1.0), pool.pull(2.0), pool.pull(3.0)]);

        for out in &outputs {
            assert!((out.borrow().d > -1.0) && (out.borrow().d < 1.0));
        }
    }
}
