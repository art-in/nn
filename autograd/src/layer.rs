use std::sync::Mutex;

use scoped_threadpool::Pool;

use crate::{neuron::Neuron, val::BVal};

pub struct Layer {
    pub neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(inputs_count: usize, outputs_count: usize) -> Self {
        let mut neurons = Vec::new();
        neurons.resize_with(outputs_count, || Neuron::new(inputs_count));

        Layer { neurons }
    }

    pub fn forward(&self, inputs: Vec<BVal>, pool: &mut Pool) -> Vec<BVal> {
        let outputs = Mutex::new(Vec::new());

        pool.scoped(|scoped| {
            for n in &self.neurons {
                scoped.execute(|| {
                    let output = n.forward(&inputs);
                    outputs.lock().unwrap().push(output);
                });
            }
        });

        outputs.into_inner().unwrap()
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
        let mut threads = Pool::new(1);
        let l = Layer::new(3, 3);

        let outputs = l.forward(
            vec![BVal::new(1.0), BVal::new(2.0), BVal::new(3.0)],
            &mut threads,
        );

        for out in &outputs {
            assert!((out.block().d > -1.0) && (out.block().d < 1.0));
        }
    }
}
