use autograd::val::BVal;

pub fn predict(output: &Vec<BVal>) -> u8 {
    assert_eq!(output.len(), 10);

    let mut max_out = f64::MIN;
    let mut max_label: u8 = 10;

    for (label, out) in output.iter().enumerate() {
        let out = out.borrow().d;
        if out > max_out {
            max_out = out;
            max_label = label as u8;
        }
    }

    max_label
}

#[cfg(test)]
mod tests {
    use autograd::{pool::BValPool, val::BVal};

    use super::*;

    #[test]
    fn test_predict() {
        let cases = vec![
            // 0
            vec![1.0, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99],
            // 1
            vec![
                -0.99, 1.0, -0.99, -0.99, -0.99, -0.99, -0.99, -0.99, -0.99, -0.99,
            ],
            // 2
            vec![1.0, 1.0, 100.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 99.0],
            // 3
            vec![
                -0.99, -0.99, -0.99, -0.5, -0.99, -0.99, -0.99, -0.99, -0.99, -0.99,
            ],
            // 4
            vec![-1.0, -1.0, -1.0, -1.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0],
            // 5
            vec![
                -1.0, -0.99, -0.98, -0.97, -0.96, -0.9, -0.95, -0.94, -0.93, -0.92,
            ],
            // 6
            vec![1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-1, 1e-6, 1e-6, 1e-6],
            // 7
            vec![1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e7, 1e6, 1e6],
            // 8
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            //9
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ];

        let pool = BValPool::default();

        for (label, output) in cases.iter().enumerate() {
            let output: Vec<BVal> = output.iter().map(|o| pool.pull(*o)).collect();
            assert_eq!(predict(&output), label as u8);
        }
    }
}
