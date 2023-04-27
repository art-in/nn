use autograd::val::BVal;

pub fn label_to_outputs(label: u8) -> Vec<f64> {
    assert!(label <= 9, "label is out of valid range");

    let mut outputs = Vec::new();
    outputs.resize(10, 0.0);

    outputs[label as usize] = 1.0;

    outputs
}

pub fn calc_prediction_loss(output: &Vec<BVal>, expected: &Vec<f64>) -> BVal {
    assert_eq!(output.len(), expected.len());

    let mut loss = BVal::new(0.0);
    for idx in 0..output.len() {
        let l = (&output[idx] - expected[idx]).pow(2.0);
        loss = &loss + &l;
    }

    loss
}

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
    use super::*;

    #[test]
    fn test_label_to_outputs() {
        let cases = vec![
            (0, vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            (1, vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            (2, vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            (3, vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            (4, vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            (5, vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            (6, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            (7, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            (8, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            (9, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        ];

        for (label, expected) in cases {
            assert_eq!(label_to_outputs(label), expected);
        }
    }

    #[test]
    fn test_calc_prediction_loss() {
        let actual = vec![
            BVal::new(1.0),
            BVal::new(0.0),
            BVal::new(0.0),
            BVal::new(0.0),
            BVal::new(0.0),
            BVal::new(0.0),
            BVal::new(0.0),
            BVal::new(0.0),
            BVal::new(0.0),
            BVal::new(0.0),
        ];

        let expected = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let res = calc_prediction_loss(&actual, &expected);

        assert_eq!(res.borrow().d, 0.0);
    }

    #[test]
    fn test_calc_prediction_loss_complex() {
        let actual = vec![
            BVal::new(2.0), // loss = 2, loss^2 = 4
            BVal::new(0.0), // loss = 1, loss^2 = 1
            BVal::new(0.0),
            BVal::new(0.0),
            BVal::new(0.0),
            BVal::new(0.0),
            BVal::new(0.0),
            BVal::new(0.0),
            BVal::new(0.0),
            BVal::new(-1.0), // loss = 1, loss^2 = 1
        ];

        let expected = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let res = calc_prediction_loss(&actual, &expected);

        assert_eq!(res.borrow().d, 6.0);

        res.borrow_mut().grad = 1.0;
        res.backward();

        assert_eq!(actual[0].borrow().grad, 4.0);
        assert_eq!(actual[1].borrow().grad, -2.0);
        assert_eq!(actual[2].borrow().grad, 0.0);
        assert_eq!(actual[9].borrow().grad, -2.0);
    }

    #[test]
    fn test_calc_prediction_loss_max() {
        let actual = vec![
            BVal::new(-1.0),
            BVal::new(1.0),
            BVal::new(1.0),
            BVal::new(1.0),
            BVal::new(1.0),
            BVal::new(1.0),
            BVal::new(1.0),
            BVal::new(1.0),
            BVal::new(1.0),
            BVal::new(1.0),
        ];

        let expected = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let res = calc_prediction_loss(&actual, &expected);

        assert_eq!(res.borrow().d, 13.0);
    }

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

        for (label, output) in cases.iter().enumerate() {
            let output: Vec<BVal> = output.iter().map(|o| BVal::new(*o)).collect();
            assert_eq!(predict(&output), label as u8);
        }
    }
}
