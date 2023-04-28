use crate::val::{BVal, Val};

use super::Op;

fn backward(child: &BVal) {
    let parent = &child.borrow().parents[0];

    let child_d = child.borrow().d;
    let child_grad = child.borrow().grad;

    parent.borrow_mut().grad += (1.0 - child_d.powf(2.0)) * child_grad;
}

impl BVal {
    pub fn tanh(&self) -> Self {
        let e = std::f64::consts::E.powf(2.0 * self.borrow().d);
        let d = (e - 1.0) / (e + 1.0);

        BVal::new_val(Val {
            d,
            parents: vec![self.clone()],
            op: Op::Tanh,
            grad: 0.0,
            backward,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::ops::Op;

    use super::*;

    #[test]
    fn forward() {
        let a = BVal::new(1.5);
        let b = a.tanh();

        assert_eq!(b.borrow().d, 0.9051482536448664);
        assert_eq!(b.borrow().op, Op::Tanh);
    }

    #[test]
    fn parents() {
        let a = BVal::new(1.5);
        let b = a.tanh();

        assert!(a.borrow().parents.is_empty());
        assert!(a.borrow().op == Op::None);

        assert!(b.borrow().parents.len() == 1);
        assert!(b.borrow().op == Op::Tanh);
    }

    #[test]
    fn backward() {
        let a = BVal::new(1.5);
        let b = a.tanh();

        b.borrow_mut().grad = 5.0;
        b.backward();

        assert_eq!(a.borrow().grad, 0.9035331946182429);
        assert_eq!(b.borrow().grad, 5.0);
    }
}
