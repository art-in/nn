use crate::val::{BVal, Val};

use super::Op;

fn backward(child: &BVal) {
    let child = child.block();
    let parent = child.parents.0.as_ref().unwrap();

    parent.block_mut().grad += (1.0 - child.d.powf(2.0)) * child.grad;
}

impl BVal {
    pub fn tanh(&self) -> Self {
        let e = std::f64::consts::E.powf(2.0 * self.block().d);
        let d = (e - 1.0) / (e + 1.0);

        BVal::new_val(Val {
            d,
            parents: (Some(self.clone()), None),
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

        assert_eq!(b.block().d, 0.9051482536448664);
        assert_eq!(b.block().op, Op::Tanh);
    }

    #[test]
    fn parents() {
        let a = BVal::new(1.5);
        let b = a.tanh();

        assert!(a.block().parents.0.is_none());
        assert!(a.block().parents.1.is_none());

        assert!(a.block().op == Op::None);

        assert!(b.block().parents.0.is_some());
        assert!(b.block().parents.1.is_none());
        assert!(b.block().op == Op::Tanh);
    }

    #[test]
    fn backward() {
        let a = BVal::new(1.5);
        let b = a.tanh();

        b.block_mut().grad = 5.0;
        b.backward();

        assert_eq!(a.block().grad, 0.9035331946182429);
        assert_eq!(b.block().grad, 5.0);
    }
}
