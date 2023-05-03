use crate::val::{BVal, Val};

use super::Op;

fn backward(child: &BVal) {
    let child = child.block();

    let mut lhs = child.parents.0.as_ref().unwrap().block_mut();
    let mut rhs = child.parents.1.as_ref().unwrap().block_mut();

    let base = lhs.d;
    let degree = rhs.d;

    lhs.grad += degree * base.powf(degree - 1.0) * child.grad;
    rhs.grad += child.d * child.grad;
}

impl BVal {
    pub fn pow(&self, degree: f64) -> BVal {
        self.pow_val(&BVal::new(degree))
    }

    pub fn pow_val(&self, degree: &BVal) -> BVal {
        BVal::new_val(Val {
            d: self.block().d.powf(degree.block().d),
            parents: (Some(self.clone()), Some(degree.clone())),
            op: Op::Pow,
            grad: 0.0,
            backward,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[test]
    fn forward() {
        let a = BVal::new(2.0);
        let b = a.pow(3.0);

        assert!(b == BVal::new(8.0));
        assert!(b.block().op == Op::Pow);
    }

    #[test]
    fn parents() {
        let a = BVal::new(2.0);
        let b = a.pow(3.0);

        assert!(a.block().parents.0.is_none());
        assert!(a.block().parents.1.is_none());

        assert!(a.block().op == Op::None);

        assert!(b.block().parents.0.as_ref().unwrap() == &a);
        assert!(b.block().parents.1.as_ref().unwrap() == &BVal::new(3.0));

        assert_eq!(
            Arc::as_ptr(b.block().parents.0.as_ref().unwrap()),
            Arc::as_ptr(&a)
        );

        assert!(b.block().op == Op::Pow);
    }

    #[test]
    fn backward() {
        let a = BVal::new(2.0);
        let b = a.pow(3.0);

        b.block_mut().grad = 2.0;
        b.backward();

        assert_eq!(a.block().grad, 24.0);
        assert!(b.block().grad == 2.0);
    }
}
