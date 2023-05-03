use std::ops::Mul;

use crate::val::{BVal, Val};

use super::Op;

fn backward(child: &BVal) {
    let child = child.block();

    let mut lhs = child.parents.0.as_ref().unwrap().block_mut();
    let mut rhs = child.parents.1.as_ref().unwrap().block_mut();

    lhs.grad += rhs.d * child.grad;
    rhs.grad += lhs.d * child.grad;
}

impl Mul<&BVal> for &BVal {
    type Output = BVal;

    fn mul(self, other: &BVal) -> Self::Output {
        BVal::new_val(Val {
            d: self.block().d * other.block().d,
            parents: (Some(self.clone()), Some(other.clone())),
            op: Op::Mul,
            grad: 0.0,
            backward,
        })
    }
}

impl Mul<f64> for &BVal {
    type Output = BVal;

    fn mul(self, other: f64) -> Self::Output {
        self * &BVal::new(other)
    }
}

impl Mul<&BVal> for f64 {
    type Output = BVal;

    fn mul(self, other: &BVal) -> Self::Output {
        &BVal::new(self) * other
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[test]
    fn simple() {
        assert!(&BVal::new(1.5) * &BVal::new(2.0) == BVal::new(3.0));
        assert!(&BVal::new(1.5) * 2.0 == BVal::new(3.0));
        assert!(1.5 * &BVal::new(2.0) == BVal::new(3.0));
    }

    #[test]
    fn complex() {
        let a = BVal::new(1.5);
        let b = BVal::new(2.0);
        let c = &a * &b;
        let d = &c * &a;

        assert!(d == BVal::new(4.5));
        assert!(d.block().op == Op::Mul);
    }

    #[test]
    fn parents_simple() {
        let a = BVal::new(1.5);
        let b = BVal::new(2.0);
        let c = &a * &b;

        assert!(a.block().parents.0.is_none());
        assert!(a.block().parents.1.is_none());

        assert!(b.block().parents.0.is_none());
        assert!(b.block().parents.1.is_none());

        assert!(a.block().op == Op::None);
        assert!(b.block().op == Op::None);

        assert!(c.block().parents.0.as_ref().unwrap() == &a);
        assert!(c.block().parents.1.as_ref().unwrap() == &b);

        assert!(c.block().op == Op::Mul);
    }

    #[test]
    fn parents_complex() {
        let a = BVal::new(1.5);
        let b = BVal::new(2.0);
        let c = &a * &b;
        let d = &c * &a;

        assert!(a.block().parents.0.is_none());
        assert!(a.block().parents.1.is_none());

        assert!(b.block().parents.0.is_none());
        assert!(b.block().parents.1.is_none());

        assert!(a.block().op == Op::None);
        assert!(b.block().op == Op::None);

        assert_eq!(
            Arc::as_ptr(c.block().parents.0.as_ref().unwrap()),
            Arc::as_ptr(&a)
        );
        assert_eq!(
            Arc::as_ptr(c.block().parents.1.as_ref().unwrap()),
            Arc::as_ptr(&b)
        );

        assert!(c.block().op == Op::Mul);

        assert_eq!(
            Arc::as_ptr(d.block().parents.0.as_ref().unwrap()),
            Arc::as_ptr(&c)
        );
        assert_eq!(
            Arc::as_ptr(d.block().parents.1.as_ref().unwrap()),
            Arc::as_ptr(&a)
        );

        assert!(d.block().op == Op::Mul);
    }

    #[test]
    fn backward() {
        let a = BVal::new(2.0);
        let b = BVal::new(-3.0);
        let c = &a * &b;

        c.block_mut().grad = 5.0;
        c.backward();

        assert!(a.block().grad == -15.0);
        assert!(b.block().grad == 10.0);
        assert!(c.block().grad == 5.0);
    }
}
