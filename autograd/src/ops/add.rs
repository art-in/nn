use std::ops::Add;

use crate::val::{BVal, Val};

use super::Op;

fn backward(child: &BVal) {
    let child = child.block();

    let lhs = child.parents.0.as_ref().unwrap();
    let rhs = child.parents.1.as_ref().unwrap();

    lhs.block_mut().grad += child.grad;
    rhs.block_mut().grad += child.grad;
}

impl Add<&BVal> for &BVal {
    type Output = BVal;

    fn add(self, other: &BVal) -> Self::Output {
        BVal::new_val(Val {
            d: self.block().d + other.block().d,
            parents: (Some(self.clone()), Some(other.clone())),
            op: Op::Add,
            grad: 0.0,
            backward,
        })
    }
}

impl Add<f64> for &BVal {
    type Output = BVal;

    fn add(self, other: f64) -> Self::Output {
        self + &BVal::new(other)
    }
}

impl Add<&BVal> for f64 {
    type Output = BVal;

    fn add(self, other: &BVal) -> Self::Output {
        &BVal::new(self) + other
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[test]
    fn forward_simple() {
        assert!(&BVal::new(1.0) + &BVal::new(2.0) == BVal::new(3.0));
        assert!(&BVal::new(1.0) + 2.0 == BVal::new(3.0));
        assert!(1.0 + &BVal::new(2.0) == BVal::new(3.0));
    }

    #[test]
    fn forward_complex() {
        let a = BVal::new(1.0);
        let b = BVal::new(2.0);
        let c = &a + &b;
        let d = &c + &a;

        assert!(d == BVal::new(4.0));
        assert!(d.block().op == Op::Add);
    }

    #[test]
    fn parents_simple() {
        let a = BVal::new(1.0);
        let b = BVal::new(2.0);
        let c = &a + &b;

        assert!(a.block().parents.0.is_none());
        assert!(a.block().parents.1.is_none());

        assert!(b.block().parents.0.is_none());
        assert!(b.block().parents.1.is_none());

        assert!(a.block().op == Op::None);
        assert!(b.block().op == Op::None);

        assert!(c.block().parents.0.as_ref().unwrap() == &a);
        assert!(c.block().parents.1.as_ref().unwrap() == &b);

        assert!(c.block().op == Op::Add);
    }

    #[test]
    fn parents_complex() {
        let a = BVal::new(1.0);
        let b = BVal::new(2.0);
        let c = &a + &b;
        let d = &c + &a;

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

        assert!(c.block().op == Op::Add);

        assert_eq!(
            Arc::as_ptr(d.block().parents.0.as_ref().unwrap()),
            Arc::as_ptr(&c)
        );
        assert_eq!(
            Arc::as_ptr(d.block().parents.1.as_ref().unwrap()),
            Arc::as_ptr(&a)
        );

        assert!(d.block().op == Op::Add);
    }

    #[test]
    fn backward() {
        let a = BVal::new(1.0);
        let b = BVal::new(2.0);
        let c = &a + &b;

        c.block_mut().grad = 5.0;
        c.backward();

        assert!(a.block().grad == 5.0);
        assert!(b.block().grad == 5.0);
        assert!(c.block().grad == 5.0);
    }
}
