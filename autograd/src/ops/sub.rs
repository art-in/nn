use std::ops::Sub;

use crate::val::BVal;

impl Sub<&BVal> for &BVal {
    type Output = BVal;

    fn sub(self, other: &BVal) -> Self::Output {
        self + &-other
    }
}

impl Sub<f64> for &BVal {
    type Output = BVal;

    fn sub(self, other: f64) -> Self::Output {
        self + &-&BVal::new(other)
    }
}

impl Sub<&BVal> for f64 {
    type Output = BVal;

    fn sub(self, other: &BVal) -> Self::Output {
        &BVal::new(self) + &-other
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::ops::Op;

    use super::*;

    #[test]
    fn simple() {
        assert_eq!(&BVal::new(3.0) - &BVal::new(2.0), BVal::new(1.0));
        assert!(&BVal::new(3.0) - 2.0 == BVal::new(1.0));
        assert!(3.0 - &BVal::new(2.0) == BVal::new(1.0));
    }

    #[test]
    fn complex() {
        let a = BVal::new(3.0);
        let b = BVal::new(2.0);
        let c = &a - &b;
        let d = &c - &a;

        assert!(d == BVal::new(-2.0));
        assert!(d.block().op == Op::Add);
    }

    #[test]
    fn parents_simple() {
        let a = BVal::new(3.0);
        let b = BVal::new(2.0);
        let c = &a - &b;

        assert!(a.block().parents.0.is_none());
        assert!(a.block().parents.1.is_none());

        assert!(b.block().parents.0.is_none());
        assert!(b.block().parents.1.is_none());

        assert!(a.block().op == Op::None);
        assert!(b.block().op == Op::None);

        assert!(c.block().parents.0.as_ref().unwrap() == &a);
        assert!(c.block().parents.1.as_ref().unwrap() == &BVal::new(-2.0));

        assert!(c.block().op == Op::Add);
    }

    #[test]
    fn parents_complex() {
        let a = BVal::new(3.0);
        let b = BVal::new(2.0);
        let c = &a - &b;
        let d = &c - &a;

        assert!(a.block().parents.0.is_none());
        assert!(a.block().parents.1.is_none());

        assert!(b.block().parents.0.is_none());
        assert!(b.block().parents.1.is_none());

        assert!(a.block().op == Op::None);
        assert!(b.block().op == Op::None);

        assert!(c.block().parents.0.is_some());
        assert!(c.block().parents.1.is_some());
        assert_eq!(
            Arc::as_ptr(c.block().parents.0.as_ref().unwrap()),
            Arc::as_ptr(&a)
        );

        assert!(c.block().op == Op::Add);

        assert!(d.block().parents.0.is_some());
        assert!(d.block().parents.1.is_some());
        assert_eq!(
            Arc::as_ptr(d.block().parents.0.as_ref().unwrap()),
            Arc::as_ptr(&c)
        );

        assert!(d.block().op == Op::Add);
    }
}
