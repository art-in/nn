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
        assert!(d.borrow().op == Op::Add);
    }

    #[test]
    fn parents_simple() {
        let a = BVal::new(3.0);
        let b = BVal::new(2.0);
        let c = &a - &b;

        assert!(a.borrow().parents.is_empty());
        assert!(b.borrow().parents.is_empty());

        assert!(a.borrow().op == Op::None);
        assert!(b.borrow().op == Op::None);

        assert!(c.borrow().parents.len() == 2);
        assert!(c.borrow().parents[0] == a);
        assert!(c.borrow().parents[1] == BVal::new(-2.0));

        assert!(c.borrow().op == Op::Add);
    }

    #[test]
    fn parents_complex() {
        let a = BVal::new(3.0);
        let b = BVal::new(2.0);
        let c = &a - &b;
        let d = &c - &a;

        assert!(a.borrow().parents.is_empty());
        assert!(b.borrow().parents.is_empty());

        assert!(a.borrow().op == Op::None);
        assert!(b.borrow().op == Op::None);

        assert!(c.borrow().parents.len() == 2);
        assert!(c.borrow().parents[0].as_ptr() == a.as_ptr());

        assert!(c.borrow().op == Op::Add);

        assert!(d.borrow().parents.len() == 2);
        assert!(d.borrow().parents[0].as_ptr() == c.as_ptr());

        assert!(d.borrow().op == Op::Add);
    }
}
