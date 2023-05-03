use std::ops::Div;

use crate::val::BVal;

impl Div<&BVal> for &BVal {
    type Output = BVal;

    fn div(self, other: &BVal) -> Self::Output {
        self * &other.pow(-1.0)
    }
}

impl Div<f64> for &BVal {
    type Output = BVal;

    fn div(self, other: f64) -> Self::Output {
        self * &BVal::new(other).pow(-1.0)
    }
}

impl Div<&BVal> for f64 {
    type Output = BVal;

    fn div(self, other: &BVal) -> Self::Output {
        &BVal::new(self) * &other.pow(-1.0)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::ops::Op;

    use super::*;

    #[test]
    fn simple() {
        assert!(&BVal::new(3.0) / &BVal::new(2.0) == BVal::new(1.5));
        assert!(&BVal::new(3.0) / 2.0 == BVal::new(1.5));
        assert!(3.0 / &BVal::new(2.0) == BVal::new(1.5));
    }

    #[test]
    fn complex() {
        let a = BVal::new(3.0);
        let b = BVal::new(2.0);
        let c = &a / &b;
        let d = &c / &a;

        assert!(d == BVal::new(0.5));
        assert!(d.block().op == Op::Mul);
    }

    #[test]
    fn parents_simple() {
        let a = BVal::new(3.0);
        let b = BVal::new(2.0);
        let c = &a / &b;

        assert!(a.block().parents.0.is_none());
        assert!(a.block().parents.1.is_none());

        assert!(b.block().parents.0.is_none());
        assert!(b.block().parents.1.is_none());

        assert!(a.block().op == Op::None);
        assert!(b.block().op == Op::None);

        assert!(c.block().parents.0.as_ref().unwrap() == &a);
        assert!(c.block().parents.1.as_ref().unwrap() == &BVal::new(0.5));

        assert!(c.block().op == Op::Mul);
    }

    #[test]
    fn parents_complex() {
        let a = BVal::new(3.0);
        let b = BVal::new(2.0);
        let c = &a / &b;
        let d = &c / &a;

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

        assert!(c.block().op == Op::Mul);

        assert!(d.block().parents.0.is_some());
        assert!(d.block().parents.1.is_some());
        assert_eq!(
            Arc::as_ptr(d.block().parents.0.as_ref().unwrap()),
            Arc::as_ptr(&c)
        );

        assert!(d.block().op == Op::Mul);
    }

    #[test]
    fn backward() {
        let a = BVal::new(3.0);
        let b = BVal::new(2.0);
        let c = &a / &b;

        c.block_mut().grad = 5.0;
        c.backward();

        assert_eq!(a.block().grad, 2.5);
        assert_eq!(b.block().grad, -3.75);
        assert_eq!(c.block().grad, 5.0);
    }
}
