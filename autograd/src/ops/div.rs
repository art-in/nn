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
        assert!(d.borrow().op == Op::Mul);
    }

    #[test]
    fn parents_simple() {
        let a = BVal::new(3.0);
        let b = BVal::new(2.0);
        let c = &a / &b;

        assert!(a.borrow().parents.is_empty());
        assert!(b.borrow().parents.is_empty());

        assert!(a.borrow().op == Op::None);
        assert!(b.borrow().op == Op::None);

        assert!(c.borrow().parents.len() == 2);
        assert!(c.borrow().parents[0] == a);
        assert!(c.borrow().parents[1] == BVal::new(0.5));

        assert!(c.borrow().op == Op::Mul);
    }

    #[test]
    fn parents_complex() {
        let a = BVal::new(3.0);
        let b = BVal::new(2.0);
        let c = &a / &b;
        let d = &c / &a;

        assert!(a.borrow().parents.is_empty());
        assert!(b.borrow().parents.is_empty());

        assert!(a.borrow().op == Op::None);
        assert!(b.borrow().op == Op::None);

        assert!(c.borrow().parents.len() == 2);
        assert!(c.borrow().parents[0].as_ptr() == a.as_ptr());

        assert!(c.borrow().op == Op::Mul);

        assert!(d.borrow().parents.len() == 2);
        assert!(d.borrow().parents[0].as_ptr() == c.as_ptr());

        assert!(d.borrow().op == Op::Mul);
    }

    #[test]
    fn backward() {
        let a = BVal::new(3.0);
        let b = BVal::new(2.0);
        let c = &a / &b;

        c.borrow_mut().grad = 5.0;
        c.backward();

        assert_eq!(a.borrow().grad, 2.5);
        assert_eq!(b.borrow().grad, -3.75);
        assert_eq!(c.borrow().grad, 5.0);
    }
}
