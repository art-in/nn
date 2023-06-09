use std::ops::Mul;

use crate::val::{BVal, Val};

use super::Op;

fn backward(child: &BVal) {
    let child = child.borrow();

    let lhs = child.parents.0.as_ref().unwrap();
    let rhs = child.parents.1.as_ref().unwrap();

    lhs.borrow_mut().grad += rhs.borrow().d * child.grad;
    rhs.borrow_mut().grad += lhs.borrow().d * child.grad;
}

impl Mul<&BVal> for &BVal {
    type Output = BVal;

    fn mul(self, other: &BVal) -> Self::Output {
        BVal::new_val(Val {
            d: self.borrow().d * other.borrow().d,
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
        assert!(d.borrow().op == Op::Mul);
    }

    #[test]
    fn parents_simple() {
        let a = BVal::new(1.5);
        let b = BVal::new(2.0);
        let c = &a * &b;

        assert!(a.borrow().parents.0.is_none());
        assert!(a.borrow().parents.1.is_none());

        assert!(b.borrow().parents.0.is_none());
        assert!(b.borrow().parents.1.is_none());

        assert!(a.borrow().op == Op::None);
        assert!(b.borrow().op == Op::None);

        assert!(c.borrow().parents.0.as_ref().unwrap() == &a);
        assert!(c.borrow().parents.1.as_ref().unwrap() == &b);

        assert!(c.borrow().op == Op::Mul);
    }

    #[test]
    fn parents_complex() {
        let a = BVal::new(1.5);
        let b = BVal::new(2.0);
        let c = &a * &b;
        let d = &c * &a;

        assert!(a.borrow().parents.0.is_none());
        assert!(a.borrow().parents.1.is_none());

        assert!(b.borrow().parents.0.is_none());
        assert!(b.borrow().parents.1.is_none());

        assert!(a.borrow().op == Op::None);
        assert!(b.borrow().op == Op::None);

        assert!(c.borrow().parents.0.as_ref().unwrap().as_ptr() == a.as_ptr());
        assert!(c.borrow().parents.1.as_ref().unwrap().as_ptr() == b.as_ptr());

        assert!(c.borrow().op == Op::Mul);

        assert!(d.borrow().parents.0.as_ref().unwrap().as_ptr() == c.as_ptr());
        assert!(d.borrow().parents.1.as_ref().unwrap().as_ptr() == a.as_ptr());

        assert!(d.borrow().op == Op::Mul);
    }

    #[test]
    fn backward() {
        let a = BVal::new(2.0);
        let b = BVal::new(-3.0);
        let c = &a * &b;

        c.borrow_mut().grad = 5.0;
        c.backward();

        assert!(a.borrow().grad == -15.0);
        assert!(b.borrow().grad == 10.0);
        assert!(c.borrow().grad == 5.0);
    }
}
