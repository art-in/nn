use std::ops::Add;

use crate::val::{BVal, Val};

use super::Op;

fn backward(child: &BVal) {
    let lhs = &child.borrow().parents[0];
    let rhs = &child.borrow().parents[1];

    lhs.borrow_mut().grad += child.borrow().grad;
    rhs.borrow_mut().grad += child.borrow().grad;
}

impl Add<&BVal> for &BVal {
    type Output = BVal;

    fn add(self, other: &BVal) -> Self::Output {
        BVal::new_val(Val {
            d: self.borrow().d + other.borrow().d,
            parents: vec![self.clone(), other.clone()],
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
        assert!(d.borrow().op == Op::Add);
    }

    #[test]
    fn parents_simple() {
        let a = BVal::new(1.0);
        let b = BVal::new(2.0);
        let c = &a + &b;

        assert!(a.borrow().parents.is_empty());
        assert!(b.borrow().parents.is_empty());

        assert!(a.borrow().op == Op::None);
        assert!(b.borrow().op == Op::None);

        assert!(c.borrow().parents.len() == 2);
        assert!(c.borrow().parents[0] == a);
        assert!(c.borrow().parents[1] == b);

        assert!(c.borrow().op == Op::Add);
    }

    #[test]
    fn parents_complex() {
        let a = BVal::new(1.0);
        let b = BVal::new(2.0);
        let c = &a + &b;
        let d = &c + &a;

        assert!(a.borrow().parents.is_empty());
        assert!(b.borrow().parents.is_empty());

        assert!(a.borrow().op == Op::None);
        assert!(b.borrow().op == Op::None);

        assert!(c.borrow().parents.len() == 2);
        assert!(c.borrow().parents[0].as_ptr() == a.as_ptr());
        assert!(c.borrow().parents[1].as_ptr() == b.as_ptr());

        assert!(c.borrow().op == Op::Add);

        assert!(d.borrow().parents.len() == 2);
        assert!(d.borrow().parents[0].as_ptr() == c.as_ptr());
        assert!(d.borrow().parents[1].as_ptr() == a.as_ptr());

        assert!(d.borrow().op == Op::Add);
    }

    #[test]
    fn backward() {
        let a = BVal::new(1.0);
        let b = BVal::new(2.0);
        let c = &a + &b;

        c.borrow_mut().grad = 5.0;
        c.backward();

        assert!(a.borrow().grad == 5.0);
        assert!(b.borrow().grad == 5.0);
        assert!(c.borrow().grad == 5.0);
    }
}
