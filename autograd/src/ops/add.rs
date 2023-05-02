use std::ops::Add;

use crate::val::{BVal, Val};

use super::Op;

fn backward(child: &BVal) {
    let child = child.borrow();

    let lhs = child.parents.0.as_ref().unwrap();
    let rhs = child.parents.1.as_ref().unwrap();

    lhs.borrow_mut().grad += child.grad;
    rhs.borrow_mut().grad += child.grad;
}

impl Add<&BVal> for &BVal {
    type Output = BVal;

    fn add(self, other: &BVal) -> Self::Output {
        other.pool().pull_val(Val {
            d: self.borrow().d + other.borrow().d,
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
        self + &self.pool().pull(other)
    }
}

impl Add<&BVal> for f64 {
    type Output = BVal;

    fn add(self, other: &BVal) -> Self::Output {
        &other.pool().pull(self) + other
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pool::BValPool;

    #[test]
    fn forward_simple() {
        let pool = BValPool::default();

        assert!(&pool.pull(1.0) + &pool.pull(2.0) == pool.pull(3.0));
        assert!(&pool.pull(1.0) + 2.0 == pool.pull(3.0));
        assert!(1.0 + &pool.pull(2.0) == pool.pull(3.0));
    }

    #[test]
    fn forward_complex() {
        let pool = BValPool::default();

        let a = pool.pull(1.0);
        let b = pool.pull(2.0);
        let c = &a + &b;
        let d = &c + &a;

        assert_eq!(d, pool.pull(4.0));
        assert_eq!(d.borrow().op, Op::Add);
    }

    #[test]
    fn parents_simple() {
        let pool = BValPool::default();

        let a = pool.pull(1.0);
        let b = pool.pull(2.0);
        let c = &a + &b;

        assert!(a.borrow().parents.0.is_none());
        assert!(a.borrow().parents.1.is_none());

        assert!(b.borrow().parents.0.is_none());
        assert!(b.borrow().parents.1.is_none());

        assert_eq!(a.borrow().op, Op::None);
        assert_eq!(b.borrow().op, Op::None);

        assert_eq!(c.borrow().parents.0.as_ref().unwrap(), &a);
        assert_eq!(c.borrow().parents.1.as_ref().unwrap(), &b);

        assert_eq!(c.borrow().op, Op::Add);
    }

    #[test]
    fn parents_complex() {
        let pool = BValPool::default();

        let a = pool.pull(1.0);
        let b = pool.pull(2.0);
        let c = &a + &b;
        let d = &c + &a;

        assert!(a.borrow().parents.0.is_none());
        assert!(a.borrow().parents.1.is_none());

        assert!(b.borrow().parents.0.is_none());
        assert!(b.borrow().parents.1.is_none());

        assert_eq!(a.borrow().op, Op::None);
        assert_eq!(b.borrow().op, Op::None);

        assert_eq!(c.borrow().parents.0.as_ref().unwrap().as_ptr(), a.as_ptr());
        assert_eq!(c.borrow().parents.1.as_ref().unwrap().as_ptr(), b.as_ptr());

        assert_eq!(c.borrow().op, Op::Add);

        assert_eq!(d.borrow().parents.0.as_ref().unwrap().as_ptr(), c.as_ptr());
        assert_eq!(d.borrow().parents.1.as_ref().unwrap().as_ptr(), a.as_ptr());

        assert_eq!(d.borrow().op, Op::Add);
    }

    #[test]
    fn backward() {
        let pool = BValPool::default();

        let a = pool.pull(1.0);
        let b = pool.pull(2.0);
        let c = &a + &b;

        c.borrow_mut().grad = 5.0;
        c.backward();

        assert_eq!(a.borrow().grad, 5.0);
        assert_eq!(b.borrow().grad, 5.0);
        assert_eq!(c.borrow().grad, 5.0);
    }
}
