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
        self + &-&self.pool().pull(other)
    }
}

impl Sub<&BVal> for f64 {
    type Output = BVal;

    fn sub(self, other: &BVal) -> Self::Output {
        &other.pool().pull(self) + &-other
    }
}

#[cfg(test)]
mod tests {
    use crate::{ops::Op, pool::BValPool};

    #[test]
    fn simple() {
        let pool = BValPool::default();

        assert!(&pool.pull(3.0) - &pool.pull(2.0) == pool.pull(1.0));
        assert!(&pool.pull(3.0) - 2.0 == pool.pull(1.0));
        assert!(3.0 - &pool.pull(2.0) == pool.pull(1.0));
    }

    #[test]
    fn complex() {
        let pool = BValPool::default();

        let a = pool.pull(3.0);
        let b = pool.pull(2.0);
        let c = &a - &b;
        let d = &c - &a;

        assert_eq!(d, pool.pull(-2.0));
        assert_eq!(d.borrow().op, Op::Add);
    }

    #[test]
    fn parents_simple() {
        let pool = BValPool::default();

        let a = pool.pull(3.0);
        let b = pool.pull(2.0);
        let c = &a - &b;

        assert!(a.borrow().parents.0.is_none());
        assert!(a.borrow().parents.1.is_none());

        assert!(b.borrow().parents.0.is_none());
        assert!(b.borrow().parents.1.is_none());

        assert_eq!(a.borrow().op, Op::None);
        assert_eq!(b.borrow().op, Op::None);

        assert_eq!(c.borrow().parents.0.as_ref().unwrap(), &a);
        assert_eq!(c.borrow().parents.1.as_ref().unwrap(), &pool.pull(-2.0));

        assert_eq!(c.borrow().op, Op::Add);
    }

    #[test]
    fn parents_complex() {
        let pool = BValPool::default();

        let a = pool.pull(3.0);
        let b = pool.pull(2.0);
        let c = &a - &b;
        let d = &c - &a;

        assert!(a.borrow().parents.0.is_none());
        assert!(a.borrow().parents.1.is_none());

        assert!(b.borrow().parents.0.is_none());
        assert!(b.borrow().parents.1.is_none());

        assert_eq!(a.borrow().op, Op::None);
        assert_eq!(b.borrow().op, Op::None);

        assert!(c.borrow().parents.0.is_some());
        assert!(c.borrow().parents.1.is_some());
        assert_eq!(c.borrow().parents.0.as_ref().unwrap().as_ptr(), a.as_ptr());

        assert_eq!(c.borrow().op, Op::Add);

        assert!(d.borrow().parents.0.is_some());
        assert!(d.borrow().parents.1.is_some());
        assert_eq!(d.borrow().parents.0.as_ref().unwrap().as_ptr(), c.as_ptr());

        assert_eq!(d.borrow().op, Op::Add);
    }
}
