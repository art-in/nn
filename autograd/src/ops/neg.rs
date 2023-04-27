use std::ops::Neg;

use crate::val::BVal;

impl Neg for &BVal {
    type Output = BVal;

    fn neg(self) -> Self::Output {
        self * -1.0
    }
}

#[cfg(test)]
mod tests {
    use crate::ops::Op;

    use super::*;

    #[test]
    fn forward() {
        let a = BVal::new(1.5);
        let b = -&a;

        assert!(b == BVal::new(-1.5));
        assert_eq!(b.borrow().op, Op::Mul);
    }

    #[test]
    fn parents() {
        let a = BVal::new(1.5);
        let b = -&a;

        assert!(a.borrow().parents.is_empty());
        assert!(a.borrow().op == Op::None);

        assert!(b.borrow().parents.len() == 2);
        assert!(b.borrow().parents[0] == a);
        assert!(b.borrow().parents[0].as_ptr() == a.as_ptr());
        assert!(b.borrow().parents[1] == BVal::new(-1.0));

        assert!(b.borrow().op == Op::Mul);
    }

    #[test]
    fn backward() {
        let a = BVal::new(1.5);
        let b = -&a;

        b.borrow_mut().grad = 5.0;
        b.backward();

        assert!(a.borrow().grad == -5.0);
        assert!(b.borrow().grad == 5.0);
    }
}
