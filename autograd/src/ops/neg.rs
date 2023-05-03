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
    use std::sync::Arc;

    use crate::ops::Op;

    use super::*;

    #[test]
    fn forward() {
        let a = BVal::new(1.5);
        let b = -&a;

        assert!(b == BVal::new(-1.5));
        assert_eq!(b.block().op, Op::Mul);
    }

    #[test]
    fn parents() {
        let a = BVal::new(1.5);
        let b = -&a;

        assert!(a.block().parents.0.is_none());
        assert!(a.block().parents.1.is_none());
        assert!(a.block().op == Op::None);

        assert!(b.block().parents.0.as_ref().unwrap() == &a);
        assert_eq!(
            Arc::as_ptr(b.block().parents.0.as_ref().unwrap()),
            Arc::as_ptr(&a)
        );
        assert!(b.block().parents.1.as_ref().unwrap() == &BVal::new(-1.0));

        assert!(b.block().op == Op::Mul);
    }

    #[test]
    fn backward() {
        let a = BVal::new(1.5);
        let b = -&a;

        b.block_mut().grad = 5.0;
        b.backward();

        assert!(a.block().grad == -5.0);
        assert!(b.block().grad == 5.0);
    }
}
