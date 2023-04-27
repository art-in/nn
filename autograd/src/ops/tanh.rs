use crate::val::BVal;

impl BVal {
    pub fn tanh(&self) -> Self {
        let e = BVal::new(std::f64::consts::E).pow_val(&(2.0 * self));
        &(&e - 1.0) / &(&e + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use crate::ops::Op;

    use super::*;

    #[test]
    fn forward() {
        let a = BVal::new(1.5);
        let b = a.tanh();

        assert_eq!(b.borrow().d, 0.9051482536448664);
        assert_eq!(b.borrow().op, Op::Mul);
    }

    #[test]
    fn parents() {
        let a = BVal::new(1.5);
        let b = a.tanh();

        assert!(a.borrow().parents.is_empty());
        assert!(a.borrow().op == Op::None);

        assert!(b.borrow().parents.len() == 2);
        assert!(b.borrow().op == Op::Mul);
    }

    #[test]
    fn backward() {
        let a = BVal::new(1.5);
        let b = a.tanh();

        b.borrow_mut().grad = 5.0;
        b.backward();

        assert_eq!(a.borrow().grad, 0.903533194618244);
        assert_eq!(b.borrow().grad, 5.0);
    }
}
