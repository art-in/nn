use crate::val::{BVal, Val};

use super::Op;

fn backward(child: &BVal) {
    let child = child.borrow();

    let lhs = child.parents.0.as_ref().unwrap();
    let rhs = child.parents.1.as_ref().unwrap();

    let base = lhs.borrow().d;
    let degree = rhs.borrow().d;

    lhs.borrow_mut().grad += degree * base.powf(degree - 1.0) * child.grad;
    rhs.borrow_mut().grad += child.d * child.grad;
}

impl BVal {
    pub fn pow(&self, degree: f64) -> BVal {
        self.pow_val(&BVal::new(degree))
    }

    pub fn pow_val(&self, degree: &BVal) -> BVal {
        BVal::new_val(Val {
            d: self.borrow().d.powf(degree.borrow().d),
            parents: (Some(self.clone()), Some(degree.clone())),
            op: Op::Pow,
            grad: 0.0,
            backward,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward() {
        let a = BVal::new(2.0);
        let b = a.pow(3.0);

        assert!(b == BVal::new(8.0));
        assert!(b.borrow().op == Op::Pow);
    }

    #[test]
    fn parents() {
        let a = BVal::new(2.0);
        let b = a.pow(3.0);

        assert!(a.borrow().parents.0.is_none());
        assert!(a.borrow().parents.1.is_none());

        assert!(a.borrow().op == Op::None);

        assert!(b.borrow().parents.0.as_ref().unwrap() == &a);
        assert!(b.borrow().parents.1.as_ref().unwrap() == &BVal::new(3.0));

        assert!(b.borrow().parents.0.as_ref().unwrap().as_ptr() == a.as_ptr());

        assert!(b.borrow().op == Op::Pow);
    }

    #[test]
    fn backward() {
        let a = BVal::new(2.0);
        let b = a.pow(3.0);

        b.borrow_mut().grad = 2.0;
        b.backward();

        assert_eq!(a.borrow().grad, 24.0);
        assert!(b.borrow().grad == 2.0);
    }
}
