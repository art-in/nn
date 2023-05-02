use crate::val::{BVal, Val};

use super::Op;

fn backward(child: &BVal) {
    let child = child.borrow();
    let parent = child.parents.0.as_ref().unwrap();

    parent.borrow_mut().grad += (1.0 - child.d.powf(2.0)) * child.grad;
}

impl BVal {
    pub fn tanh(&self) -> Self {
        let e = std::f64::consts::E.powf(2.0 * self.borrow().d);
        let d = (e - 1.0) / (e + 1.0);

        self.pool().pull_val(Val {
            d,
            parents: (Some(self.clone()), None),
            op: Op::Tanh,
            grad: 0.0,
            backward,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::{ops::Op, pool::BValPool};

    #[test]
    fn forward() {
        let pool = BValPool::default();

        let a = pool.pull(1.5);
        let b = a.tanh();

        assert_eq!(b.borrow().d, 0.9051482536448664);
        assert_eq!(b.borrow().op, Op::Tanh);
    }

    #[test]
    fn parents() {
        let pool = BValPool::default();

        let a = pool.pull(1.5);
        let b = a.tanh();

        assert!(a.borrow().parents.0.is_none());
        assert!(a.borrow().parents.1.is_none());

        assert_eq!(a.borrow().op, Op::None);

        assert!(b.borrow().parents.0.is_some());
        assert!(b.borrow().parents.1.is_none());
        assert_eq!(b.borrow().op, Op::Tanh);
    }

    #[test]
    fn backward() {
        let pool = BValPool::default();

        let a = pool.pull(1.5);
        let b = a.tanh();

        b.borrow_mut().grad = 5.0;
        b.backward();

        assert_eq!(a.borrow().grad, 0.9035331946182429);
        assert_eq!(b.borrow().grad, 5.0);
    }
}
