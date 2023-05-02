use std::{cell::RefCell, ops::Deref, rc::Rc};

use crate::val::{BVal, Val};

impl Drop for BVal {
    fn drop(&mut self) {
        // if this is the last reference to underlying val and that reference is not from pool
        if (Rc::strong_count(&self.0) == 1) && (!self.pool().borrow().is_dropping) {
            // drop parent vals, so entire downstream graph will go back to pool
            self.borrow_mut().parents.0 = None;
            self.borrow_mut().parents.1 = None;

            // return to pool
            self.pool().push(self.clone())
        }
    }
}

#[derive(Default)]
pub struct Pool {
    pub free: Vec<BVal>,
    pub is_dropping: bool,
}

#[derive(Clone, Default)]
pub struct BValPool(Rc<RefCell<Pool>>);

impl BValPool {
    pub fn pull(&self, d: f64) -> BVal {
        self.pull_val(Val::new(d))
    }

    pub fn pull_val(&self, data: Val) -> BVal {
        if let Some(val) = self.0.borrow_mut().free.pop() {
            *val.borrow_mut() = data;
            val
        } else {
            BVal::new_val(data, self.clone())
        }
    }

    pub fn push(&self, item: BVal) {
        self.0.borrow_mut().free.push(item);
    }
}

impl Deref for BValPool {
    type Target = Rc<RefCell<Pool>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Drop for BValPool {
    fn drop(&mut self) {
        if Rc::strong_count(&self.0) == 1 {
            self.borrow_mut().is_dropping = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::Op;

    #[test]
    fn pull_val() {
        let pool = BValPool::default();

        // construct new val and drop it back to pool
        let val_a_addr = pool
            .pull_val(Val {
                d: 1.0,
                op: Op::Tanh,
                // this two val`s will be constructed and dropped first,
                // so val_a will stay on top of pool
                parents: (Some(pool.pull(9.9)), Some(pool.pull(9.9))),
                grad: 9.9,
                backward: |_| {},
            })
            .deref()
            .as_ptr(); // implicit drop

        // pull next val and make sure it has requested value and everything else cleared
        let val_b = pool.pull_val(Val::new(2.0));

        assert_eq!(val_b.borrow().d, 2.0);
        assert_eq!(val_b.borrow().op, Op::None);
        assert_eq!(val_b.borrow().parents.0, None);
        assert_eq!(val_b.borrow().parents.1, None);
        assert_eq!(val_b.borrow().grad, 0.0);

        // but it actually utilizes same object as previous val
        assert_eq!(val_b.deref().as_ptr(), val_a_addr);
    }

    #[test]
    fn drop() {
        let pool = BValPool::default();

        assert_eq!(pool.0.borrow().free.len(), 0);

        {
            let a = pool.pull(3.0);
            let b = pool.pull(2.0);
            let c = &a + &b;
            let d = pool.pull(5.0);
            assert_eq!(c.borrow().d, d.borrow().d);
        }

        assert_eq!(pool.0.borrow().free.len(), 4);
    }
}
