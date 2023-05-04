use core::fmt;
use std::{
    cell::RefCell,
    cmp::Ordering,
    collections::HashSet,
    fmt::{Debug, Display},
    ops::Deref,
    rc::Rc,
};

use crate::ops::Op;

type BackwardFn = fn(&BVal) -> ();

pub struct Val {
    pub d: f64,
    pub op: Op,
    pub parents: (Option<BVal>, Option<BVal>),
    pub grad: f64,
    pub backward: BackwardFn,
}

impl Val {
    fn new(d: f64) -> Self {
        Val {
            d,
            parents: (None, None),
            op: Op::None,
            grad: 0.0,
            backward: |_| (),
        }
    }
}

impl PartialEq for Val {
    fn eq(&self, other: &Self) -> bool {
        self.d == other.d
    }
}

impl Eq for Val {}

impl PartialOrd for Val {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Val {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.d == other.d {
            Ordering::Equal
        } else if self.d < other.d {
            Ordering::Less
        } else {
            Ordering::Greater
        }
    }
}

#[derive(PartialEq, Clone)]
pub struct BVal(pub Rc<RefCell<Val>>);

impl BVal {
    pub fn new(d: f64) -> Self {
        BVal(Rc::new(RefCell::new(Val::new(d))))
    }

    pub fn new_val(val: Val) -> Self {
        BVal(Rc::new(RefCell::new(val)))
    }

    pub fn backward(&self) {
        let mut topo: Vec<BVal> = Vec::new();
        let mut visited: HashSet<*mut Val> = HashSet::new();

        fn build_topo(node: BVal, visited: &mut HashSet<*mut Val>, topo: &mut Vec<BVal>) {
            if !visited.contains(&node.as_ptr()) {
                visited.insert(node.as_ptr());

                let node_ref = node.borrow();
                let parents = [node_ref.parents.0.as_ref(), node_ref.parents.1.as_ref()];

                for parent in parents.into_iter().flatten() {
                    build_topo(parent.clone(), visited, topo);
                }

                topo.push(node.clone());
            }
        }

        build_topo(self.clone(), &mut visited, &mut topo);

        for node in topo.iter().rev() {
            (node.borrow().backward)(node);
        }
    }
}

impl Deref for BVal {
    type Target = Rc<RefCell<Val>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Debug for BVal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // implementing manually to omit func field "backward"
        f.debug_struct("")
            .field("d", &self.borrow().d)
            .field("grad", &self.borrow().grad)
            .field("op", &self.borrow().op)
            .field("parents", &self.borrow().parents)
            .finish()
    }
}

impl Display for BVal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let tmp = self.0.as_ref().borrow();
        write!(f, "{}", tmp.d)
    }
}

#[cfg(test)]
mod tests {
    use float_cmp::assert_approx_eq;

    use super::*;

    // https://youtu.be/VMj-3S1tku0?t=5877
    #[test]
    fn backward() {
        let x1 = BVal::new(2.0);
        let x2 = BVal::new(0.0);
        let w1 = BVal::new(-3.0);
        let w2 = BVal::new(1.0);
        let b = BVal::new(6.8813735870195432);

        let x1w1 = &x1 * &w1;
        let x2w2 = &x2 * &w2;
        let x1w1x2w2 = &x1w1 + &x2w2;
        let n = &x1w1x2w2 + &b;
        let o = n.tanh();

        assert_eq!(o.borrow().d, 0.7071067811865476);

        o.borrow_mut().grad = 1.0;
        o.backward();

        assert_approx_eq!(f64, x1.borrow().grad, -1.5);
        assert_approx_eq!(f64, x2.borrow().grad, 0.5);
        assert_approx_eq!(f64, w1.borrow().grad, 1.0);
        assert_approx_eq!(f64, w2.borrow().grad, 0.0);
    }
}
