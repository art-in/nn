use core::fmt;
use std::{
    cmp::Ordering,
    collections::HashSet,
    fmt::{Debug, Display},
    ops::Deref,
    sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard},
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

#[derive(Clone)]
pub struct BVal(pub Arc<RwLock<Val>>);

impl BVal {
    pub fn block(&self) -> RwLockReadGuard<'_, Val> {
        self.read().unwrap()
    }

    pub fn block_mut(&self) -> RwLockWriteGuard<'_, Val> {
        self.write().unwrap()
    }

    pub fn new(d: f64) -> Self {
        Self::new_val(Val::new(d))
    }

    pub fn new_val(val: Val) -> Self {
        BVal(Arc::new(RwLock::new(val)))
    }

    pub fn backward(&self) {
        let mut topo: Vec<BVal> = Vec::new();
        let mut visited: HashSet<*const RwLock<Val>> = HashSet::new();

        fn build_topo(node: BVal, visited: &mut HashSet<*const RwLock<Val>>, topo: &mut Vec<BVal>) {
            let node_ptr = Arc::as_ptr(node.deref());
            if !visited.contains(&node_ptr) {
                visited.insert(node_ptr);

                let node_ref = node.block();
                let parents = [node_ref.parents.0.as_ref(), node_ref.parents.1.as_ref()];

                for parent in parents {
                    if let Some(parent) = parent {
                        build_topo(parent.clone(), visited, topo);
                    }
                }

                topo.push(node.clone());
            }
        }

        build_topo(self.clone(), &mut visited, &mut topo);

        for node in topo.iter().rev() {
            let backward = node.block().backward;
            (backward)(&node);
        }
    }
}

impl Deref for BVal {
    type Target = Arc<RwLock<Val>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Debug for BVal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // implementing manually to omit function field "backward"
        let val = self.block();
        f.debug_struct("")
            .field("d", &val.d)
            .field("grad", &val.grad)
            .field("op", &val.op)
            .field("parents", &val.parents)
            .finish()
    }
}

impl Display for BVal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.block().d)
    }
}

impl PartialEq for BVal {
    fn eq(&self, other: &Self) -> bool {
        (Arc::as_ptr(self) == Arc::as_ptr(other)) || (self.block().d == other.block().d)
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

        assert_eq!(o.block().d, 0.7071067811865476);

        o.block_mut().grad = 1.0;
        o.backward();

        assert_approx_eq!(f64, x1.block().grad, -1.5);
        assert_approx_eq!(f64, x2.block().grad, 0.5);
        assert_approx_eq!(f64, w1.block().grad, 1.0);
        assert_approx_eq!(f64, w2.block().grad, 0.0);
    }
}
