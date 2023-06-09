mod add;
mod div;
mod mul;
mod neg;
mod pow;
mod sub;
mod tanh;

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Op {
    None,
    Add,
    Mul,
    Pow,
    Tanh,
}
