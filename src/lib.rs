//! <div class="warning">This is a very crappy solution, but no one in their right mind would ever use this library in production, so I'm ok with it!</div>
use rand::random;
use std::cell::RefCell;
use std::collections::BTreeSet;
use std::fmt::{Display, Formatter};
use std::iter::{once, Sum};
use std::mem;
use std::ops::{Add, Deref, Mul, Sub};
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

static COUNTER: AtomicUsize = AtomicUsize::new(0);
macro_rules! id {
    () => {
        COUNTER.fetch_add(1, Ordering::Relaxed)
    };
}

type DType = f64;

pub type InnerValueRef = Rc<RefCell<InnerValue>>;

enum ValueOp {
    None,
    Add(InnerValueRef, InnerValueRef),
    Sub(InnerValueRef, InnerValueRef),
    Mul(InnerValueRef, InnerValueRef),
    Pow(InnerValueRef, DType),
    Tanh(InnerValueRef),
}

pub enum ValueInit {
    Const(DType),
    Uniform { lo: DType, up: DType },
}

pub struct InnerValue {
    pub data: DType,
    pub grad: DType,
    op: ValueOp,
    id: usize,
}

impl InnerValue {
    pub fn backward(&self) {
        match &self.op {
            ValueOp::None => {}
            ValueOp::Add(child1, child2) => {
                child1.borrow_mut().grad += self.grad * 1.0;
                child2.borrow_mut().grad += self.grad * 1.0;
            }
            ValueOp::Mul(child1, child2) => {
                child1.borrow_mut().grad += self.grad * child2.borrow().data;
                child2.borrow_mut().grad += self.grad * child1.borrow().data;
            }
            ValueOp::Tanh(child) => {
                let tanh_grad = 1.0 - child.borrow().data.tanh().powi(2);
                child.borrow_mut().grad += self.grad * tanh_grad;
            }
            ValueOp::Pow(child, n) => {
                let pow_grad = n * child.borrow().data.powf(n - 1.0);
                child.borrow_mut().grad += self.grad * pow_grad;
            }
            ValueOp::Sub(child1, child2) => {
                child1.borrow_mut().grad += self.grad * 1.0;
                child2.borrow_mut().grad += self.grad * 1.0;
            }
        }
    }
}

impl Eq for InnerValue {}

impl PartialEq<Self> for InnerValue {
    fn eq(&self, other: &Self) -> bool {
        self.id.eq(&other.id)
    }
}

impl PartialOrd<Self> for InnerValue {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for InnerValue {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.id.cmp(&other.id)
    }
}

#[derive(Clone)]
pub struct Value(InnerValueRef);

impl Deref for Value {
    type Target = InnerValueRef;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Value {
    pub fn init(init: &ValueInit) -> Self {
        match init {
            ValueInit::Const(v) => Self::new(*v),
            ValueInit::Uniform { lo, up } => Self::new((up - lo) * random::<DType>() + lo),
        }
    }

    pub fn new(data: DType) -> Self {
        Self(Rc::new(RefCell::new(InnerValue {
            data,
            op: ValueOp::None,
            grad: 0.0,
            id: id!(),
        })))
    }

    pub fn tanh(&self) -> Self {
        Self(Rc::new(RefCell::new(InnerValue {
            data: self.borrow().data.tanh(),
            op: ValueOp::Tanh(self.deref().clone()),
            grad: 0.0,
            id: id!(),
        })))
    }

    pub fn pow(&self, n: DType) -> Self {
        Self(Rc::new(RefCell::new(InnerValue {
            data: self.borrow().data.powf(n),
            op: ValueOp::Pow(self.deref().clone(), n),
            grad: 0.0,
            id: id!(),
        })))
    }

    pub fn backward(&self) {
        fn topo(v: &InnerValueRef, set: &mut BTreeSet<InnerValueRef>) {
            set.insert(v.clone());
            match &v.borrow().op {
                ValueOp::None => {}
                ValueOp::Add(c1, c2) => {
                    topo(c1, set);
                    topo(c2, set);
                }
                ValueOp::Sub(c1, c2) => {
                    topo(c1, set);
                    topo(c2, set);
                }
                ValueOp::Mul(c1, c2) => {
                    topo(c1, set);
                    topo(c2, set);
                }
                ValueOp::Tanh(c) => topo(c, set),
                ValueOp::Pow(c, _) => topo(c, set),
            }
        }

        self.borrow_mut().grad = 1.0;
        let mut set = BTreeSet::new();
        topo(self, &mut set);

        for child in set.iter().rev() {
            child.borrow().backward()
        }
    }
}

impl Add<&Value> for &Value {
    type Output = Value;

    fn add(self, rhs: &Value) -> Self::Output {
        Value(Rc::new(RefCell::new(InnerValue {
            data: self.borrow().data + rhs.borrow().data,
            op: ValueOp::Add(self.deref().clone(), rhs.deref().clone()),
            grad: 0.0,
            id: id!(),
        })))
    }
}

impl Add<DType> for &Value {
    type Output = Value;

    fn add(self, rhs: DType) -> Self::Output {
        self + &Value::new(rhs)
    }
}

impl Sub<&Value> for &Value {
    type Output = Value;

    fn sub(self, rhs: &Value) -> Self::Output {
        Value(Rc::new(RefCell::new(InnerValue {
            data: self.borrow().data - rhs.borrow().data,
            op: ValueOp::Sub(self.deref().clone(), rhs.deref().clone()),
            grad: 0.0,
            id: id!(),
        })))
    }
}

impl Mul<&Value> for &Value {
    type Output = Value;

    fn mul(self, rhs: &Value) -> Self::Output {
        Value(Rc::new(RefCell::new(InnerValue {
            data: self.borrow().data * rhs.borrow().data,
            op: ValueOp::Mul(self.deref().clone(), rhs.deref().clone()),
            grad: 0.0,
            id: id!(),
        })))
    }
}

impl Mul<DType> for &Value {
    type Output = Value;

    fn mul(self, rhs: DType) -> Self::Output {
        self * &Value::new(rhs)
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Value(data: {:.4}, grad: {:.4})",
            self.borrow().data,
            self.borrow().grad
        )
    }
}

impl Sum for Value {
    fn sum<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        let first = iter.next();
        if let Some(first) = first {
            iter.fold(first, |accum, x| &accum + &x)
        } else {
            Value::new(0.0)
        }
    }
}

struct Neuron {
    weights: Vec<Value>,
    b: Value,
}

impl Neuron {
    pub fn new(size: usize, value_init: &ValueInit) -> Self {
        let mut weights: Vec<Value> = Vec::with_capacity(size);
        for _ in 0..size {
            weights.push(Value::init(value_init))
        }

        Self {
            weights,
            b: Value::init(value_init),
        }
    }

    pub fn parameters(&self) -> impl Iterator<Item = &Value> {
        self.weights.iter().chain(once(&self.b))
    }

    pub fn call(&self, inputs: &[Value]) -> Value {
        assert_eq!(self.weights.len(), inputs.len());
        self.weights
            .iter()
            .zip(inputs)
            .map(|(w, i)| w * i)
            .fold(self.b.clone(), |accum, x| &accum + &x)
            .tanh()
    }
}

pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(nin: usize, nout: usize, value_init: ValueInit) -> Self {
        let mut neurons: Vec<Neuron> = Vec::with_capacity(nout);
        for _ in 0..nout {
            neurons.push(Neuron::new(nin, &value_init));
        }
        Self { neurons }
    }

    pub fn call(&self, inputs: &[Value]) -> Vec<Value> {
        self.neurons
            .iter()
            .map(|neuron| neuron.call(inputs))
            .collect()
    }

    pub fn parameters(&self) -> impl Iterator<Item = &Value> {
        self.neurons.iter().flat_map(|neuron| neuron.parameters())
    }
}

pub struct Mlp {
    layers: Vec<Layer>,
}

impl Mlp {
    pub fn new(layers: Vec<Layer>) -> Self {
        Self { layers }
    }

    pub fn parameters(&self) -> impl Iterator<Item = &Value> {
        self.layers.iter().flat_map(|layer| layer.parameters())
    }

    pub fn zero_grad(&self) {
        for parameter in self.parameters() {
            parameter.borrow_mut().grad = 0.0
        }
    }

    pub fn call(&self, inputs: &[Value]) -> Vec<Value> {
        let mut inputs = Vec::from(inputs);
        for layer in &self.layers {
            let mut res = layer.call(&inputs);
            mem::swap(&mut res, &mut inputs);
        }
        inputs
    }
}
