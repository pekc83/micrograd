use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{init, Linear, Module, Optimizer, VarBuilder, VarMap, SGD};

use micrograd::*;

fn linear(in_dim: usize, out_dim: usize, vb: VarBuilder, init_const: f64) -> Result<Linear> {
    let init_ws = init::Init::Const(init_const);
    let ws = vb.get_with_hints((out_dim, in_dim), "weight", init_ws)?;
    let init_bs = init::Init::Const(init_const);
    let bs = vb.get_with_hints(out_dim, "bias", init_bs)?;
    Ok(Linear::new(ws, Some(bs)))
}

struct CandleMlp {
    first: Linear,
    second: Linear,
    third: Linear,
}

impl CandleMlp {
    pub fn new(var_map: &VarMap, init_const: f64, device: &Device) -> Result<Self> {
        let vb = VarBuilder::from_varmap(var_map, DType::F64, device);
        let first = linear(3, 4, vb.pp("layer1"), init_const)?;
        let second = linear(4, 4, vb.pp("layer2"), init_const)?;
        let third = linear(4, 1, vb.pp("layer3"), init_const)?;

        Ok(Self {
            first,
            second,
            third,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.first.forward(xs)?.tanh()?;
        let xs = self.second.forward(&xs)?.tanh()?;
        self.third.forward(&xs)?.tanh()
    }
}

fn run_candle(
    xs: [[f64; 3]; 4],
    ys_gt: [f64; 4],
    learning_rate: f64,
    init_const: f64,
    cycles: usize,
) -> Result<Vec<f64>> {
    let device = &Device::Cpu;
    let varmap = VarMap::new();
    let model = CandleMlp::new(&varmap, init_const, device)?;

    let mut sgd = SGD::new(varmap.all_vars(), learning_rate)?;

    let xs = Tensor::from_vec(xs.into_iter().flatten().collect(), (4, 3), device)?;
    let out_gt = Tensor::from_vec(ys_gt.to_vec(), (4,), device)?;

    let mut out_sc: Vec<f64> = vec![];

    for _ in 0..cycles {
        let out = model.forward(&xs)?.squeeze(1)?;
        out_sc = out.flatten_all()?.to_vec1()?;

        let loss = (&out - &out_gt)?.sqr()?.sum(0)?;
        sgd.backward_step(&loss)?;
    }

    Ok(out_sc)
}

fn run_micrograd(
    xs: [[f64; 3]; 4],
    ys_gt: [f64; 4],
    learning_rate: f64,
    init_const: f64,
    cycles: usize,
) -> Result<Vec<f64>> {
    let layer1 = Layer::new(3, 4, ValueInit::Const(init_const));
    let layer2 = Layer::new(4, 4, ValueInit::Const(init_const));
    let layer3 = Layer::new(4, 1, ValueInit::Const(init_const));
    let mlp: Mlp = Mlp::new(vec![layer1, layer2, layer3]);
    let inputs = xs.map(|row| row.map(Value::new));
    let outputs = ys_gt.map(Value::new);

    let mut out_sc: Vec<f64> = vec![];

    for _ in 0..cycles {
        let res = inputs
            .iter()
            .map(|x| mlp.call(x).into_iter().next().unwrap())
            .collect::<Vec<_>>();
        out_sc = res.iter().map(|v| v.borrow().data).collect();
        let loss: Value = outputs
            .iter()
            .zip(res.iter())
            .map(|(ygt, yout)| (yout - ygt).pow(2.0))
            .sum();

        mlp.zero_grad();

        loss.backward();

        for parameter in mlp.parameters() {
            let change = parameter.borrow().grad * (-learning_rate);
            parameter.borrow_mut().data += change
        }
    }

    Ok(out_sc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candle_compare() -> Result<()> {
        let xs = [
            [1.0, 1.9, -1.5],
            [3.0, -1.0, 0.5],
            [0.5, 1.0, 1.0],
            [1.0, 1.0, -1.0],
        ];
        let ys_gt = [1.0, -1.0, -1.0, 1.0];

        let learning_rate = 0.1;
        let init_const = 0.2;
        let cycles = 10;

        let candle_result = run_candle(xs, ys_gt, learning_rate, init_const, cycles)?;
        let micrograd_result = run_micrograd(xs, ys_gt, learning_rate, init_const, cycles)?;

        for (a, b) in candle_result.iter().zip(micrograd_result.iter()) {
            let diff = (a - b).abs();
            assert!(diff < f64::EPSILON);
        }

        Ok(())
    }
}
