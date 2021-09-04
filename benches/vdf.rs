use criterion::{criterion_group, criterion_main, Criterion};
use pasta_curves::arithmetic::FieldExt;
use pasta_curves::{pallas, vesta};
use vdf::minroot::{EvalMode, MinRootVDF, PallasVDF, State, VanillaVDFProof, VestaVDF};

fn bench_eval<V: MinRootVDF<F>, F: FieldExt>(eval_mode: EvalMode, c: &mut Criterion, name: &str) {
    let t = 10000;
    let mut group = c.benchmark_group(format!("{}VDF-eval-{:?}-{}", name, eval_mode, t));

    let x = State {
        x: V::element(123),
        y: V::element(321),
        i: F::zero(),
    };

    group.bench_function("eval_and_prove", |b| {
        b.iter(|| {
            VanillaVDFProof::<V, F>::eval_and_prove_with_mode(eval_mode, x, t);
        });
    });

    group.finish();
}

fn bench_verify<V: MinRootVDF<F>, F: FieldExt>(c: &mut Criterion, name: &str) {
    let t = 10000;
    let mut group = c.benchmark_group(format!("{}VDF-verify-{}", name, t));

    let x = State {
        x: V::element(123),
        y: V::element(321),
        i: F::zero(),
    };

    group.bench_function("verify", |b| {
        let proof = VanillaVDFProof::<V, F>::eval_and_prove(x, t);

        b.iter(|| {
            proof.verify(x);
        });
    });
    group.finish();
}

fn bench_pallas(c: &mut Criterion) {
    for eval_mode in EvalMode::all().iter() {
        bench_eval::<PallasVDF, pallas::Scalar>(*eval_mode, c, "Pallas")
    }

    bench_verify::<PallasVDF, pallas::Scalar>(c, "Pallas")
}
fn bench_vesta(c: &mut Criterion) {
    bench_eval::<VestaVDF, vesta::Scalar>(EvalMode::LTRSequential, c, "Vesta");
    bench_verify::<VestaVDF, vesta::Scalar>(c, "Vesta")
}

criterion_group! {
    name = vdf;
    config = Criterion::default().sample_size(60);
    targets = bench_pallas, bench_vesta
}

criterion_main!(vdf);
