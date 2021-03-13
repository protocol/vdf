use criterion::{criterion_group, criterion_main, Criterion};
use halo2::arithmetic::FieldExt;
use halo2::pasta::{pallas, vesta};
use vdf::{PallasVDF, RaguVDF, RoundValue, VanillaVDFProof, VestaVDF};

fn bench_vdf<V: RaguVDF<F>, F: FieldExt>(c: &mut Criterion, name: &str) {
    let t = 10000;
    let mut group = c.benchmark_group(format!("{}VDF-{}", name, t));

    let x = RoundValue {
        value: V::element(123),
        round: F::zero(),
    };

    group.bench_function("eval_and_prove", |b| {
        b.iter(|| {
            VanillaVDFProof::<V, F>::eval_and_prove(x, t);
        });
    });

    group.bench_function("verify", |b| {
        let proof = VanillaVDFProof::<V, F>::eval_and_prove(x, t);

        b.iter(|| {
            proof.verify(x);
        });
    });
    group.finish();
}

fn bench_pallas(c: &mut Criterion) {
    bench_vdf::<PallasVDF, pallas::Scalar>(c, "Pallas")
}
fn bench_vesta(c: &mut Criterion) {
    bench_vdf::<VestaVDF, vesta::Scalar>(c, "Vesta")
}

criterion_group! {
    name = vdf;
    config = Criterion::default().sample_size(60);
    targets = bench_pallas, bench_vesta
}

criterion_main!(vdf);
