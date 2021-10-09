use pasta_curves::arithmetic::Field;
use pasta_curves::pallas;

use vdf::minroot::{MinRootVDF, PallasVDF, State, VanillaVDFProof};

use vdf::nova::proof::{make_nova_proof, PallasScalar, RawVanillaProof};

use vdf::TEST_SEED;

use rand::SeedableRng;
use rand_xorshift::XorShiftRng;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

fn bench_nova_proof<V: MinRootVDF<pallas::Scalar>>(c: &mut Criterion) {
    let mut rng = XorShiftRng::from_seed(TEST_SEED);

    type F = pallas::Scalar;

    let x = Field::random(&mut rng);
    let y = F::zero();
    let x = State { x, y, i: F::zero() };
    let t = 100;
    let n = 100;

    let first_vanilla_proof = VanillaVDFProof::<V, F>::eval_and_prove(x, t);

    let mut all_vanilla_proofs = Vec::with_capacity(12);
    all_vanilla_proofs.push(first_vanilla_proof.clone());

    c.bench_with_input(
        BenchmarkId::new("Nova VDF Proof", format!("t: {}; n: {}", t, n)),
        &t,
        |b, n| {
            let final_vanilla_proof = (1..*n).fold(first_vanilla_proof.clone(), |acc, _| {
                let new_proof = VanillaVDFProof::<V, F>::eval_and_prove(acc.result, t);
                all_vanilla_proofs.push(new_proof.clone());
                acc.append(new_proof).expect("failed to append proof")
            });

            assert_eq!(
                V::element(final_vanilla_proof.t),
                final_vanilla_proof.result.i
            );
            assert_eq!(n * t, final_vanilla_proof.t);
            assert!(final_vanilla_proof.verify(x));

            let raw_vanilla_proofs: Vec<RawVanillaProof<pallas::Scalar>> = all_vanilla_proofs
                .iter()
                .map(|p| (p.clone()).into())
                .collect();

            let (shape, gens) = RawVanillaProof::<PallasScalar>::new_empty(raw_vanilla_proofs[0].t)
                .make_nova_shape_and_gens();

            b.iter(|| make_nova_proof::<PallasScalar>(&raw_vanilla_proofs, &shape, &gens));
        },
    );
}

fn bench_nova(c: &mut Criterion) {
    bench_nova_proof::<PallasVDF>(c);
}

criterion_group! {
    name = nova;
    config = Criterion::default().sample_size(60);
    targets = bench_nova
}

criterion_main!(nova);
