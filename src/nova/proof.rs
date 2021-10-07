// use bellperson::nova::metric_cs::MetricCS;
// use bellperson::nova::prover::ProvingAssignment;
// use bellperson::nova::r1cs::{NovaShape, NovaWitness};

// use bellperson::{Circuit, ConstraintSystem, SynthesisError};
use merlin::Transcript;
use nova::r1cs::{
    R1CSGens, R1CSInstance, R1CSShape, R1CSWitness, RelaxedR1CSInstance, RelaxedR1CSWitness,
};
use nova::traits::PrimeField;
use pasta_curves::arithmetic::FieldExt;
use pasta_curves::pallas;

use crate::minroot::{MinRootVDF, State, VanillaVDFProof};

use nova::{FinalSNARK, StepSNARK};

type PallasPoint = pallas::Point;
type PallasScalar = pallas::Scalar;

type MainGroup = PallasPoint;

pub struct NovaVDFProof {
    final_proof: FinalSNARK<MainGroup>,
    final_instance: RelaxedR1CSInstance<MainGroup>,
}

#[derive(Debug)]
struct RawVanillaProof<S>
where
    S: std::fmt::Debug,
{
    pub inverse_exponent: u64,
    pub result: State<S>,
    pub intermediates: Option<Vec<State<S>>>,
    pub t: u64,
}

#[derive(Clone, Copy, Debug)]
struct RawVanillaProofPallas {
    pub inverse_exponent: u64,
    pub result: State<PallasScalar>,
    pub t: u64,
}

impl<F: Copy + Clone + Into<PallasScalar> + std::fmt::Debug> RawVanillaProof<F> {
    fn make_nova_r1cs(
        &self,
    ) -> (
        (R1CSInstance<PallasPoint>, R1CSWitness<PallasPoint>),
        (R1CSShape<MainGroup>, R1CSGens<MainGroup>),
    ) {
        let result_i = self.result.i.clone().into();
        let result_x = self.result.x.clone().into();
        let result_y = self.result.y.clone().into();
        let Self { t, .. } = *self;

        let one = PrimeField::one();
        let zero: PallasScalar = PrimeField::zero();
        let neg_one = zero - one;

        let mut num_cons = 0;
        let num_vars = 5 * t as usize;
        let num_inputs = 6;

        // For legibility: z[num_vars] = one.
        let one_index = num_vars;

        let mut witness: Vec<PallasScalar> = Vec::new();
        let mut inputs: Vec<PallasScalar> = Vec::new();

        let mut A: Vec<(usize, usize, PallasScalar)> = Vec::new();
        let mut B: Vec<(usize, usize, PallasScalar)> = Vec::new();
        let mut C: Vec<(usize, usize, PallasScalar)> = Vec::new();

        let mut X = (&mut A, &mut B, &mut C, &mut num_cons);

        // One step:
        // i_n+1 = i_n - 1
        // x_n+1 = y_n - i_n + 1
        // y_n+1 = x_n^5 - x_n+1

        // I0 = i_0
        // I1 = x_0
        // I2 = y_0
        // I3 = i_n
        // I4 = x_n
        // I5 = y_n

        // Z0 = I0 - 1 = i_1
        // Z1 = I2 - Z0 = x_1
        // Z2 = Z1 * Z1
        // Z3 = Z2 * Z2
        // Z4 = (Z3 * Z1) - Z1 = y_1

        // when n > 0, i_n = Zk: k = 0 + 5(n-1)
        // when n > 0, x_n = Zk: k = 1 + 5(n-1)
        // when n > 0, y_n = Zk: k = 4 + 5(n-1)

        // R1CS

        // Initial:
        // (I0 - 1) * 1 - Z0 = 0
        // (I2 - Z)) * 1 - Z1 = 0
        // I1 * I1 - Z2 = 0
        // Z2 * Z2 - Z3 = 0
        // Z3 * Z1 - (Z4 + Z1) = 0

        // Repeated:
        // (Z0 - 1) * 1 - Z5 = 0
        // (Z2 - Z0) * 1 - Z6 = 0
        // Z1 * Z1 - Z7 = 0
        // Z7 * Z7 - Z8 = 0
        // Z8 * Z1 - (Z9 + Z6) = 0

        // Repeat, t-1 times.

        // Witness is:
        // Z0 Z1 Z2 ... Zn-1 One I0 I1 I2 I3 I4
        //
        // Z0 = W[0]
        // One = W[num_vars]
        // I0 = W[num_vars + 1]

        // let mut i_index = num_vars + 1; // I0
        //                                 //    let mut x_index = num_vars + 2; // I1
        // let mut y_index = num_vars + 3; // I2

        // Add constraints and construct witness
        let mut add_step_constraints = |i_index, x_index, y_index, w| {
            add_constraint(
                &mut X,
                vec![(i_index, one), (one_index, neg_one)],
                vec![(one_index, one)],
                vec![(w, one)],
            );

            add_constraint(
                &mut X,
                vec![(y_index, one), (w, neg_one)],
                vec![(one_index, one)],
                vec![(w + 1, one)],
            );

            add_constraint(
                &mut X,
                vec![(x_index, one)],
                vec![(x_index, one)],
                vec![(w + 2, one)],
            );

            add_constraint(
                &mut X,
                vec![(w + 2, one)],
                vec![(w + 2, one)],
                vec![(w + 3, one)],
            );

            add_constraint(
                &mut X,
                vec![(w + 3, one)],
                vec![(x_index, one)],
                vec![(w + 4, one), (w + 1, one)],
            );
        };

        let mut add_step_witnesses = |i: &PallasScalar, x: &PallasScalar, y: &PallasScalar| {
            let new_i = *i - one;
            witness.push(new_i);

            let new_x = *y - new_i;
            witness.push(new_x);

            let mut new_y = *x * *x;
            witness.push(new_y);

            new_y *= new_y;
            witness.push(new_y);

            new_y *= x;
            new_y = new_y.sub(&new_x);
            witness.push(new_y);

            (new_i, new_x, new_y)
        };

        {
            let mut w = 0;
            let mut i = result_i;
            let mut x = result_x;
            let mut y = result_y;

            for j in 0..t {
                let (i_index, x_index, y_index) = if w == 0 {
                    (num_vars + 1, num_vars + 2, num_vars + 3)
                } else {
                    assert_eq!(0, w % 5);
                    (w - 5, w - 4, w - 1)
                };

                add_step_constraints(i_index, x_index, y_index, w);

                let (new_i, new_x, new_y) = add_step_witnesses(&i, &x, &y);

                i = new_i;
                x = new_x;
                y = new_y;

                w += 5;
            }
        }

        let add_final_constraints = || {
            // TODO: Add equality constraints or else optimize away the witness allocations.
        };

        let mut add_final_witnesses = || {
            let w = witness.len();
            inputs.push(result_i);
            inputs.push(result_x);
            inputs.push(result_y);
            // FIXME: Add equality constraints or else optimize away the witness allocations.
            inputs.push(witness[w - 5]);
            inputs.push(witness[w - 4]);
            inputs.push(witness[w - 1]);
        };

        add_final_constraints();
        add_final_witnesses();

        assert_eq!(witness.len(), num_vars);

        let (S, gens) = make_nova_shape_and_gens(num_cons, num_vars, num_inputs, A, B, C);

        let W = {
            let res = R1CSWitness::new(&S, &witness);
            assert!(res.is_ok());
            res.unwrap()
        };
        let U = {
            let comm_W = W.commit(&gens);
            let res = R1CSInstance::new(&S, &comm_W, &inputs);
            assert!(res.is_ok());
            res.unwrap()
        };
        ((U, W), (S, gens))
    }
}

impl<V: MinRootVDF<F>, F: FieldExt> From<VanillaVDFProof<V, F>> for RawVanillaProof<F> {
    fn from(v: VanillaVDFProof<V, F>) -> Self {
        RawVanillaProof {
            inverse_exponent: V::inverse_exponent(),
            result: v.result,
            intermediates: v.intermediates,
            t: v.t,
        }
    }
}

fn make_nova_proof<S: Into<PallasScalar> + Copy + Clone + std::fmt::Debug>(
    proofs: Vec<RawVanillaProof<S>>,
) -> (
    NovaVDFProof,
    (
        RelaxedR1CSInstance<MainGroup>,
        RelaxedR1CSWitness<MainGroup>,
    ),
) {
    let mut r1cs_instances = proofs
        .iter()
        .map(|p| p.make_nova_r1cs())
        .collect::<Vec<_>>();

    r1cs_instances.reverse();
    // TODO: Handle other cases.
    assert!(r1cs_instances.len() > 1);

    let mut step_proofs = Vec::new();
    let mut prover_transcript = Transcript::new(b"MinRootPallas");

    let (S, gens) = &r1cs_instances[0].1;
    let initial_acc = (
        RelaxedR1CSInstance::default(&gens, &S),
        RelaxedR1CSWitness::default(&S),
    );

    let (acc_U, acc_W) =
        r1cs_instances
            .iter()
            .skip(1)
            .fold(initial_acc, |(acc_U, acc_W), ((next_U, next_W), _)| {
                let (step_proof, (step_U, step_W)) = make_step_snark(
                    gens,
                    S,
                    &acc_U,
                    &acc_W,
                    next_U,
                    next_W,
                    &mut prover_transcript,
                );
                step_proofs.push(step_proof);
                (step_U.clone(), step_W.clone())
            });

    let final_proof = make_final_snark(&acc_W);

    let proof = NovaVDFProof {
        final_proof,
        final_instance: acc_U.clone(),
    };

    assert!(proof.verify(gens, S, &acc_U));

    (proof, (acc_U, acc_W))
}

fn make_step_snark(
    gens: &R1CSGens<MainGroup>,
    S: &R1CSShape<MainGroup>,
    r_U: &RelaxedR1CSInstance<MainGroup>,
    r_W: &RelaxedR1CSWitness<MainGroup>,
    U2: &R1CSInstance<MainGroup>,
    W2: &R1CSWitness<MainGroup>,
    prover_transcript: &mut merlin::Transcript,
) -> (
    StepSNARK<MainGroup>,
    (
        RelaxedR1CSInstance<PallasPoint>,
        RelaxedR1CSWitness<PallasPoint>,
    ),
) {
    let res = StepSNARK::prove(gens, S, r_U, r_W, U2, W2, prover_transcript);
    res.expect("make_step_snark failed")
}

fn make_final_snark(W: &RelaxedR1CSWitness<PallasPoint>) -> FinalSNARK<MainGroup> {
    // produce a final SNARK
    let res = FinalSNARK::prove(W);
    res.expect("make_final_snark failed")
}

impl NovaVDFProof {
    fn verify(
        &self,
        gens: &R1CSGens<MainGroup>,
        S: &R1CSShape<MainGroup>,
        U: &RelaxedR1CSInstance<MainGroup>,
    ) -> bool {
        let res = self.final_proof.verify(gens, S, U);
        res.clone().unwrap();
        res.is_ok()
    }
}

// TODO: Use this to only generate shape and gens once.
fn make_nova_shape_and_gens(
    num_cons: usize,
    num_vars: usize,
    num_inputs: usize,
    A: Vec<(usize, usize, PallasScalar)>,
    B: Vec<(usize, usize, PallasScalar)>,
    C: Vec<(usize, usize, PallasScalar)>,
) -> (R1CSShape<MainGroup>, R1CSGens<MainGroup>) {
    // create a shape object
    let S: R1CSShape<MainGroup> = {
        let res = R1CSShape::new(num_cons, num_vars, num_inputs, &A, &B, &C);
        assert!(res.is_ok());
        res.unwrap()
    };

    // generate generators
    let gens: R1CSGens<MainGroup> = R1CSGens::new(num_cons, num_vars);

    (S, gens)
}

fn add_constraint<S: PrimeField>(
    X: &mut (
        &mut Vec<(usize, usize, S)>,
        &mut Vec<(usize, usize, S)>,
        &mut Vec<(usize, usize, S)>,
        &mut usize,
    ),
    a_index_coeff_pairs: Vec<(usize, S)>,
    b_index_coeff_pairs: Vec<(usize, S)>,
    c_index_coeff_pairs: Vec<(usize, S)>,
) {
    let (A, B, C, nn) = X;
    let n = **nn;
    let one = S::one();

    for (index, coeff) in a_index_coeff_pairs {
        A.push((n, index, one * coeff));
    }
    for (index, coeff) in b_index_coeff_pairs {
        B.push((n, index, one * coeff));
    }
    for (index, coeff) in c_index_coeff_pairs {
        C.push((n, index, one * coeff));
    }
    **nn += 1;
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::minroot::{PallasVDF, State};
    use crate::TEST_SEED;
    use merlin::Transcript;
    use nova::traits::PrimeField;

    use pasta_curves::pallas;
    use rand::rngs::OsRng;
    use rand::SeedableRng;
    use rand_xorshift::XorShiftRng;

    type S = PallasScalar;
    type G = PallasPoint;

    #[test]
    #[allow(non_snake_case)]
    fn test_tiny_r1cs() {
        let one = S::one();
        let (num_cons, num_vars, num_inputs, A, B, C) = {
            let mut num_cons = 0;
            let num_vars = 4;
            let num_inputs = 2;

            // Consider a cubic equation: `x^3 + x + 5 = y`, where `x` and `y` are respectively the input and output.
            // The R1CS for this problem consists of the following constraints:
            // `I0 * I0 - Z0 = 0`
            // `Z0 * I0 - Z1 = 0`
            // `(Z1 + I0) * 1 - Z2 = 0`
            // `(Z2 + 5) * 1 - I1 = 0`

            // Relaxed R1CS is a set of three sparse matrices (A B C), where there is a row for every
            // constraint and a column for every entry in z = (vars, u, inputs)
            // An R1CS instance is satisfiable iff:
            // Az \circ Bz = u \cdot Cz + E, where z = (vars, 1, inputs)
            let mut A: Vec<(usize, usize, S)> = Vec::new();
            let mut B: Vec<(usize, usize, S)> = Vec::new();
            let mut C: Vec<(usize, usize, S)> = Vec::new();

            // // The R1CS for this problem consists of the following constraints:
            // // `Z0 * Z0 - Z1 = 0`
            // // `Z1 * Z0 - Z2 = 0`
            // // `(Z2 + Z0) * 1 - Z3 = 0`
            // // `(Z3 + 5) * 1 - I0 = 0`

            // // Relaxed R1CS is a set of three sparse matrices (A B C), where there is a row for every
            // // constraint and a column for every entry in z = (vars, u, inputs)
            // // An R1CS instance is satisfiable iff:
            // // Az \circ Bz = u \cdot Cz + E, where z = (vars, 1, inputs)
            // let mut A: Vec<(usize, usize, S)> = Vec::new();
            // let mut B: Vec<(usize, usize, S)> = Vec::new();
            // let mut C: Vec<(usize, usize, S)> = Vec::new();

            let mut X = (&mut A, &mut B, &mut C, &mut num_cons);

            // constraint 0 entries in (A,B,C)
            // `I0 * I0 - Z0 = 0`
            add_constraint(
                &mut X,
                vec![(num_vars + 1, one)],
                vec![(num_vars + 1, one)],
                vec![(0, one)],
            );

            // constraint 1 entries in (A,B,C)
            // `Z0 * I0 - Z1 = 0`
            add_constraint(
                &mut X,
                vec![(0, one)],
                vec![(num_vars + 1, one)],
                vec![(1, one)],
            );

            // constraint 2 entries in (A,B,C)
            // `(Z1 + I0) * 1 - Z2 = 0`
            add_constraint(
                &mut X,
                vec![(1, one), (num_vars + 1, one)],
                vec![(num_vars, one)],
                vec![(2, one)],
            );

            // constraint 3 entries in (A,B,C)
            // `(Z2 + 5) * 1 - I1 = 0`
            add_constraint(
                &mut X,
                vec![(2, one), (num_vars, one + one + one + one + one)],
                vec![(num_vars, one)],
                vec![(num_vars + 2, one)],
            );

            (num_cons, num_vars, num_inputs, A, B, C)
        };

        // create a shape object
        let S = {
            let res = R1CSShape::new(num_cons, num_vars, num_inputs, &A, &B, &C);
            assert!(res.is_ok());
            res.unwrap()
        };

        // generate generators
        let gens = R1CSGens::new(num_cons, num_vars);

        let rand_inst_witness_generator =
            |gens: &R1CSGens<G>, I: &S| -> (S, R1CSInstance<G>, R1CSWitness<G>) {
                let i0 = *I;

                // compute a satisfying (vars, X) tuple
                let (O, vars, X) = {
                    let z0 = i0 * i0; // constraint 0
                    let z1 = i0 * z0; // constraint 1
                    let z2 = z1 + i0; // constraint 2
                    let i1 = z2 + one + one + one + one + one; // constraint 3

                    // store the witness and IO for the instance
                    let W = vec![z0, z1, z2, S::zero()];
                    let X = vec![i0, i1];
                    (i1, W, X)
                };

                let W = {
                    let res = R1CSWitness::new(&S, &vars);
                    assert!(res.is_ok());
                    res.unwrap()
                };
                let U = {
                    let comm_W = W.commit(gens);
                    let res = R1CSInstance::new(&S, &comm_W, &X);
                    assert!(res.is_ok());
                    res.unwrap()
                };

                // check that generated instance is satisfiable
                assert!(S.is_sat(gens, &U, &W).is_ok());

                (O, U, W)
            };

        let mut csprng: OsRng = OsRng;
        let I = S::random(&mut csprng); // the first input is picked randomly for the first instance
        let (O, U1, W1) = rand_inst_witness_generator(&gens, &I);
        let (_O, U2, W2) = rand_inst_witness_generator(&gens, &O);

        // produce a default running instance
        let mut r_W = RelaxedR1CSWitness::default(&S);
        let mut r_U = RelaxedR1CSInstance::default(&gens, &S);

        // produce a step SNARK with (W1, U1) as the first incoming witness-instance pair
        let mut prover_transcript = Transcript::new(b"StepSNARKExample");
        let res = StepSNARK::prove(&gens, &S, &r_U, &r_W, &U1, &W1, &mut prover_transcript);
        assert!(res.is_ok());
        let (step_snark, (_U, W)) = res.unwrap();

        // verify the step SNARK with U1 as the first incoming instance
        let mut verifier_transcript = Transcript::new(b"StepSNARKExample");
        let res = step_snark.verify(&r_U, &U1, &mut verifier_transcript);
        assert!(res.is_ok());
        let U = res.unwrap();

        assert_eq!(U, _U);

        // update the running witness and instance
        r_W = W;
        r_U = U;

        // produce a step SNARK with (W2, U2) as the second incoming witness-instance pair
        let res = StepSNARK::prove(&gens, &S, &r_U, &r_W, &U2, &W2, &mut prover_transcript);
        assert!(res.is_ok());
        let (step_snark, (_U, W)) = res.unwrap();

        // verify the step SNARK with U1 as the first incoming instance
        let res = step_snark.verify(&r_U, &U2, &mut verifier_transcript);
        assert!(res.is_ok());
        let U = res.unwrap();

        assert_eq!(U, _U);

        // update the running witness and instance
        r_W = W;
        r_U = U;

        // produce a final SNARK
        let res = FinalSNARK::prove(&r_W);
        assert!(res.is_ok());
        let final_snark = res.unwrap();
        // verify the final SNARK
        let res = final_snark.verify(&gens, &S, &r_U);
        assert!(res.is_ok());
    }

    #[test]
    fn test_nova_proof() {
        test_nova_proof_aux::<PallasVDF>();
    }

    fn test_nova_proof_aux<V: MinRootVDF<pallas::Scalar>>() {
        use pasta_curves::arithmetic::Field;

        let mut rng = XorShiftRng::from_seed(TEST_SEED);

        type F = pallas::Scalar;

        let x = Field::random(&mut rng);
        let y = F::zero();
        let x = State { x, y, i: F::zero() };
        let t = 4;
        let n = 3;

        let first_vanilla_proof = VanillaVDFProof::<V, F>::eval_and_prove(x, t);

        let mut all_vanilla_proofs = Vec::with_capacity(12);
        all_vanilla_proofs.push(first_vanilla_proof.clone());

        let final_vanilla_proof = (1..n).fold(first_vanilla_proof, |acc, _| {
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
            .map(|p| RawVanillaProof::<pallas::Scalar> {
                inverse_exponent: V::inverse_exponent(),
                result: p.result,
                intermediates: p.intermediates.clone(),
                t: p.t,
            })
            .collect();

        let _nova_proof = make_nova_proof(raw_vanilla_proofs);
    }
}
