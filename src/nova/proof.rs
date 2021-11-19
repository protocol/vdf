use std::fmt::Debug;

use bellperson::{
    gadgets::boolean::Boolean, gadgets::num::AllocatedNum, gadgets::num::Num, Circuit,
    ConstraintSystem, LinearCombination, SynthesisError,
};

use merlin::Transcript;
use nova::{
    bellperson::{
        r1cs::{NovaShape, NovaWitness},
        shape_cs::ShapeCS,
        solver::SatisfyingAssignment,
    },
    errors::NovaError,
    r1cs::{
        R1CSGens, R1CSInstance, R1CSShape, R1CSWitness, RelaxedR1CSInstance, RelaxedR1CSWitness,
    },
    traits::PrimeField,
};

use pasta_curves::{arithmetic::FieldExt, pallas};

use crate::minroot::{MinRootVDF, State, VanillaVDFProof};

use nova::{FinalSNARK, StepSNARK};

pub type PallasPoint = pallas::Point;
pub type PallasScalar = pallas::Scalar;

pub type PallasGroup = PallasPoint;

pub struct NovaVDFProof {
    final_proof: FinalSNARK<PallasGroup>,
    final_instance: RelaxedR1CSInstance<PallasGroup>,
}

#[derive(Clone, Debug, Default)]
pub struct RawVanillaProof<S>
where
    S: Debug,
{
    pub inverse_exponent: u64,
    pub result: Option<State<S>>,
    pub intermediates: Option<Vec<State<S>>>,
    pub t: u64,
}

impl<S: Debug + Default> RawVanillaProof<S> {
    pub fn new_empty(t: u64) -> Self {
        Self {
            inverse_exponent: 5,
            result: None,
            intermediates: None,
            t,
        }
    }
}

impl RawVanillaProof<PallasScalar> {
    pub fn make_nova_r1cs(
        self,
        shape: &R1CSShape<PallasGroup>,
        gens: &R1CSGens<PallasGroup>,
    ) -> Result<(R1CSInstance<PallasPoint>, R1CSWitness<PallasPoint>), NovaError> {
        let mut cs = SatisfyingAssignment::<PallasGroup>::new();

        self.synthesize(&mut cs).unwrap();

        let (instance, witness) = cs.r1cs_instance_and_witness(shape, gens)?;

        Ok((instance, witness))
    }

    pub fn make_nova_shape_and_gens(&self) -> (R1CSShape<PallasGroup>, R1CSGens<PallasGroup>) {
        let mut cs = ShapeCS::<PallasGroup>::new();
        self.clone().synthesize(&mut cs).unwrap();
        let shape = cs.r1cs_shape();
        let gens = cs.r1cs_gens();

        (shape, gens)
    }
}

impl<V: MinRootVDF<F>, F: FieldExt> From<VanillaVDFProof<V, F>> for RawVanillaProof<F> {
    fn from(v: VanillaVDFProof<V, F>) -> Self {
        RawVanillaProof {
            inverse_exponent: V::inverse_exponent(),
            result: Some(v.result),
            intermediates: v.intermediates,
            t: v.t,
        }
    }
}

pub fn make_nova_proof<S: Into<PallasScalar> + Copy + Clone + std::fmt::Debug>(
    proofs: &[RawVanillaProof<PallasScalar>],
    shape: &R1CSShape<PallasGroup>,
    gens: &R1CSGens<PallasGroup>,
    verify_steps: bool, // Sanity check for development, until we have recursion.
) -> (NovaVDFProof, RelaxedR1CSInstance<PallasGroup>) {
    let mut r1cs_instances = proofs
        .iter()
        .map(|p| p.clone().make_nova_r1cs(shape, gens).unwrap())
        .collect::<Vec<_>>();

    r1cs_instances.reverse();

    // TODO: Handle other cases.
    assert!(r1cs_instances.len() > 1);

    let mut step_proofs = Vec::new();
    let mut prover_transcript = Transcript::new(b"MinRootPallas");
    let mut verifier_transcript = Transcript::new(b"MinRootPallas");

    let initial_acc = (
        RelaxedR1CSInstance::default(gens, shape),
        RelaxedR1CSWitness::default(shape),
    );

    let (acc_U, acc_W) =
        r1cs_instances
            .iter()
            .skip(1)
            .fold(initial_acc, |(acc_U, acc_W), (next_U, next_W)| {
                let (step_proof, (step_U, step_W)) = make_step_snark(
                    gens,
                    shape,
                    &acc_U,
                    &acc_W,
                    next_U,
                    next_W,
                    &mut prover_transcript,
                );
                if verify_steps {
                    step_proof
                        .verify(&acc_U, next_U, &mut verifier_transcript)
                        .unwrap();
                    step_proofs.push(step_proof);
                };
                (step_U, step_W)
            });

    let final_proof = make_final_snark(&acc_W);

    let proof = NovaVDFProof {
        final_proof,
        final_instance: acc_U.clone(),
    };

    (proof, acc_U)
}

fn make_step_snark(
    gens: &R1CSGens<PallasGroup>,
    S: &R1CSShape<PallasGroup>,
    r_U: &RelaxedR1CSInstance<PallasGroup>,
    r_W: &RelaxedR1CSWitness<PallasGroup>,
    U2: &R1CSInstance<PallasGroup>,
    W2: &R1CSWitness<PallasGroup>,
    prover_transcript: &mut merlin::Transcript,
) -> (
    StepSNARK<PallasGroup>,
    (
        RelaxedR1CSInstance<PallasPoint>,
        RelaxedR1CSWitness<PallasPoint>,
    ),
) {
    let res = StepSNARK::prove(gens, S, r_U, r_W, U2, W2, prover_transcript);
    res.expect("make_step_snark failed")
}

fn make_final_snark(W: &RelaxedR1CSWitness<PallasPoint>) -> FinalSNARK<PallasGroup> {
    // produce a final SNARK
    let res = FinalSNARK::prove(W);
    res.expect("make_final_snark failed")
}

impl NovaVDFProof {
    fn verify(
        &self,
        gens: &R1CSGens<PallasGroup>,
        S: &R1CSShape<PallasGroup>,
        U: &RelaxedR1CSInstance<PallasGroup>,
    ) -> bool {
        let res = self.final_proof.verify(gens, S, U);
        res.is_ok()
    }
}

impl Circuit<PallasScalar> for RawVanillaProof<PallasScalar> {
    fn synthesize<CS>(self, cs: &mut CS) -> Result<(), SynthesisError>
    where
        CS: ConstraintSystem<PallasScalar>,
    {
        let (result_i, result_x, result_y) = if let Some(result) = self.result {
            (Some(result.i), Some(result.x), Some(result.y))
        } else {
            (None, None, None)
            //panic!("Cannot generate R1CSWitness or R1CSInstance without result values.");
        };

        let Self { t, .. } = self;

        let allocated_i =
            AllocatedNum::<PallasScalar>::alloc_input(&mut cs.namespace(|| "result_i"), || {
                result_i.ok_or(SynthesisError::AssignmentMissing)
            })?;
        let mut x =
            AllocatedNum::<PallasScalar>::alloc_input(&mut cs.namespace(|| "result_x"), || {
                result_x.ok_or(SynthesisError::AssignmentMissing)
            })?;
        let mut y =
            AllocatedNum::<PallasScalar>::alloc_input(&mut cs.namespace(|| "result_y"), || {
                result_y.ok_or(SynthesisError::AssignmentMissing)
            })?;

        let mut i = Num::from(allocated_i);

        for j in 0..t {
            let (new_i, new_x, new_y) = inverse_round(
                &mut cs.namespace(|| format!("inverse_round_{}", j)),
                i,
                x,
                y,
                j == t - 1,
            )?;
            i = new_i;
            x = new_x;
            y = new_y;
        }

        Ok(())
    }
}

fn inverse_round<CS: ConstraintSystem<PallasScalar>>(
    cs: &mut CS,
    i: Num<PallasScalar>,
    x: AllocatedNum<PallasScalar>,
    y: AllocatedNum<PallasScalar>,
    last_round: bool,
) -> Result<
    (
        Num<PallasScalar>,
        AllocatedNum<PallasScalar>,
        AllocatedNum<PallasScalar>,
    ),
    SynthesisError,
> {
    // i = i - 1
    let new_i =
        i.clone()
            .add_bool_with_coeff(CS::one(), &Boolean::Constant(true), -PallasScalar::from(1));

    if last_round {
        AllocatedNum::<PallasScalar>::alloc_input(&mut cs.namespace(|| "initial_i"), || {
            new_i.get_value().ok_or(SynthesisError::AssignmentMissing)
        })?;
    }

    // new_x = y - new_i = y - i + 1
    let new_x = AllocatedNum::<PallasScalar>::alloc_maybe_input(
        &mut cs.namespace(|| "new_x"),
        last_round,
        || {
            if let (Some(y), Some(new_i)) = (y.get_value(), new_i.get_value()) {
                Ok(y - new_i)
            } else {
                Err(SynthesisError::AssignmentMissing)
            }
        },
    )?;

    // tmp1 = x * x
    let tmp1 = x.square(&mut cs.namespace(|| "tmp1"))?;
    // tmp2 = tmp1 * tmp1
    let tmp2 = tmp1.square(&mut cs.namespace(|| "tmp2"))?;

    // new_y = (tmp2 * x) - new_x
    let new_y = AllocatedNum::<PallasScalar>::alloc_maybe_input(
        &mut cs.namespace(|| "new_y"),
        last_round,
        || {
            if let (Some(x), Some(new_x), Some(tmp2)) =
                (x.get_value(), new_x.get_value(), tmp2.get_value())
            {
                Ok((tmp2 * x) - new_x)
            } else {
                Err(SynthesisError::AssignmentMissing)
            }
        },
    )?;

    // new_y = (tmp2 * x) - new_x
    // (tmp2 * x) = new_y + new_x
    // (tmp2 * x) = new_y + y - i + 1
    cs.enforce(
        || "new_y + new_x = (tmp2 * x)",
        |lc| lc + tmp2.get_variable(),
        |lc| lc + x.get_variable(),
        |lc| {
            lc + new_y.get_variable() + y.get_variable() - &i.lc(1.into())
                + &LinearCombination::from_coeff(CS::one(), 1.into())
        },
    );

    Ok((new_i, new_x, new_y))
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
            .map(|p| (p.clone()).into())
            .collect();

        let (shape, gens) = RawVanillaProof::<PallasScalar>::new_empty(raw_vanilla_proofs[0].t)
            .make_nova_shape_and_gens();

        // This will panic if proof does not verify.
        // Actual complete verification is still awkward without recursion,
        // since we would need a verifier transcript supplied by the prover.
        let (nova_proof, acc_U) =
            make_nova_proof::<PallasScalar>(&raw_vanilla_proofs, &shape, &gens, true);

        assert!(nova_proof.verify(&gens, &shape, &acc_U));
    }
}
