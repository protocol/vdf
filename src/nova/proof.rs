use bellperson::nova::metric_cs::MetricCS;
use bellperson::nova::prover::ProvingAssignment;
use bellperson::nova::r1cs::{NovaShape, NovaWitness};

use bellperson::{Circuit, ConstraintSystem, SynthesisError};
use merlin::Transcript;
use nova::r1cs::{R1CSGens, R1CSInstance, R1CSShape, R1CSWitness};
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
    final_instance: R1CSInstance<MainGroup>,
}

#[derive(Debug)]
struct RawVanillaProof<S> {
    pub inverse_exponent: u64,
    pub result: State<S>,
    pub t: u64,
}

#[derive(Clone, Copy, Debug)]
struct RawVanillaProofPallas {
    pub inverse_exponent: u64,
    pub result: State<PallasScalar>,
    pub t: u64,
}

impl<F: Clone + Into<PallasScalar>> RawVanillaProof<F> {
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
        // x_n+1 = y_n - 1
        // y_n+1 = x_n^5

        // I0 = i_0
        // I1 = x_0
        // I2 = y_0
        // I3 = i_n
        // I4 = x_n
        // I5 = y_n

        // Z0 = I0 - 1 = i_1
        // Z1 = I2 - 1 = x_1
        // Z2 = Z1 * Z1
        // Z3 = Z2 * Z2
        // Z4 = Z3 * Z1 = y_1

        // when n > 0, i_n = Zk: k = 0 + 5(n-1)
        // when n > 0, x_n = Zk: k = 1 + 5(n-1)
        // when n > 0, y_n = Zk: k = 4 + 5(n-1)

        // R1CS

        // Initial:
        // (I0 - 1) * 1 - Z0 = 0
        // (I2 - 1) * 1 - Z1 = 0
        // I1 * I1 - Z2 = 0
        // Z2 * Z2 - Z3 = 0
        // Z3 * Z1 - Z4 = 0

        // Repeated:
        // (Z0 - 1) * 1 - Z5 = 0
        // (Z2 - 1) * 1 - Z6 = 0
        // Z1 * Z1 - Z7 = 0
        // Z7 * Z7 - Z8 = 0
        // Z8 * Z1 - Z9 = 0

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
                vec![(y_index, one), (one_index, neg_one)],
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
                vec![(w + 4, one)],
            );
        };

        let mut add_step_witnesses = |i: &PallasScalar, x: &PallasScalar, y: &PallasScalar| {
            let new_i = *i - one;
            witness.push(new_i);

            let new_x = *y - one;
            witness.push(new_x);

            let mut new_y = *x * *x;
            witness.push(new_y);

            new_y *= new_y;
            witness.push(new_y);

            new_y *= x;
            witness.push(new_y);

            (new_i, new_x, new_y)
        };

        {
            let mut w = 0;
            let mut i = result_i;
            let mut x = result_x;
            let mut y = result_y;

            for _ in 0..t {
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
            let E = vec![PallasScalar::zero(); num_cons]; // default E
            let res = R1CSWitness::new(&S, &witness, &E);
            assert!(res.is_ok());
            res.unwrap()
        };
        let U = {
            let (comm_W, comm_E) = W.commit(&gens);
            let u = PallasScalar::one(); //default u
            let res = R1CSInstance::new(&S, &comm_W, &comm_E, &inputs, &u);
            assert!(res.is_ok());
            res.unwrap()
        };
        ((U, W), (S, gens))
    }
}

impl RawVanillaProofPallas {
    fn make_nova_r1cs_with_bellman(
        self,
    ) -> (
        (R1CSInstance<PallasPoint>, R1CSWitness<PallasPoint>),
        (R1CSShape<MainGroup>, R1CSGens<MainGroup>),
    ) {
        let mut cs = ProvingAssignment::<PallasPoint>::new();
        self.clone().synthesize(&mut cs).unwrap();

        let (shape, gens) = self.make_nova_shape_and_gens_with_bellman();

        let instance = cs.r1cs_instance();
        let witness = cs.r1cs_witness();

        ((instance, witness), (shape, gens))
    }

    fn make_nova_shape_and_gens_with_bellman(&self) -> (R1CSShape<MainGroup>, R1CSGens<MainGroup>) {
        let mut cs = MetricCS::<MainGroup>::new();
        self.synthesize(&mut cs).unwrap();

        let shape = cs.r1cs_shape().into();
        let gens = cs.r1cs_gens().into();

        (shape, gens)
    }
}

impl Circuit<PallasScalar> for RawVanillaProofPallas {
    fn synthesize<CS>(self, _: &mut CS) -> Result<(), SynthesisError>
    where
        CS: ConstraintSystem<PallasScalar>,
    {
        todo!()
    }
}

impl<V: MinRootVDF<F>, F: FieldExt> From<VanillaVDFProof<V, F>> for RawVanillaProof<F> {
    fn from(v: VanillaVDFProof<V, F>) -> Self {
        RawVanillaProof {
            inverse_exponent: V::inverse_exponent(),
            result: v.result,
            t: v.t,
        }
    }
}

fn make_nova_proof<S: Into<PallasScalar> + Clone>(
    proofs: Vec<RawVanillaProof<S>>,
) -> (
    NovaVDFProof,
    (R1CSInstance<MainGroup>, R1CSWitness<MainGroup>),
) {
    let r1cs_instances = proofs
        .iter()
        .map(|p| p.make_nova_r1cs())
        .collect::<Vec<_>>();

    // TODO: Handle other cases.
    //assert!(r1cs_instances.len() > 1);

    let mut step_proofs = Vec::new();
    let mut prover_transcript = Transcript::new(b"MinRootPallas");

    let (S, gens) = &r1cs_instances[0].1;
    let initial_acc = r1cs_instances[0].0.clone();

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

    // FIXME: Without recursion, we actually need to retain all the step proofs;
    // and verification should verify them, as well as their relationships. For
    // now, skip all that. Given that we will never actually use this
    // non-recursive verification strategy, should we even bother, or just wait for
    // recursion?
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
    U1: &R1CSInstance<MainGroup>,
    W1: &R1CSWitness<MainGroup>,
    U2: &R1CSInstance<MainGroup>,
    W2: &R1CSWitness<MainGroup>,
    prover_transcript: &mut merlin::Transcript,
) -> (
    StepSNARK<MainGroup>,
    (R1CSInstance<PallasPoint>, R1CSWitness<PallasPoint>),
) {
    let res = StepSNARK::prove(gens, S, U1, W1, U2, W2, prover_transcript);
    assert!(res.is_ok());
    res.unwrap()
}

fn make_final_snark(W: &R1CSWitness<PallasPoint>) -> FinalSNARK<MainGroup> {
    // produce a final SNARK
    let res = FinalSNARK::prove(W);
    assert!(res.is_ok());
    res.unwrap()
}

impl NovaVDFProof {
    fn verify(
        &self,
        gens: &R1CSGens<MainGroup>,
        S: &R1CSShape<MainGroup>,
        U: &R1CSInstance<MainGroup>,
    ) -> bool {
        let res = self.final_proof.verify(gens, S, U);
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

fn make_nova_r1cs(
    exp: u64,
    t: u64,
    result_i: PallasScalar,
    result_x: PallasScalar,
    result_y: PallasScalar,
) -> (
    (R1CSInstance<PallasPoint>, R1CSWitness<PallasPoint>),
    (R1CSShape<MainGroup>, R1CSGens<MainGroup>),
) {
    // For now, hard code this for simplicity.
    assert_eq!(5, exp);

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
    // x_n+1 = y_n - 1
    // y_n+1 = x_n^5

    // I0 = i_0
    // I1 = x_0
    // I2 = y_0
    // I3 = i_n
    // I4 = x_n
    // I5 = y_n

    // Z0 = I0 - 1 = i_1
    // Z1 = I2 - 1 = x_1
    // Z2 = Z1 * Z1
    // Z3 = Z2 * Z2
    // Z4 = Z3 * Z1 = y_1

    // when n > 0, i_n = Zk: k = 0 + 5(n-1)
    // when n > 0, x_n = Zk: k = 1 + 5(n-1)
    // when n > 0, y_n = Zk: k = 4 + 5(n-1)

    // R1CS

    // Initial:
    // (I0 - 1) * 1 - Z0 = 0
    // (I2 - 1) * 1 - Z1 = 0
    // I1 * I1 - Z2 = 0
    // Z2 * Z2 - Z3 = 0
    // Z3 * Z1 - Z4 = 0

    // Repeated:
    // (Z0 - 1) * 1 - Z5 = 0
    // (Z2 - 1) * 1 - Z6 = 0
    // Z1 * Z1 - Z7 = 0
    // Z7 * Z7 - Z8 = 0
    // Z8 * Z1 - Z9 = 0

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
            vec![(y_index, one), (one_index, neg_one)],
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
            vec![(w + 4, one)],
        );
    };

    let mut add_step_witnesses = |i: &PallasScalar, x: &PallasScalar, y: &PallasScalar| {
        let new_i = *i - one;
        witness.push(new_i);

        let new_x = *y - one;
        witness.push(new_x);

        let mut new_y = *x * *x;
        witness.push(new_y);

        new_y *= new_y;
        witness.push(new_y);

        new_y *= x;
        witness.push(new_y);

        (new_i, new_x, new_y)
    };

    {
        let mut w = 0;
        let mut i = result_i;
        let mut x = result_x;
        let mut y = result_y;

        for _ in 0..t {
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
        let E = vec![PallasScalar::zero(); num_cons]; // default E
        let res = R1CSWitness::new(&S, &witness, &E);
        assert!(res.is_ok());
        res.unwrap()
    };
    let U = {
        let (comm_W, comm_E) = W.commit(&gens);
        let u = PallasScalar::one(); //default u
        let res = R1CSInstance::new(&S, &comm_W, &comm_E, &inputs, &u);
        assert!(res.is_ok());
        res.unwrap()
    };
    ((U, W), (S, gens))
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
            let num_inputs = 1;

            // The R1CS for this problem consists of the following constraints:
            // `Z0 * Z0 - Z1 = 0`
            // `Z1 * Z0 - Z2 = 0`
            // `(Z2 + Z0) * 1 - Z3 = 0`
            // `(Z3 + 5) * 1 - I0 = 0`

            // Relaxed R1CS is a set of three sparse matrices (A B C), where there is a row for every
            // constraint and a column for every entry in z = (vars, u, inputs)
            // An R1CS instance is satisfiable iff:
            // Az \circ Bz = u \cdot Cz + E, where z = (vars, 1, inputs)
            let mut A: Vec<(usize, usize, S)> = Vec::new();
            let mut B: Vec<(usize, usize, S)> = Vec::new();
            let mut C: Vec<(usize, usize, S)> = Vec::new();

            let mut X = (&mut A, &mut B, &mut C, &mut num_cons);

            // constraint 0 entries in (A,B,C)
            add_constraint(&mut X, vec![(0, one)], vec![(0, one)], vec![(1, one)]);

            // constraint 1 entries in (A,B,C)
            add_constraint(&mut X, vec![(1, one)], vec![(0, one)], vec![(2, one)]);

            // constraint 2 entries in (A,B,C)
            add_constraint(
                &mut X,
                vec![(2, one), (0, one)],
                vec![(num_vars, one)],
                vec![(3, one)],
            );

            // constraint 3 entries in (A,B,C)
            add_constraint(
                &mut X,
                vec![(3, one), (num_vars, one + one + one + one + one)],
                vec![(num_vars, one)],
                vec![(num_vars + 1, one)],
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
            |gens: &R1CSGens<G>| -> (R1CSInstance<G>, R1CSWitness<G>) {
                // compute a satisfying (vars, X) tuple
                let (vars, X) = {
                    let mut csprng: OsRng = OsRng;
                    let z0 = S::random(&mut csprng);
                    let z1 = z0 * z0; // constraint 0
                    let z2 = z1 * z0; // constraint 1
                    let z3 = z2 + z0; // constraint 2
                    let i0 = z3 + one + one + one + one + one; // constraint 3

                    let vars = vec![z0, z1, z2, z3];
                    let X = vec![i0];
                    (vars, X)
                };

                let W = {
                    let E = vec![S::zero(); num_cons]; // default E
                    let res = R1CSWitness::new(&S, &vars, &E);
                    assert!(res.is_ok());
                    res.unwrap()
                };
                let U = {
                    let (comm_W, comm_E) = W.commit(gens);
                    let u = S::one(); //default u
                    let res = R1CSInstance::new(&S, &comm_W, &comm_E, &X, &u);
                    assert!(res.is_ok());
                    res.unwrap()
                };

                // check that generated instance is satisfiable
                let is_sat = S.is_sat(gens, &U, &W);
                assert!(is_sat.is_ok());
                (U, W)
            };
        let (U1, W1) = rand_inst_witness_generator(&gens);
        let (U2, W2) = rand_inst_witness_generator(&gens);

        // produce a step SNARK
        let mut prover_transcript = Transcript::new(b"MinRootPallas");
        let res = StepSNARK::prove(&gens, &S, &U1, &W1, &U2, &W2, &mut prover_transcript);
        assert!(res.is_ok());
        let (step_snark, (_U, W)) = res.unwrap();

        // verify the step SNARK
        let mut verifier_transcript = Transcript::new(b"MinRootPallas");
        let res = step_snark.verify(&U1, &U2, &mut verifier_transcript);
        assert!(res.is_ok());
        let U = res.unwrap();
        assert_eq!(U, _U);

        // produce a final SNARK
        let res = FinalSNARK::prove(&W);
        assert!(res.is_ok());
        let final_snark = res.unwrap();
        // verify the final SNARK
        let res = final_snark.verify(&gens, &S, &U);
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
        let t = 11;
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
                t: p.t,
            })
            .collect();

        let _nova_proof = make_nova_proof(raw_vanilla_proofs);
    }
}
