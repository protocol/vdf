use std::fmt::Debug;

use bellperson::{
    gadgets::{
        boolean::Boolean,
        num::{AllocatedNum, Num},
    },
    ConstraintSystem, LinearCombination, SynthesisError,
};

use ff::{Field, PrimeField};

use nova::{
    errors::NovaError,
    traits::{
        circuit::{StepCircuit, TrivialTestCircuit},
        Group,
    },
    CompressedSNARK, RecursiveSNARK,
};

use pasta_curves::{pallas, vesta};

use crate::minroot::{MinRootVDF, State, VanillaVDFProof};

type G1 = pallas::Point;
type G2 = vesta::Point;

type S1 = pallas::Scalar;
type S2 = vesta::Scalar;

type SS1 = nova::spartan_with_ipa_pc::RelaxedR1CSSNARK<G1>;
type SS2 = nova::spartan_with_ipa_pc::RelaxedR1CSSNARK<G2>;

type C1 = InverseMinRootCircuit<G1>;
type C2 = TrivialTestCircuit<<G2 as Group>::Scalar>;

type NovaVDFPublicParams = nova::PublicParams<
    G1,
    G2,
    InverseMinRootCircuit<G1>,
    TrivialTestCircuit<<G2 as Group>::Scalar>,
>;

#[derive(Debug)]
pub enum Error {
    Nova(NovaError),
    Synthesis(SynthesisError),
}

#[allow(clippy::large_enum_variant)]
pub enum NovaVDFProof {
    Recursive(RecursiveSNARK<G1, G2, C1, C2>),
    Compressed(CompressedSNARK<G1, G2, C1, C2, SS1, SS2>),
}

#[derive(Clone, Debug)]
pub struct InverseMinRootCircuit<G>
where
    G: Debug + Group,
{
    pub inverse_exponent: u64,
    pub result: Option<State<G::Scalar>>,
    pub input: Option<State<G::Scalar>>,
    pub t: u64,
}

impl<G: Group> InverseMinRootCircuit<G> {
    fn new<V: MinRootVDF<G>>(v: &VanillaVDFProof<V, G>, previous_state: State<G::Scalar>) -> Self {
        InverseMinRootCircuit {
            inverse_exponent: V::inverse_exponent(),
            result: Some(v.result),
            input: Some(previous_state),
            t: v.t,
        }
    }
}

impl<G> StepCircuit<G::Scalar> for InverseMinRootCircuit<G>
where
    G: Group,
{
    fn arity(&self) -> usize {
        3
    }

    fn synthesize<CS>(
        &self,
        cs: &mut CS,
        z: &[AllocatedNum<G::Scalar>],
    ) -> Result<Vec<AllocatedNum<G::Scalar>>, SynthesisError>
    where
        CS: ConstraintSystem<G::Scalar>,
    {
        assert_eq!(self.arity(), z.len());

        let t = self.t;
        let mut x = z[0].clone();
        let mut y = z[1].clone();
        let i = z[2].clone();
        let mut i_num = Num::from(i);

        let mut final_x = x.clone();
        let mut final_y = y.clone();
        let mut final_i_num = i_num.clone();

        for j in 0..t {
            let (new_i, new_x, new_y) = inverse_round(
                &mut cs.namespace(|| format!("inverse_round_{}", j)),
                i_num,
                x,
                y,
            )?;
            final_x = new_x.clone();
            final_y = new_y.clone();
            final_i_num = new_i.clone();
            i_num = new_i;
            x = new_x;
            y = new_y;
        }

        let final_i = AllocatedNum::<G::Scalar>::alloc(&mut cs.namespace(|| "final_i"), || {
            final_i_num
                .get_value()
                .ok_or(SynthesisError::AssignmentMissing)
        })?;

        cs.enforce(
            || "final_i matches final_i_num",
            |lc| lc + final_i.get_variable(),
            |lc| lc + CS::one(),
            |_| final_i_num.lc(G::Scalar::one()),
        );

        let res = vec![final_x, final_y, final_i];

        assert_eq!(self.arity(), z.len());

        Ok(res)
    }

    fn output(&self, z: &[G::Scalar]) -> Vec<G::Scalar> {
        // sanity check
        let result = self.result.expect("result missing");
        let state = self.input.expect("state missing");

        debug_assert_eq!(z[0], result.x);
        debug_assert_eq!(z[1], result.y);
        debug_assert_eq!(z[2], result.i);

        vec![state.x, state.y, state.i]
    }
}

fn inverse_round<CS: ConstraintSystem<F>, F: PrimeField>(
    cs: &mut CS,
    i: Num<F>,
    x: AllocatedNum<F>,
    y: AllocatedNum<F>,
) -> Result<(Num<F>, AllocatedNum<F>, AllocatedNum<F>), SynthesisError> {
    // i = i - 1
    let new_i = i
        .clone()
        .add_bool_with_coeff(CS::one(), &Boolean::Constant(true), -F::from(1));

    // new_x = y - new_i = y - i + 1
    let new_x = AllocatedNum::<F>::alloc(&mut cs.namespace(|| "new_x"), || {
        if let (Some(y), Some(new_i)) = (y.get_value(), new_i.get_value()) {
            Ok(y - new_i)
        } else {
            Err(SynthesisError::AssignmentMissing)
        }
    })?;

    // tmp1 = x * x
    let tmp1 = x.square(&mut cs.namespace(|| "tmp1"))?;
    // tmp2 = tmp1 * tmp1
    let tmp2 = tmp1.square(&mut cs.namespace(|| "tmp2"))?;

    // new_y = (tmp2 * x) - new_x
    let new_y = AllocatedNum::<F>::alloc(&mut cs.namespace(|| "new_y"), || {
        if let (Some(x), Some(new_x), Some(tmp2)) =
            (x.get_value(), new_x.get_value(), tmp2.get_value())
        {
            Ok((tmp2 * x) - new_x)
        } else {
            Err(SynthesisError::AssignmentMissing)
        }
    })?;

    // new_y = (tmp2 * x) - new_x
    // (tmp2 * x) = new_y + new_x
    // (tmp2 * x) = new_y + y - i + 1
    if tmp2.get_value().is_some() {
        debug_assert_eq!(
            tmp2.get_value().ok_or(SynthesisError::AssignmentMissing)?
                * x.get_value().ok_or(SynthesisError::AssignmentMissing)?,
            new_y.get_value().ok_or(SynthesisError::AssignmentMissing)?
                + new_x.get_value().ok_or(SynthesisError::AssignmentMissing)?,
        );

        debug_assert_eq!(
            new_x.get_value().ok_or(SynthesisError::AssignmentMissing)?,
            y.get_value().ok_or(SynthesisError::AssignmentMissing)?
                - i.get_value().ok_or(SynthesisError::AssignmentMissing)?
                + F::one()
        );

        debug_assert_eq!(
            tmp2.get_value().ok_or(SynthesisError::AssignmentMissing)?
                * x.get_value().ok_or(SynthesisError::AssignmentMissing)?,
            new_y.get_value().ok_or(SynthesisError::AssignmentMissing)?
                + y.get_value().ok_or(SynthesisError::AssignmentMissing)?
                - i.get_value().ok_or(SynthesisError::AssignmentMissing)?
                + F::one()
        );
    }

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

impl<G: Group> InverseMinRootCircuit<G> {
    pub fn circuits(
        num_iters_per_step: u64,
    ) -> (InverseMinRootCircuit<G>, TrivialTestCircuit<G::Base>) {
        (
            Self::circuit_primary(num_iters_per_step),
            Self::circuit_secondary(),
        )
    }

    pub fn circuit_primary(num_iters_per_step: u64) -> InverseMinRootCircuit<G> {
        InverseMinRootCircuit {
            inverse_exponent: 5,
            result: None,
            input: None,
            t: num_iters_per_step,
        }
    }

    pub fn circuit_secondary() -> TrivialTestCircuit<G::Base> {
        TrivialTestCircuit::default()
    }

    pub fn eval_and_make_circuits<V: MinRootVDF<G>>(
        _v: V,
        num_iters_per_step: u64,
        num_steps: usize,
        initial_state: State<G::Scalar>,
    ) -> (Vec<G::Scalar>, Vec<InverseMinRootCircuit<G>>) {
        assert!(num_steps > 0);

        let (z0_primary, all_vanilla_proofs) = {
            let mut all_vanilla_proofs = Vec::with_capacity(num_steps);
            let mut state = initial_state;
            let mut z0_primary_opt = None;
            for _ in 0..num_steps {
                let (z0, proof) =
                    VanillaVDFProof::<V, G>::eval_and_prove(state, num_iters_per_step);
                state = proof.result;
                all_vanilla_proofs.push(proof);
                z0_primary_opt = Some(z0);
            }
            let z0_primary = z0_primary_opt.unwrap();
            (z0_primary, all_vanilla_proofs)
        };

        let circuits = {
            let mut previous_state = initial_state;
            let mut circuits = all_vanilla_proofs
                .iter()
                .map(|p| {
                    let rvp = Self::new(p, previous_state);
                    previous_state = rvp.result.unwrap();
                    rvp
                })
                .collect::<Vec<_>>();
            circuits.reverse();
            circuits
        };
        (z0_primary, circuits)
    }
}

impl NovaVDFProof {
    pub fn prove_recursively(
        pp: &NovaVDFPublicParams,
        circuits: &[InverseMinRootCircuit<G1>],
        num_iters_per_step: u64,
        z0: Vec<S1>,
    ) -> Result<Self, Error> {
        let debug = false;
        let z0_primary = z0;
        let z0_secondary = Self::z0_secondary();

        let (_circuit_primary, circuit_secondary) =
            InverseMinRootCircuit::<G1>::circuits(num_iters_per_step);

        // produce a recursive SNARK
        let mut recursive_snark: Option<RecursiveSNARK<G1, G2, C1, C2>> = None;

        for (i, circuit_primary) in circuits.iter().enumerate() {
            if debug {
                // For debugging purposes, synthesize the circuit and check that the constraint system is satisfied.
                use bellperson::util_cs::test_cs::TestConstraintSystem;
                let mut cs = TestConstraintSystem::<<G1 as Group>::Scalar>::new();

                let r = circuit_primary.result.unwrap();

                let zi_allocated = vec![
                    AllocatedNum::alloc(cs.namespace(|| format!("z{}_1", i)), || Ok(r.x))
                        .map_err(Error::Synthesis)?,
                    AllocatedNum::alloc(cs.namespace(|| format!("z{}_2", i)), || Ok(r.y))
                        .map_err(Error::Synthesis)?,
                    AllocatedNum::alloc(cs.namespace(|| format!("z{}_0", i)), || Ok(r.i))
                        .map_err(Error::Synthesis)?,
                ];

                circuit_primary
                    .synthesize(&mut cs, zi_allocated.as_slice())
                    .map_err(Error::Synthesis)?;

                assert!(cs.is_satisfied());
            }

            let res = RecursiveSNARK::prove_step(
                pp,
                recursive_snark,
                circuit_primary.clone(),
                circuit_secondary.clone(),
                z0_primary.clone(),
                z0_secondary.clone(),
            );
            if res.is_err() {
                dbg!(&res);
            }
            assert!(res.is_ok());
            recursive_snark = Some(res.map_err(Error::Nova)?);
        }

        Ok(Self::Recursive(recursive_snark.unwrap()))
    }

    pub fn compress(self, pp: &NovaVDFPublicParams) -> Result<Self, Error> {
        match &self {
            Self::Recursive(recursive_snark) => Ok(Self::Compressed(
                CompressedSNARK::<_, _, _, _, SS1, SS2>::prove(pp, recursive_snark)
                    .map_err(Error::Nova)?,
            )),
            Self::Compressed(_) => Ok(self),
        }
    }

    pub fn verify(
        &self,
        pp: &NovaVDFPublicParams,
        num_steps: usize,
        z0: Vec<S1>,
        zi: &[S1],
    ) -> Result<bool, NovaError> {
        let (z0_primary, zi_primary) = (z0, zi);
        let z0_secondary = Self::z0_secondary();
        let zi_secondary = z0_secondary.clone();

        let (zi_primary_verified, zi_secondary_verified) = match self {
            Self::Recursive(p) => p.verify(pp, num_steps, z0_primary, z0_secondary),
            Self::Compressed(p) => p.verify(pp, num_steps, z0_primary, z0_secondary),
        }?;

        Ok(zi_primary == zi_primary_verified && zi_secondary == zi_secondary_verified)
    }

    fn z0_secondary() -> Vec<S2> {
        vec![<G2 as Group>::Scalar::zero()]
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::minroot::{PallasVDF, State};
    use crate::TEST_SEED;

    use rand::SeedableRng;
    use rand_xorshift::XorShiftRng;

    #[test]
    fn test_nova_proof() {
        test_nova_proof_aux::<PallasVDF>(5, 3);
    }

    fn test_nova_proof_aux<V: MinRootVDF<G1> + PartialEq>(
        num_iters_per_step: u64,
        num_steps: usize,
    ) {
        let mut rng = XorShiftRng::from_seed(TEST_SEED);

        type F = S1;
        type G = G1;

        let x = Field::random(&mut rng);
        let y = F::zero();
        let initial_i = F::one();

        let initial_state = State { x, y, i: initial_i };
        let zi_primary = vec![x, y, initial_i];

        let (circuit_primary, circuit_secondary) =
            InverseMinRootCircuit::circuits(num_iters_per_step);

        // produce public parameters
        let pp = NovaVDFPublicParams::setup(circuit_primary, circuit_secondary.clone());

        let (z0_primary, circuits) = InverseMinRootCircuit::eval_and_make_circuits(
            V::new(),
            num_iters_per_step,
            num_steps,
            initial_state,
        );

        let recursive_snark =
            NovaVDFProof::prove_recursively(&pp, &circuits, num_iters_per_step, z0_primary.clone())
                .unwrap();

        // verify the recursive SNARK
        let res = recursive_snark.verify(&pp, num_steps, z0_primary.clone(), &zi_primary);

        if !res.is_ok() {
            dbg!(&res);
        }
        assert!(res.unwrap());

        // produce a compressed SNARK
        let compressed_snark = recursive_snark.compress(&pp).unwrap();
        // verify the compressed SNARK
        let res = compressed_snark.verify(&pp, num_steps, z0_primary, &zi_primary);
        assert!(res.is_ok());
    }
}
