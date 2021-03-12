use core::fmt::Debug;
use std::marker::PhantomData;

use halo2::arithmetic::FieldExt;
use halo2::pasta::{pallas, vesta};

pub const TEST_SEED: [u8; 16] = [42; 16];

// Question: Should the naming of `PallasVDF` and `VestaVDF` be reversed?

/// Modulus is that of `Fq`, which is the base field of `Vesta` and scalar field of `Pallas`.
#[derive(Debug)]
pub struct PallasVDF {}
impl VDF<pallas::Scalar> for PallasVDF {
    fn element(n: u64) -> pallas::Scalar {
        pallas::Scalar::from(n)
    }
}

/// Modulus is that of `Fp`, which is the base field of `Pallas and scalar field of Vesta.
#[derive(Debug)]
pub struct VestaVDF {}
impl VDF<vesta::Scalar> for VestaVDF {
    fn element(n: u64) -> vesta::Scalar {
        vesta::Scalar::from(n)
    }
}

// Question: Is this right, or is it the reverse? Which scalar fields' modulus do we want to target?
pub type TargetVDF = PallasVDF;

#[derive(std::cmp::PartialEq, Debug, Clone, Copy)]
pub struct RoundValue<T> {
    pub value: T,
    pub round: T,
}

pub trait VDF<F>: Debug
where
    F: FieldExt,
{
    #[inline]
    /// Exponent used to take a root in the 'slow' direction.
    fn exponent() -> [u64; 4] {
        F::RESCUE_INVALPHA
    }

    #[inline]
    /// Exponent used in the 'fast' direction.
    fn inverse_exponent() -> u64 {
        F::RESCUE_ALPHA
    }

    #[inline]
    /// The building block of a round in the slow, 'forward' direction.
    fn forward_step(x: F) -> F {
        x.pow_vartime(Self::exponent())
    }

    #[inline]
    /// The building block of a round in the fast, 'inverse' direction.
    fn inverse_step(x: F) -> F {
        x.pow_vartime([Self::inverse_exponent(), 0, 0, 0])
    }

    /// One round in the slow/forward direction.
    fn round(x: RoundValue<F>) -> RoundValue<F> {
        RoundValue {
            // Increment the value by the round number so problematic values
            // like 0 and 1 don't consistently defeat the asymmetry.
            value: Self::forward_step(F::add(x.value, x.round)),
            // Increment the round.
            round: F::add(x.round, F::one()),
        }
    }

    /// One round in the fast/inverse direction.
    fn inverse_round(x: RoundValue<F>) -> RoundValue<F> {
        RoundValue {
            value: F::add(F::sub(Self::inverse_step(x.value), x.round), F::one()),
            round: F::sub(x.round, F::one()),
        }
    }

    /// Evaluate input `x` with time/difficulty parameter, `t` in the
    /// slow/forward direction.
    fn eval(x: RoundValue<F>, t: u64) -> RoundValue<F> {
        (0..t).fold(x, |acc, _| Self::round(acc))
    }

    /// Invert evaluation of output `x` with time/difficulty parameter, `t` in
    /// the fast/inverse direction.
    fn inverse_eval(x: RoundValue<F>, t: u64) -> RoundValue<F> {
        (0..t).fold(x, |acc, _| Self::inverse_round(acc))
    }

    /// Quickly check that `result` is the result of having slowly evaluated
    /// `original` with time/difficulty parameter `t`.
    fn check(result: RoundValue<F>, t: u64, original: RoundValue<F>) -> bool {
        original == Self::inverse_eval(result, t)
    }

    fn element(n: u64) -> F;
}

#[derive(Debug)]
pub struct VanillaVDFProof<V: VDF<F> + Debug, F: FieldExt> {
    result: RoundValue<F>,
    t: u64,
    _v: PhantomData<V>,
}

impl<V: VDF<F>, F: FieldExt> VanillaVDFProof<V, F> {
    pub fn eval_and_prove(x: RoundValue<F>, t: u64) -> Self {
        let result = V::eval(x, t);
        Self {
            result,
            t,
            _v: PhantomData::<V>,
        }
    }

    pub fn result(&self) -> RoundValue<F> {
        self.result
    }

    pub fn verify(&self, original: RoundValue<F>) -> bool {
        V::check(self.result, self.t, original)
    }

    pub fn append(&self, other: Self) -> Option<Self> {
        if other.verify(self.result) {
            Some(Self {
                result: other.result,
                t: self.t + other.t,
                _v: PhantomData::<V>,
            })
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_xorshift::XorShiftRng;

    #[test]
    fn test_exponents() {
        test_exponents_aux::<PallasVDF, pallas::Scalar>();
        test_exponents_aux::<VestaVDF, vesta::Scalar>();
    }

    fn test_exponents_aux<V: VDF<F>, F: FieldExt>() {
        assert_eq!(V::inverse_exponent(), 5);
        assert_eq!(V::inverse_exponent(), 5);
    }

    #[test]
    fn test_steps() {
        test_steps_aux::<PallasVDF, pallas::Scalar>();
        test_steps_aux::<VestaVDF, vesta::Scalar>();
    }

    fn test_steps_aux<V: VDF<F>, F: FieldExt>() {
        let mut rng = XorShiftRng::from_seed(TEST_SEED);

        for _ in 0..100 {
            let x = F::random(&mut rng);
            let y = V::forward_step(x);
            let z = V::inverse_step(y);

            assert_eq!(x, z);
        }
    }

    #[test]
    fn test_eval() {
        test_eval_aux::<PallasVDF, pallas::Scalar>();
        test_eval_aux::<VestaVDF, vesta::Scalar>();
    }

    fn test_eval_aux<V: VDF<F>, F: FieldExt>() {
        let mut rng = XorShiftRng::from_seed(TEST_SEED);

        for _ in 0..10 {
            let t = 10;
            let value = F::random(&mut rng);
            let round = F::random(&mut rng);
            let x = RoundValue { value, round };
            let y = V::eval(x, t);
            let z = V::inverse_eval(y, t);

            assert_eq!(x, z);
            assert!(V::check(y, t, x));
        }
    }

    #[test]
    fn test_vanilla_proof() {
        test_vanilla_proof_aux::<PallasVDF, pallas::Scalar>();
        test_vanilla_proof_aux::<VestaVDF, vesta::Scalar>();
    }

    fn test_vanilla_proof_aux<V: VDF<F>, F: FieldExt>() {
        let mut rng = XorShiftRng::from_seed(TEST_SEED);

        let value = F::random(&mut rng);
        let round = F::zero();
        let x = RoundValue { value, round };
        let t = 12;
        let n = 11;

        let first_proof = VanillaVDFProof::<V, F>::eval_and_prove(x, t);

        let final_proof = (1..11).fold(first_proof, |acc, _| {
            let new_proof = VanillaVDFProof::<V, F>::eval_and_prove(acc.result, t);
            acc.append(new_proof).expect("failed to append proof")
        });

        assert_eq!(V::element(final_proof.t), final_proof.result.round);
        assert_eq!(n * t, final_proof.t);
        assert!(final_proof.verify(x));
    }
}
