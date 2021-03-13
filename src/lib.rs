use core::fmt::Debug;
use std::marker::PhantomData;

use halo2::arithmetic::FieldExt;
use halo2::pasta::{pallas, vesta};

pub const TEST_SEED: [u8; 16] = [42; 16];

// Question: Should the naming of `PallasVDF` and `VestaVDF` be reversed?

/// Modulus is that of `Fq`, which is the base field of `Vesta` and scalar field of `Pallas`.
#[derive(Debug)]
pub struct PallasVDF {}
impl RaguVDF<pallas::Scalar> for PallasVDF {
    fn element(n: u64) -> pallas::Scalar {
        pallas::Scalar::from(n)
    }

    /// Pallas' inverse_exponent is 5, so we can hardcode this.
    fn inverse_step(x: pallas::Scalar) -> pallas::Scalar {
        x.mul(&x.square().square())
    }

    fn forward_step(x: pallas::Scalar) -> pallas::Scalar {
        let sqr = |x: pallas::Scalar, i: u32| (0..i).fold(x, |x, _| x.square());

        let mul = |x: pallas::Scalar, y| x.mul(y);
        let sqr_mul = |x, n, y: pallas::Scalar| y.mul(&sqr(x, n));

        let q1 = x;
        let q10 = sqr(q1, 1);
        let q11 = mul(q10, &q1);
        let q101 = mul(q10, &q11);
        let q110 = sqr(q11, 1);
        let q111 = mul(q110, &q1);
        let q1001 = mul(q111, &q10);
        let q1111 = mul(q1001, &q110);
        let qr2 = sqr_mul(q110, 3, q11);
        let qr4 = sqr_mul(qr2, 8, qr2);
        let qr8 = sqr_mul(qr4, 16, qr4);
        let qr16 = sqr_mul(qr8, 32, qr8);
        let qr32 = sqr_mul(qr16, 64, qr16);
        let qr32a = sqr_mul(qr32, 5, q1001);
        let qr32b = sqr_mul(qr32a, 8, q111);
        let qr32c = sqr_mul(qr32b, 4, q1);
        let qr32d = sqr_mul(qr32c, 2, qr4);
        let qr32e = sqr_mul(qr32d, 7, q11);
        let qr32f = sqr_mul(qr32e, 6, q1001);
        let qr32g = sqr_mul(qr32f, 3, q101);
        let qr32h = sqr_mul(qr32g, 7, q101);
        let qr32i = sqr_mul(qr32h, 7, q111);
        let qr32j = sqr_mul(qr32i, 4, q111);
        let qr32k = sqr_mul(qr32j, 5, q1001);
        let qr32l = sqr_mul(qr32k, 5, q101);
        let qr32m = sqr_mul(qr32l, 3, q11);
        let qr32n = sqr_mul(qr32m, 4, q101);
        let qr32o = sqr_mul(qr32n, 3, q101);
        let qr32p = sqr_mul(qr32o, 6, q1111);
        let qr32q = sqr_mul(qr32p, 4, q1001);
        let qr32r = sqr_mul(qr32q, 6, q101);
        let qr32s = sqr_mul(qr32r, 37, qr8);
        let qr32t = sqr_mul(qr32s, 2, q1);
        qr32t
    }
}

/// Modulus is that of `Fp`, which is the base field of `Pallas and scalar field of Vesta.
#[derive(Debug)]
pub struct VestaVDF {}
impl RaguVDF<vesta::Scalar> for VestaVDF {
    fn element(n: u64) -> vesta::Scalar {
        vesta::Scalar::from(n)
    }
    /// Vesta's inverse_exponent is 5, so we can hardcode this.
    fn inverse_step(x: vesta::Scalar) -> vesta::Scalar {
        x.mul(&x.square().square())
    }
    fn forward_step(x: vesta::Scalar) -> vesta::Scalar {
        let sqr = |x: vesta::Scalar, i: u32| (0..i).fold(x, |x, _| x.square());

        let mul = |x: vesta::Scalar, y| x.mul(y);
        let sqr_mul = |x, n, y: vesta::Scalar| y.mul(&sqr(x, n));

        let p1 = x;
        let p10 = sqr(p1, 1);
        let p11 = mul(p10, &p1);
        let p101 = mul(p10, &p11);
        let p110 = sqr(p11, 1);
        let p111 = mul(p110, &p1);
        let p1001 = mul(p111, &p10);
        let p1111 = mul(p1001, &p110);
        let pr2 = sqr_mul(p110, 3, p11);
        let pr4 = sqr_mul(pr2, 8, pr2);
        let pr8 = sqr_mul(pr4, 16, pr4);
        let pr16 = sqr_mul(pr8, 32, pr8);
        let pr32 = sqr_mul(pr16, 64, pr16);
        let pr32a = sqr_mul(pr32, 5, p1001);
        let pr32b = sqr_mul(pr32a, 8, p111);
        let pr32c = sqr_mul(pr32b, 4, p1);
        let pr32d = sqr_mul(pr32c, 2, pr4);
        let pr32e = sqr_mul(pr32d, 7, p11);
        let pr32f = sqr_mul(pr32e, 6, p1001);
        let pr32g = sqr_mul(pr32f, 3, p101);
        let pr32h = sqr_mul(pr32g, 5, p1);
        let pr32i = sqr_mul(pr32h, 7, p101);
        let pr32j = sqr_mul(pr32i, 4, p11);
        let pr32k = sqr_mul(pr32j, 8, p111);
        let pr32l = sqr_mul(pr32k, 4, p1);
        let pr32m = sqr_mul(pr32l, 4, p111);
        let pr32n = sqr_mul(pr32m, 9, p1111);
        let pr32o = sqr_mul(pr32n, 8, p1111);
        let pr32p = sqr_mul(pr32o, 6, p1111);
        let pr32q = sqr_mul(pr32p, 2, p11);
        let pr32r = sqr_mul(pr32q, 34, pr8);
        let pr32s = sqr_mul(pr32r, 2, p1);
        pr32s
    }
}

// Question: Is this right, or is it the reverse? Which scalar fields' modulus do we want to target?
pub type TargetVDF = PallasVDF;

#[derive(std::cmp::PartialEq, Debug, Clone, Copy)]
pub struct RoundValue<T> {
    pub value: T,
    pub round: T,
}

pub trait RaguVDF<F>: Debug
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

    /// one round in the slow/forward direction.
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
pub struct VanillaVDFProof<V: RaguVDF<F> + Debug, F: FieldExt> {
    result: RoundValue<F>,
    t: u64,
    _v: PhantomData<V>,
}

impl<V: RaguVDF<F>, F: FieldExt> VanillaVDFProof<V, F> {
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

    fn test_exponents_aux<V: RaguVDF<F>, F: FieldExt>() {
        assert_eq!(V::inverse_exponent(), 5);
        assert_eq!(V::inverse_exponent(), 5);
    }

    #[test]
    fn test_steps() {
        test_steps_aux::<PallasVDF, pallas::Scalar>();
        test_steps_aux::<VestaVDF, vesta::Scalar>();
    }

    fn test_steps_aux<V: RaguVDF<F>, F: FieldExt>() {
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

    fn test_eval_aux<V: RaguVDF<F>, F: FieldExt>() {
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

    fn test_vanilla_proof_aux<V: RaguVDF<F>, F: FieldExt>() {
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
