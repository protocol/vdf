use merlin::Transcript;
use nova::traits::{ChallengeTrait, CompressedGroup, Group, PrimeField};
use pasta_curves::arithmetic::{CurveExt, FieldExt, Group as Grp};
use pasta_curves::group::GroupEncoding;
use pasta_curves::{self, pallas, Ep, Fq};
use rand::{CryptoRng, RngCore};
use std::borrow::Borrow;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PallasPoint(pallas::Point);

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PallasScalar(pallas::Scalar);

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PallasCompressedPoint(<pallas::Point as GroupEncoding>::Repr);

impl Group for PallasPoint {
    type Scalar = PallasScalar;
    type CompressedGroupElement = PallasCompressedPoint;

    fn vartime_multiscalar_mul<I, J>(scalars: I, points: J) -> Self
    where
        I: IntoIterator,
        I::Item: Borrow<Self::Scalar>,
        J: IntoIterator,
        J::Item: Borrow<Self>,
        Self: Clone,
    {
        // Unoptimized.
        scalars
            .into_iter()
            .zip(points)
            .map(|(scalar, point)| (*point.borrow()).mul(*scalar.borrow()))
            .fold(PallasPoint(Ep::group_zero()), |acc, x| acc + x)
    }

    fn compress(&self) -> Self::CompressedGroupElement {
        PallasCompressedPoint(self.0.to_bytes())
    }

    fn from_uniform_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != 64 {
            dbg!(bytes.len());
            None
        } else {
            let mut arr = [0; 32];
            arr.copy_from_slice(&bytes[0..32]);
            dbg!(&arr);

            let hash = Ep::hash_to_curve("from_uniform_bytes");
            Some(Self(hash(&arr)))
        }
    }
}

impl PrimeField for PallasScalar {
    fn zero() -> Self {
        Self(Fq::zero())
    }
    fn one() -> Self {
        Self(Fq::one())
    }
    fn from_bytes_mod_order_wide(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != 64 {
            None
        } else {
            let mut arr = [0; 64];
            arr.copy_from_slice(&bytes[0..64]);
            Some(Self(Fq::from_bytes_wide(&arr)))
        }
    }

    fn random(_rng: &mut (impl RngCore + CryptoRng)) -> Self {
        Self(Fq::rand())
    }
}

impl ChallengeTrait for PallasScalar {
    fn challenge(label: &'static [u8], transcript: &mut Transcript) -> Self {
        let mut buf = [0u8; 64];
        transcript.challenge_bytes(label, &mut buf);
        PallasScalar::from_bytes_mod_order_wide(&buf).unwrap()
    }
}

impl CompressedGroup for PallasCompressedPoint {
    type GroupElement = PallasPoint;
    fn decompress(&self) -> Option<<Self as CompressedGroup>::GroupElement> {
        Some(PallasPoint(Ep::from_bytes(&self.0).unwrap()))
    }
    fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

impl Add<PallasPoint> for PallasPoint {
    type Output = PallasPoint;

    fn add(self, x: PallasPoint) -> PallasPoint {
        Self(self.0.add(x.0))
    }
}

impl<'r> Add<&'r PallasPoint> for PallasPoint {
    type Output = PallasPoint;

    fn add(self, x: &PallasPoint) -> PallasPoint {
        Self(self.0.add(x.0))
    }
}

impl AddAssign<PallasPoint> for PallasPoint {
    fn add_assign(&mut self, x: PallasPoint) {
        self.0.add_assign(x.0);
    }
}

impl<'r> AddAssign<&'r PallasPoint> for PallasPoint {
    fn add_assign(&mut self, x: &PallasPoint) {
        self.0.add_assign(x.0);
    }
}

impl Sub<PallasPoint> for PallasPoint {
    type Output = PallasPoint;

    fn sub(self, x: PallasPoint) -> PallasPoint {
        Self(self.0.sub(x.0))
    }
}

impl<'r> Sub<&'r PallasPoint> for PallasPoint {
    type Output = PallasPoint;

    fn sub(self, x: &PallasPoint) -> PallasPoint {
        Self(self.0.sub(x.0))
    }
}
impl SubAssign<PallasPoint> for PallasPoint {
    fn sub_assign(&mut self, x: PallasPoint) {
        self.0.sub_assign(x.0);
    }
}
impl<'r> SubAssign<&'r PallasPoint> for PallasPoint {
    fn sub_assign(&mut self, x: &PallasPoint) {
        self.0.sub_assign(x.0);
    }
}

impl Mul<PallasScalar> for PallasPoint {
    type Output = PallasPoint;

    fn mul(self, x: PallasScalar) -> PallasPoint {
        Self(self.0.mul(x.0))
    }
}

impl<'r> Mul<&'r PallasScalar> for PallasPoint {
    type Output = PallasPoint;

    fn mul(self, x: &PallasScalar) -> PallasPoint {
        Self(self.0.mul(x.0))
    }
}

impl MulAssign<PallasScalar> for PallasPoint {
    fn mul_assign(&mut self, x: PallasScalar) {
        self.0.mul_assign(x.0);
    }
}
impl<'r> MulAssign<&'r PallasScalar> for PallasPoint {
    fn mul_assign(&mut self, x: &PallasScalar) {
        self.0.mul_assign(x.0);
    }
}

impl Add<PallasScalar> for PallasScalar {
    type Output = PallasScalar;

    fn add(self, x: PallasScalar) -> PallasScalar {
        Self(self.0.add(x.0))
    }
}

impl<'r> Add<&'r PallasScalar> for PallasScalar {
    type Output = PallasScalar;

    fn add(self, x: &PallasScalar) -> PallasScalar {
        Self(self.0.add(x.0))
    }
}

impl AddAssign<PallasScalar> for PallasScalar {
    fn add_assign(&mut self, x: PallasScalar) {
        self.0.add_assign(x.0);
    }
}

impl<'r> AddAssign<&'r PallasScalar> for PallasScalar {
    fn add_assign(&mut self, x: &PallasScalar) {
        self.0.add_assign(x.0);
    }
}

impl Sub<PallasScalar> for PallasScalar {
    type Output = PallasScalar;

    fn sub(self, x: PallasScalar) -> PallasScalar {
        Self(self.0.sub(x.0))
    }
}

impl<'r> Sub<&'r PallasScalar> for PallasScalar {
    type Output = PallasScalar;

    fn sub(self, x: &PallasScalar) -> PallasScalar {
        Self(self.0.sub(x.0))
    }
}
impl SubAssign<PallasScalar> for PallasScalar {
    fn sub_assign(&mut self, x: PallasScalar) {
        self.0.sub_assign(x.0)
    }
}
impl<'r> SubAssign<&'r PallasScalar> for PallasScalar {
    fn sub_assign(&mut self, x: &PallasScalar) {
        self.0.sub_assign(x.0)
    }
}

impl Mul<PallasScalar> for PallasScalar {
    type Output = PallasScalar;

    fn mul(self, x: PallasScalar) -> PallasScalar {
        Self(self.0.mul(x.0))
    }
}

impl<'r> Mul<&'r PallasScalar> for PallasScalar {
    type Output = PallasScalar;

    fn mul(self, x: &PallasScalar) -> PallasScalar {
        Self(self.0.mul(x.0))
    }
}

impl MulAssign<PallasScalar> for PallasScalar {
    fn mul_assign(&mut self, x: PallasScalar) {
        self.0.mul_assign(x.0)
    }
}
impl<'r> MulAssign<&'r PallasScalar> for PallasScalar {
    fn mul_assign(&mut self, x: &PallasScalar) {
        self.0.mul_assign(x.0)
    }
}

impl Neg for PallasScalar {
    type Output = Self;
    fn neg(self) -> Self {
        Self(self.0.neg())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use nova::r1cs::*;
    use nova::traits::PrimeField;
    use nova::*;
    use rand::rngs::OsRng;

    type S = PallasScalar;
    type G = PallasPoint;

    #[test]
    #[allow(non_snake_case)]
    fn test_tiny_r1cs() {
        let one = S::one();
        let (num_cons, num_vars, num_inputs, A, B, C) = {
            let num_cons = 4;
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

            // constraint 0 entries in (A,B,C)
            A.push((0, 0, one));
            B.push((0, 0, one));
            C.push((0, 1, one));

            // constraint 1 entries in (A,B,C)
            A.push((1, 1, one));
            B.push((1, 0, one));
            C.push((1, 2, one));

            // constraint 2 entries in (A,B,C)
            A.push((2, 2, one));
            A.push((2, 0, one));
            B.push((2, num_vars, one));
            C.push((2, 3, one));

            // constraint 3 entries in (A,B,C)
            A.push((3, 3, one));
            A.push((3, num_vars, one + one + one + one + one));
            B.push((3, num_vars, one));
            C.push((3, num_vars + 1, one));

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
        let mut prover_transcript = Transcript::new(b"StepSNARKExample");
        let res = StepSNARK::prove(&gens, &S, &U1, &W1, &U2, &W2, &mut prover_transcript);
        assert!(res.is_ok());
        let (step_snark, (_U, W)) = res.unwrap();

        // verify the step SNARK
        let mut verifier_transcript = Transcript::new(b"StepSNARKExample");
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
}
